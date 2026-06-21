"""Beam search generation for FlashVLM models."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


@dataclass
class BeamHypothesis:
    """A single beam hypothesis during search."""

    tokens: list[int] = field(default_factory=list)
    score: float = 0.0
    is_finished: bool = False

    @property
    def length(self) -> int:
        return len(self.tokens)

    def normalized_score(self, length_penalty: float = 1.0) -> float:
        """Length-normalized score."""
        if self.length == 0:
            return self.score
        return self.score / (self.length**length_penalty)


class BeamSearchGenerator:
    """Beam search text generation with length penalty and early stopping."""

    def __init__(
        self,
        num_beams: int = 4,
        max_length: int = 256,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ):
        self.num_beams = num_beams
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run beam search generation.

        Args:
            model: Language model with forward method returning logits.
            input_ids: Initial input token IDs (batch_size, seq_len).
            attention_mask: Optional attention mask.
            inputs_embeds: Optional pre-computed input embeddings.

        Returns:
            Generated token IDs (batch_size, generated_len).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        beam_input_ids = input_ids.repeat_interleave(self.num_beams, dim=0)
        if attention_mask is not None:
            beam_attention_mask = attention_mask.repeat_interleave(self.num_beams, dim=0)
        else:
            beam_attention_mask = None

        beam_scores = torch.zeros(batch_size * self.num_beams, device=device)
        beam_scores[1 :: self.num_beams] = float("-inf")

        finished_hypotheses: list[list[BeamHypothesis]] = [[] for _ in range(batch_size)]
        active_beams = torch.ones(batch_size * self.num_beams, dtype=torch.bool, device=device)

        for step in range(self.max_length):
            if not active_beams.any():
                break

            if hasattr(model, "forward"):
                try:
                    outputs = model(input_ids=beam_input_ids, attention_mask=beam_attention_mask)
                    if isinstance(outputs, dict):
                        logits = outputs.get("logits", outputs.get("last_hidden_state"))
                    else:
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                except (TypeError, AttributeError):
                    break
            else:
                break

            if logits is None:
                break

            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            if self.no_repeat_ngram_size > 0:
                log_probs = self._apply_no_repeat_ngram(
                    log_probs, beam_input_ids, self.no_repeat_ngram_size
                )

            next_scores = beam_scores.unsqueeze(-1) + log_probs
            vocab_size = next_scores.shape[-1]

            next_scores = next_scores.view(batch_size, self.num_beams * vocab_size)
            top_scores, top_indices = torch.topk(next_scores, 2 * self.num_beams, dim=-1)

            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            new_beam_input_ids = []
            new_beam_scores = torch.zeros(batch_size * self.num_beams, device=device)

            for batch_idx in range(batch_size):
                beam_count = 0
                for rank in range(2 * self.num_beams):
                    if beam_count >= self.num_beams:
                        break

                    beam_idx = beam_indices[batch_idx, rank]
                    token_id = token_indices[batch_idx, rank].item()
                    score = top_scores[batch_idx, rank].item()

                    global_beam_idx = batch_idx * self.num_beams + beam_idx

                    if token_id == self.eos_token_id:
                        hyp = BeamHypothesis(
                            tokens=beam_input_ids[global_beam_idx].tolist() + [token_id],
                            score=score,
                            is_finished=True,
                        )
                        finished_hypotheses[batch_idx].append(hyp)
                        finished = finished_hypotheses[batch_idx]
                        if self.early_stopping and len(finished) >= self.num_beams:
                            start = batch_idx * self.num_beams
                            end = (batch_idx + 1) * self.num_beams
                            active_beams[start:end] = False
                    else:
                        new_ids = torch.cat(
                            [
                                beam_input_ids[global_beam_idx].unsqueeze(0),
                                torch.tensor([[token_id]], device=device),
                            ],
                            dim=-1,
                        )
                        new_beam_input_ids.append(new_ids)
                        new_beam_scores[batch_idx * self.num_beams + beam_count] = score
                        beam_count += 1

                while beam_count < self.num_beams:
                    padding = torch.full(
                        (1, beam_input_ids.shape[1] + 1), self.pad_token_id, device=device
                    )
                    new_beam_input_ids.append(padding)
                    new_beam_scores[batch_idx * self.num_beams + beam_count] = float("-inf")
                    beam_count += 1

            if new_beam_input_ids:
                max_len = max(ids.shape[1] for ids in new_beam_input_ids)
                padded = []
                for ids in new_beam_input_ids:
                    if ids.shape[1] < max_len:
                        pad = torch.full(
                            (1, max_len - ids.shape[1]), self.pad_token_id, device=device
                        )
                        ids = torch.cat([ids, pad], dim=-1)
                    padded.append(ids)
                beam_input_ids = torch.cat(padded, dim=0)
                beam_scores = new_beam_scores

                if beam_attention_mask is not None:
                    beam_attention_mask = (beam_input_ids != self.pad_token_id).long()

        best_sequences = []
        for batch_idx in range(batch_size):
            hypotheses = finished_hypotheses[batch_idx]
            if not hypotheses:
                best_ids = beam_input_ids[batch_idx * self.num_beams]
                best_sequences.append(best_ids)
            else:
                best = max(hypotheses, key=lambda h: h.normalized_score(self.length_penalty))
                best_sequences.append(torch.tensor(best.tokens, device=device))

        max_out_len = max(seq.shape[0] for seq in best_sequences)
        output = torch.full((batch_size, max_out_len), self.pad_token_id, device=device)
        for i, seq in enumerate(best_sequences):
            output[i, : seq.shape[0]] = seq

        return output

    def _apply_no_repeat_ngram(
        self, log_probs: torch.Tensor, input_ids: torch.Tensor, ngram_size: int
    ) -> torch.Tensor:
        """Prevent repeated n-grams in generated text."""
        batch_size = input_ids.shape[0]
        for batch_idx in range(batch_size):
            generated = input_ids[batch_idx].tolist()
            if len(generated) < ngram_size:
                continue

            for i in range(len(generated) - ngram_size + 1):
                ngram = tuple(generated[i : i + ngram_size - 1])
                suffix = tuple(generated[-(ngram_size - 1) :])
                if ngram == suffix and i + ngram_size - 1 < len(generated):
                    banned_token = generated[i + ngram_size - 1]
                    log_probs[batch_idx, banned_token] = float("-inf")

        return log_probs
