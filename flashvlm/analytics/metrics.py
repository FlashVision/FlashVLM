"""Evaluation metrics for VLM tasks: VQA accuracy, BLEU, CIDEr, METEOR."""

from __future__ import annotations

import math
from collections import Counter


def vqa_accuracy(
    predictions: list[str],
    ground_truths: list[list[str]],
    num_annotators: int = 10,
) -> float:
    """Compute VQA accuracy following the standard VQAv2 metric.

    accuracy = min(count(answer) / 3, 1) where count is the number of
    annotators who gave the same answer.

    Args:
        predictions: Model predictions.
        ground_truths: List of annotator answers per question.
        num_annotators: Expected number of annotators.

    Returns:
        Average VQA accuracy score.
    """
    total_accuracy = 0.0

    for pred, gt_answers in zip(predictions, ground_truths):
        pred_normalized = _normalize_answer(pred)
        gt_normalized = [_normalize_answer(a) for a in gt_answers]

        count = sum(1 for a in gt_normalized if a == pred_normalized)
        accuracy = min(count / 3.0, 1.0)
        total_accuracy += accuracy

    return total_accuracy / max(len(predictions), 1)


def compute_bleu(
    predictions: list[str],
    references: list[list[str]],
    max_n: int = 4,
    weights: list[float] | None = None,
) -> float:
    """Compute BLEU score (up to n-gram).

    Args:
        predictions: Generated texts.
        references: List of reference text lists per sample.
        max_n: Maximum n-gram order.
        weights: Weights for each n-gram precision (uniform if None).

    Returns:
        Corpus-level BLEU score.
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    precisions = []
    bp_c = 0
    bp_r = 0

    for n in range(1, max_n + 1):
        matches = 0
        total = 0

        for pred, refs in zip(predictions, references):
            pred_tokens = pred.lower().split()
            pred_ngrams = _get_ngrams(pred_tokens, n)
            total += max(len(pred_tokens) - n + 1, 0)

            max_ref_counts: Counter = Counter()
            for ref in refs:
                ref_tokens = ref.lower().split()
                ref_ngrams = _get_ngrams(ref_tokens, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            clipped = {
                ngram: min(count, max_ref_counts.get(ngram, 0))
                for ngram, count in pred_ngrams.items()
            }
            matches += sum(clipped.values())

            if n == 1:
                bp_c += len(pred_tokens)
                best_ref_len = (
                    min(
                        (abs(len(pred_tokens) - len(ref.split())), len(ref.split())) for ref in refs
                    )[1]
                    if refs
                    else 0
                )
                bp_r += best_ref_len

        precision = matches / max(total, 1)
        precisions.append(precision)

    if any(p == 0 for p in precisions):
        return 0.0

    log_bleu = sum(w * math.log(p) for w, p in zip(weights, precisions))

    if bp_c < bp_r:
        bp = math.exp(1 - bp_r / max(bp_c, 1))
    else:
        bp = 1.0

    return bp * math.exp(log_bleu)


def compute_cider(
    predictions: list[str],
    references: list[list[str]],
    n: int = 4,
) -> float:
    """Compute CIDEr-D score (simplified implementation).

    Args:
        predictions: Generated captions.
        references: List of reference caption lists.
        n: Maximum n-gram length.

    Returns:
        CIDEr-D score.
    """
    num_docs = len(predictions)
    if num_docs == 0:
        return 0.0

    df: Counter = Counter()
    for refs in references:
        seen: set = set()
        for ref in refs:
            tokens = ref.lower().split()
            for i in range(1, n + 1):
                for ngram in _get_ngrams(tokens, i):
                    if ngram not in seen:
                        df[ngram] += 1
                        seen.add(ngram)

    total_score = 0.0
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.lower().split()
        pred_vec = _compute_tfidf(pred_tokens, df, num_docs, n)

        ref_scores = []
        for ref in refs:
            ref_tokens = ref.lower().split()
            ref_vec = _compute_tfidf(ref_tokens, df, num_docs, n)
            sim = _cosine_similarity(pred_vec, ref_vec)
            ref_scores.append(sim)

        total_score += sum(ref_scores) / max(len(ref_scores), 1)

    return 10.0 * total_score / max(num_docs, 1)


def compute_meteor(
    predictions: list[str],
    references: list[list[str]],
) -> float:
    """Compute METEOR score (simplified word-matching version).

    Args:
        predictions: Generated texts.
        references: Reference text lists.

    Returns:
        Average METEOR score.
    """
    total_score = 0.0

    for pred, refs in zip(predictions, references):
        pred_words = set(pred.lower().split())
        best_score = 0.0

        for ref in refs:
            ref_words = set(ref.lower().split())
            matches = pred_words & ref_words

            if not matches:
                continue

            precision = len(matches) / max(len(pred_words), 1)
            recall = len(matches) / max(len(ref_words), 1)

            if precision + recall == 0:
                continue

            f1 = (10 * precision * recall) / (9 * precision + recall)

            chunks = _count_chunks(pred.lower().split(), ref.lower().split(), matches)
            penalty = 0.5 * (chunks / max(len(matches), 1)) ** 3

            score = f1 * (1 - penalty)
            best_score = max(best_score, score)

        total_score += best_score

    return total_score / max(len(predictions), 1)


def _normalize_answer(answer: str) -> str:
    """Normalize VQA answer for evaluation."""
    answer = answer.lower().strip()
    answer = answer.rstrip(".")
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    for word, digit in number_words.items():
        answer = answer.replace(word, digit)
    articles = ["a ", "an ", "the "]
    for article in articles:
        if answer.startswith(article):
            answer = answer[len(article) :]
    return answer.strip()


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Extract n-grams from a token list."""
    ngrams: Counter = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams[ngram] += 1
    return ngrams


def _compute_tfidf(tokens: list[str], df: Counter, num_docs: int, max_n: int) -> dict[tuple, float]:
    """Compute TF-IDF vector for CIDEr."""
    vec: dict[tuple, float] = {}
    for n in range(1, max_n + 1):
        ngrams = _get_ngrams(tokens, n)
        for ngram, tf in ngrams.items():
            idf = math.log(max(num_docs, 1) / max(df.get(ngram, 0) + 1, 1))
            vec[ngram] = tf * idf
    return vec


def _cosine_similarity(vec1: dict[tuple, float], vec2: dict[tuple, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    keys = set(vec1.keys()) | set(vec2.keys())
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _count_chunks(pred_tokens: list[str], ref_tokens: list[str], matches: set) -> int:
    """Count contiguous matched chunks for METEOR penalty."""
    in_chunk = False
    chunks = 0
    for token in pred_tokens:
        if token in matches:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False
    return chunks
