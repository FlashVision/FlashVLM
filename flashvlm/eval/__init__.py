"""FlashVLM evaluation benchmarks: VQAv2, TextVQA, MMBench, POPE."""

from flashvlm.eval.evaluator import (
    MMBenchEvaluator,
    POPEEvaluator,
    TextVQAEvaluator,
    VQAv2Evaluator,
    run_evaluation,
)

__all__ = [
    "VQAv2Evaluator",
    "TextVQAEvaluator",
    "MMBenchEvaluator",
    "POPEEvaluator",
    "run_evaluation",
]
