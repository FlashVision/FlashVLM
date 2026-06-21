"""Tests for FlashVLM task modules."""

import pytest

from flashvlm.analytics.metrics import (
    compute_bleu,
    compute_cider,
    compute_meteor,
    vqa_accuracy,
)
from flashvlm.tasks.grounding import GroundingTask
from flashvlm.tasks.vqa import VQATask


class TestGroundingTask:
    def test_parse_bbox_brackets(self):
        task = GroundingTask(model=None)
        bbox = task._parse_bbox("[0.1, 0.2, 0.5, 0.6]")
        assert bbox is not None
        assert len(bbox) == 4
        assert abs(bbox[0] - 0.1) < 1e-6

    def test_parse_bbox_no_brackets(self):
        task = GroundingTask(model=None)
        bbox = task._parse_bbox("0.1 0.2 0.5 0.6")
        assert bbox is not None
        assert len(bbox) == 4

    def test_parse_bbox_large_values(self):
        task = GroundingTask(model=None)
        bbox = task._parse_bbox("[100, 200, 500, 600]")
        assert bbox is not None
        assert all(0 <= c <= 1 for c in bbox)

    def test_parse_bbox_invalid(self):
        task = GroundingTask(model=None)
        bbox = task._parse_bbox("no coordinates here")
        assert bbox is None

    def test_compute_iou_identical(self):
        iou = GroundingTask._compute_iou([0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5])
        assert abs(iou - 1.0) < 1e-6

    def test_compute_iou_no_overlap(self):
        iou = GroundingTask._compute_iou([0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 1.0, 1.0])
        assert iou == 0.0

    def test_compute_iou_partial(self):
        iou = GroundingTask._compute_iou([0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75])
        assert 0.0 < iou < 1.0

    def test_evaluate(self):
        task = GroundingTask(model=None)
        predictions = [[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7], None]
        ground_truths = [[0.1, 0.1, 0.5, 0.5], [0.0, 0.0, 0.3, 0.3], [0.5, 0.5, 1.0, 1.0]]
        metrics = task.evaluate(predictions, ground_truths)
        assert "accuracy@0.5" in metrics
        assert "mean_iou" in metrics
        assert "parse_rate" in metrics


class TestVQATask:
    def test_postprocess_answer(self):
        task = VQATask(model=None)
        assert task._postprocess_answer("  yes \n") == "yes"
        assert task._postprocess_answer("A red car\nQuestion: what") == "A red car"

    def test_evaluate_exact_match(self):
        task = VQATask(model=None)
        preds = ["yes", "cat", "blue"]
        gts = ["yes", "cat", "red"]
        metrics = task.evaluate(preds, gts)
        assert metrics["accuracy"] == pytest.approx(2 / 3, abs=1e-6)

    def test_evaluate_case_insensitive(self):
        task = VQATask(model=None)
        preds = ["YES", "Cat"]
        gts = ["yes", "cat"]
        metrics = task.evaluate(preds, gts)
        assert metrics["accuracy"] == 1.0


class TestMetrics:
    def test_vqa_accuracy_perfect(self):
        preds = ["yes", "no", "cat"]
        gts = [["yes", "yes", "yes"], ["no", "no", "no"], ["cat", "cat", "cat"]]
        acc = vqa_accuracy(preds, gts)
        assert acc == 1.0

    def test_vqa_accuracy_zero(self):
        preds = ["abc", "xyz"]
        gts = [["yes", "yes"], ["no", "no"]]
        acc = vqa_accuracy(preds, gts)
        assert acc == 0.0

    def test_bleu_identical(self):
        preds = ["the cat sat on the mat"]
        refs = [["the cat sat on the mat"]]
        score = compute_bleu(preds, refs)
        assert score > 0.9

    def test_bleu_no_overlap(self):
        preds = ["apple banana cherry"]
        refs = [["one two three four five"]]
        score = compute_bleu(preds, refs)
        assert score == 0.0

    def test_cider_identical(self):
        preds = ["a dog running in the park"]
        refs = [["a dog running in the park"]]
        score = compute_cider(preds, refs)
        assert score > 0.0

    def test_meteor_identical(self):
        preds = ["the quick brown fox"]
        refs = [["the quick brown fox"]]
        score = compute_meteor(preds, refs)
        assert score > 0.9

    def test_meteor_no_overlap(self):
        preds = ["abc def ghi"]
        refs = [["xyz uvw rst"]]
        score = compute_meteor(preds, refs)
        assert score == 0.0
