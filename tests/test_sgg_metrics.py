from __future__ import annotations

import unittest
from pathlib import Path

import torch

from orsgg.evaluation import SGGMeanRecallAccumulator, evaluate_scene_graph
from orsgg.evaluation.sgg_metrics import fast_evaluate_sgdet_graph_constraint
from orsgg.ops.rotated_iou import obb5_to_poly
from orsgg.structures import SceneGraphPrediction, SceneGraphTarget


def _target() -> SceneGraphTarget:
    boxes = obb5_to_poly(
        torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0, 0.0],
                [5.0, 0.0, 2.0, 2.0, 0.0],
                [10.0, 0.0, 2.0, 2.0, 0.0],
            ]
        )
    )
    labels = torch.tensor([1, 2, 3])
    relations = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return SceneGraphTarget(
        image_id=1,
        file_name="dummy.png",
        image_path=Path("dummy.png"),
        width=16,
        height=8,
        boxes=boxes,
        labels=labels,
        relations=relations,
    )


class TestSGGMetrics(unittest.TestCase):
    def test_predcls_recall(self) -> None:
        gt = _target()
        rel_pairs = torch.tensor([[0, 1], [1, 2]])
        scores = torch.zeros((2, 4))
        scores[0, 2] = 0.9
        scores[1, 1] = 0.8
        pred = SceneGraphPrediction(
            boxes=gt.boxes,
            labels=gt.labels,
            object_scores=None,
            rel_pairs=rel_pairs,
            rel_scores=scores,
        )
        hits = evaluate_scene_graph(gt, pred, mode="predcls", ks=(1, 2), num_predicates=3)
        self.assertEqual(int(hits[1].sum()), 1)
        self.assertEqual(int(hits[2].sum()), 1)

    def test_sgcls_label_check(self) -> None:
        gt = _target()
        scores = torch.zeros((2, 4))
        scores[0, 2] = 0.9
        scores[1, 3] = 0.8
        pred = SceneGraphPrediction(
            boxes=gt.boxes,
            labels=torch.tensor([1, 2, 99]),
            object_scores=None,
            rel_pairs=torch.tensor([[0, 1], [1, 2]]),
            rel_scores=scores,
        )
        hits = evaluate_scene_graph(gt, pred, mode="sgcls", ks=(2,), num_predicates=3)
        self.assertEqual(int(hits[2].sum()), 1)

    def test_sgdet_iou_match_and_mr(self) -> None:
        gt = _target()
        scores = torch.zeros((2, 4))
        scores[0, 2] = 0.9
        scores[1, 3] = 0.8
        pred = SceneGraphPrediction(
            boxes=gt.boxes.clone(),
            labels=gt.labels.clone(),
            object_scores=torch.ones(3),
            rel_pairs=torch.tensor([[0, 1], [1, 2]]),
            rel_scores=scores,
        )
        hits = evaluate_scene_graph(gt, pred, mode="sgdet", ks=(1, 2), num_predicates=3)
        acc = SGGMeanRecallAccumulator((1, 2), num_predicates=3)
        acc.update(gt, hits)
        summary = acc.summarize()
        self.assertAlmostEqual(summary["R@1"], 0.5)
        self.assertAlmostEqual(summary["R@2"], 1.0)
        self.assertAlmostEqual(summary["mR@1"], 0.5)
        self.assertAlmostEqual(summary["mR@2"], 1.0)

    def test_accumulator_uses_image_macro_not_dataset_micro(self) -> None:
        gt1 = _target()
        gt2 = SceneGraphTarget(
            image_id=2,
            file_name="dummy2.png",
            image_path=Path("dummy2.png"),
            width=gt1.width,
            height=gt1.height,
            boxes=gt1.boxes,
            labels=gt1.labels,
            relations=torch.tensor([[0, 1, 2]] * 10, dtype=torch.long),
        )
        acc = SGGMeanRecallAccumulator((1,), num_predicates=3)
        acc.update(gt1, {1: torch.tensor([True, False])})
        acc.update(gt2, {1: torch.zeros(10, dtype=torch.bool)})
        summary = acc.summarize()
        self.assertAlmostEqual(summary["R@1"], 0.25)
        self.assertAlmostEqual(summary["mR@1"], 0.25)
        self.assertAlmostEqual(summary["recall_images"], 2.0)
        self.assertAlmostEqual(summary["mean_recall_classes"], 2.0)

    def test_perfect_predcls_reaches_one_when_k_covers_all_gt(self) -> None:
        gt = _target()
        pairs = gt.relations[:, :2]
        scores = torch.zeros((len(pairs), 4), dtype=torch.float32)
        for idx, rel in enumerate(gt.relations):
            scores[idx, int(rel[2])] = 1.0
        pred = SceneGraphPrediction(
            boxes=gt.boxes,
            labels=gt.labels,
            object_scores=None,
            rel_pairs=pairs,
            rel_scores=scores,
        )
        hits = evaluate_scene_graph(gt, pred, mode="predcls", ks=(len(pairs),), num_predicates=3)
        acc = SGGMeanRecallAccumulator((len(pairs),), num_predicates=3)
        acc.update(gt, hits)
        summary = acc.summarize()
        self.assertAlmostEqual(summary[f"R@{len(pairs)}"], 1.0)
        self.assertAlmostEqual(summary[f"mR@{len(pairs)}"], 1.0)

    def test_no_graph_constraint_allows_multiple_predicates_per_pair(self) -> None:
        gt = _target()
        scores = torch.zeros((1, 4), dtype=torch.float32)
        scores[0, 2] = 0.8
        scores[0, 3] = 0.9
        pred = SceneGraphPrediction(
            boxes=gt.boxes,
            labels=gt.labels,
            object_scores=None,
            rel_pairs=torch.tensor([[0, 1]]),
            rel_scores=scores,
        )
        hits_graph = evaluate_scene_graph(
            gt, pred, mode="predcls", ks=(2,), num_predicates=3, graph_constraint=True
        )
        hits_nograph = evaluate_scene_graph(
            gt, pred, mode="predcls", ks=(2,), num_predicates=3, graph_constraint=False
        )
        self.assertEqual(int(hits_graph[2].sum()), 0)
        self.assertEqual(int(hits_nograph[2].sum()), 1)

    def test_duplicate_predictions_do_not_double_count_gt(self) -> None:
        gt = _target()
        scores = torch.zeros((2, 4), dtype=torch.float32)
        scores[:, 2] = torch.tensor([0.9, 0.8])
        pred = SceneGraphPrediction(
            boxes=gt.boxes,
            labels=gt.labels,
            object_scores=torch.ones(3),
            rel_pairs=torch.tensor([[0, 1], [0, 1]]),
            rel_scores=scores,
        )
        hits = evaluate_scene_graph(gt, pred, mode="sgdet", ks=(2,), num_predicates=3)
        self.assertEqual(int(hits[2].sum()), 1)

    def test_fast_sgdet_matches_reference_graph_constraint(self) -> None:
        gt = _target()
        scores = torch.zeros((4, 4), dtype=torch.float32)
        scores[0, 2] = 0.9
        scores[1, 3] = 0.7
        scores[2, 1] = 0.95
        scores[3, 2] = 0.4
        pred = SceneGraphPrediction(
            boxes=gt.boxes.clone(),
            labels=gt.labels.clone(),
            object_scores=torch.tensor([0.9, 0.8, 0.7]),
            rel_pairs=torch.tensor([[0, 1], [1, 2], [2, 1], [0, 1]]),
            rel_scores=scores,
        )
        slow = evaluate_scene_graph(
            gt,
            pred,
            mode="sgdet",
            ks=(1, 2, 4),
            num_predicates=3,
            graph_constraint=True,
        )
        fast = fast_evaluate_sgdet_graph_constraint(
            gt,
            pred,
            ks=(1, 2, 4),
            num_predicates=3,
        )
        for k in (1, 2, 4):
            self.assertTrue(torch.equal(slow[k], fast[k]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_sgdet_allows_prediction_boxes_on_cuda(self) -> None:
        gt = _target()
        scores = torch.zeros((2, 4), dtype=torch.float32, device="cuda")
        scores[0, 2] = 0.9
        scores[1, 3] = 0.8
        pred = SceneGraphPrediction(
            boxes=gt.boxes.clone().cuda(),
            labels=gt.labels.clone().cuda(),
            object_scores=torch.ones(3, device="cuda"),
            rel_pairs=torch.tensor([[0, 1], [1, 2]], device="cuda"),
            rel_scores=scores,
        )
        hits = evaluate_scene_graph(gt, pred, mode="sgdet", ks=(1, 2), num_predicates=3)
        self.assertEqual(int(hits[2].sum()), 2)


if __name__ == "__main__":
    unittest.main()
