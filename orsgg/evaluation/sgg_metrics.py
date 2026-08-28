from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import torch

from orsgg.ops import obb_iou
from orsgg.structures import SceneGraphPrediction, SceneGraphTarget

EvalMode = Literal["predcls", "sgcls", "sgdet"]


@dataclass(frozen=True)
class Triplet:
    subj: int
    obj: int
    pred: int
    score: float


def _prediction_triplets(
    pred: SceneGraphPrediction,
    *,
    graph_constraint: bool,
    predicate_offset: int = 0,
) -> List[Triplet]:
    pairs = pred.rel_pairs.detach().cpu().long()
    scores = pred.rel_scores.detach().cpu().float()
    obj_scores = pred.object_scores.detach().cpu().float() if pred.object_scores is not None else None
    triplets: List[Triplet] = []

    if scores.ndim != 2:
        raise ValueError(f"rel_scores must be [R,C], got {tuple(scores.shape)}")
    if len(pairs) != len(scores):
        raise ValueError("rel_pairs and rel_scores length mismatch")

    positive_start = 1 if scores.shape[1] > 1 and predicate_offset == 0 else 0
    for pair_idx, (subj, obj) in enumerate(pairs.tolist()):
        row = scores[pair_idx]
        if graph_constraint:
            pos_scores = row[positive_start:]
            rel_col = int(torch.argmax(pos_scores).item()) + positive_start
            rel_score = float(row[rel_col].item())
            rel_id = rel_col + predicate_offset
            if rel_id == 0:
                continue
            score = rel_score
            if obj_scores is not None:
                score *= float(obj_scores[subj].item()) * float(obj_scores[obj].item())
            triplets.append(Triplet(subj, obj, rel_id, score))
        else:
            for rel_col in range(positive_start, scores.shape[1]):
                rel_id = rel_col + predicate_offset
                if rel_id == 0:
                    continue
                score = float(row[rel_col].item())
                if obj_scores is not None:
                    score *= float(obj_scores[subj].item()) * float(obj_scores[obj].item())
                triplets.append(Triplet(subj, obj, rel_id, score))
    triplets.sort(key=lambda x: x.score, reverse=True)
    return triplets


def _gt_totals_by_predicate(gt: SceneGraphTarget, num_predicates: int) -> torch.Tensor:
    totals = torch.zeros(num_predicates + 1, dtype=torch.long)
    if gt.relations.numel() == 0:
        return totals
    preds = gt.relations[:, 2].long().cpu()
    for p in preds.tolist():
        if 0 <= p <= num_predicates:
            totals[p] += 1
    return totals


def _match_predcls_or_sgcls(
    gt: SceneGraphTarget,
    pred: SceneGraphPrediction,
    triplets: Sequence[Triplet],
    k: int,
    *,
    mode: EvalMode,
) -> torch.Tensor:
    gt_rel = gt.relations.detach().cpu().long()
    gt_labels = gt.labels.detach().cpu().long()
    pred_labels = pred.labels.detach().cpu().long()
    hit = torch.zeros(len(gt_rel), dtype=torch.bool)
    gt_lookup: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, (s, o, p) in enumerate(gt_rel.tolist()):
        gt_lookup.setdefault((s, o, p), []).append(idx)

    for t in triplets[:k]:
        if t.subj >= len(gt_labels) or t.obj >= len(gt_labels):
            continue
        if mode == "sgcls":
            if t.subj >= len(pred_labels) or t.obj >= len(pred_labels):
                continue
            if int(pred_labels[t.subj]) != int(gt_labels[t.subj]):
                continue
            if int(pred_labels[t.obj]) != int(gt_labels[t.obj]):
                continue
        for idx in gt_lookup.get((t.subj, t.obj, t.pred), []):
            hit[idx] = True
    return hit


def _match_sgdet(
    gt: SceneGraphTarget,
    pred: SceneGraphPrediction,
    triplets: Sequence[Triplet],
    k: int,
    *,
    ious: torch.Tensor,
    iou_thr: float,
) -> torch.Tensor:
    gt_rel = gt.relations.detach().cpu().long()
    gt_labels = gt.labels.detach().cpu().long()
    pred_labels = pred.labels.detach().cpu().long()
    hit = torch.zeros(len(gt_rel), dtype=torch.bool)
    if len(gt_rel) == 0 or pred.boxes.numel() == 0:
        return hit
    for t in triplets[:k]:
        if t.subj >= len(pred_labels) or t.obj >= len(pred_labels) or t.subj == t.obj:
            continue
        subj_label = int(pred_labels[t.subj])
        obj_label = int(pred_labels[t.obj])
        for gt_idx, (gs, go, gp) in enumerate(gt_rel.tolist()):
            if hit[gt_idx] or gp != t.pred:
                continue
            if subj_label != int(gt_labels[gs]) or obj_label != int(gt_labels[go]):
                continue
            if float(ious[t.subj, gs]) >= iou_thr and float(ious[t.obj, go]) >= iou_thr:
                hit[gt_idx] = True
    return hit


def evaluate_scene_graph(
    gt: SceneGraphTarget,
    pred: SceneGraphPrediction,
    *,
    mode: EvalMode,
    ks: Sequence[int] = (20, 50, 100),
    num_predicates: int = 59,
    iou_thr: float = 0.5,
    graph_constraint: bool = True,
    predicate_offset: int = 0,
) -> Dict[int, torch.Tensor]:
    """Return hit mask per K for one image.

    The caller can aggregate hit masks for R@K and mR@K. For PredCls/SGCls, GT
    object indices are used. For SGDet, predicted objects are matched to GT OBBs.
    """

    if mode not in {"predcls", "sgcls", "sgdet"}:
        raise ValueError(f"Unsupported mode: {mode}")
    triplets = _prediction_triplets(
        pred, graph_constraint=graph_constraint, predicate_offset=predicate_offset
    )
    out: Dict[int, torch.Tensor] = {}
    if mode in {"predcls", "sgcls"}:
        for k in ks:
            out[k] = _match_predcls_or_sgcls(gt, pred, triplets, k, mode=mode)
    else:
        if pred.boxes.numel() == 0 or gt.boxes.numel() == 0:
            ious = torch.zeros((len(pred.boxes), len(gt.boxes)), dtype=torch.float32)
        else:
            gt_boxes = gt.boxes.to(device=pred.boxes.device)
            ious = obb_iou(pred.boxes, gt_boxes).detach().cpu()
        for k in ks:
            out[k] = _match_sgdet(gt, pred, triplets, k, iou_thr=iou_thr, ious=ious)
    return out


def fast_evaluate_sgdet_graph_constraint(
    gt: SceneGraphTarget,
    pred: SceneGraphPrediction,
    *,
    ks: Sequence[int] = (20, 50, 100),
    num_predicates: int = 59,
    iou_thr: float = 0.5,
    predicate_offset: int = 0,
) -> Dict[int, torch.Tensor]:
    """Vectorized SGDet evaluator for graph-constrained predictions.

    This is equivalent to `evaluate_scene_graph(..., mode="sgdet",
    graph_constraint=True)` for the common SGDet setting, but avoids Python
    triplet materialization and nested top-K/GT loops. It is intended for full
    validation runs where each image can have thousands of candidate pairs.
    """

    gt_rel = gt.relations.detach().cpu().long()
    num_gt = len(gt_rel)
    out = {int(k): torch.zeros(num_gt, dtype=torch.bool) for k in ks}
    if num_gt == 0 or pred.boxes.numel() == 0 or pred.rel_pairs.numel() == 0:
        return out

    device = pred.boxes.device
    rel_pairs = pred.rel_pairs.to(device=device, dtype=torch.long)
    rel_scores = pred.rel_scores.to(device=device, dtype=torch.float32)
    pred_labels = pred.labels.to(device=device, dtype=torch.long)
    gt_boxes = gt.boxes.to(device=device, dtype=torch.float32)
    gt_labels = gt.labels.to(device=device, dtype=torch.long)
    gt_rel_dev = gt_rel.to(device=device)

    if rel_scores.ndim != 2:
        raise ValueError(f"rel_scores must be [R,C], got {tuple(rel_scores.shape)}")
    if len(rel_pairs) != len(rel_scores):
        raise ValueError("rel_pairs and rel_scores length mismatch")
    if len(rel_pairs) == 0 or rel_scores.numel() == 0:
        return out

    valid_pair = (rel_pairs[:, 0] >= 0) & (rel_pairs[:, 0] < len(pred_labels))
    valid_pair = valid_pair & (rel_pairs[:, 1] >= 0) & (rel_pairs[:, 1] < len(pred_labels))
    valid_pair = valid_pair & (rel_pairs[:, 0] != rel_pairs[:, 1])
    if not bool(valid_pair.any()):
        return out
    rel_pairs = rel_pairs[valid_pair]
    rel_scores = rel_scores[valid_pair]

    positive_start = 1 if rel_scores.shape[1] > 1 and predicate_offset == 0 else 0
    if rel_scores.shape[1] <= positive_start:
        return out
    pos_scores = rel_scores[:, positive_start:]
    best_scores, best_cols = pos_scores.max(dim=1)
    pred_ids = best_cols + positive_start + predicate_offset

    object_scores = pred.object_scores
    if object_scores is None:
        pair_object_scores = best_scores.new_ones((len(rel_pairs),))
    else:
        object_scores = object_scores.to(device=device, dtype=torch.float32).clamp_min(1e-6)
        pair_object_scores = object_scores[rel_pairs[:, 0]] * object_scores[rel_pairs[:, 1]]
    triplet_scores = best_scores * pair_object_scores
    top_n = min(max(int(k) for k in ks), len(triplet_scores))
    if top_n <= 0:
        return out
    order = torch.topk(triplet_scores, k=top_n).indices

    ious = obb_iou(pred.boxes.to(device=device, dtype=torch.float32), gt_boxes)
    label_match = pred_labels[:, None] == gt_labels[None, :]
    object_match = (ious >= iou_thr) & label_match
    gt_subj = gt_rel_dev[:, 0].clamp(min=0, max=max(len(gt_labels) - 1, 0))
    gt_obj = gt_rel_dev[:, 1].clamp(min=0, max=max(len(gt_labels) - 1, 0))
    endpoint_match = object_match[rel_pairs[order, 0]][:, gt_subj]
    endpoint_match = endpoint_match & object_match[rel_pairs[order, 1]][:, gt_obj]
    predicate_match = gt_rel_dev[:, 2][None, :] == pred_ids[order, None]
    matches = endpoint_match & predicate_match
    for k in ks:
        k_eff = min(int(k), top_n)
        out[int(k)] = matches[:k_eff].any(dim=0).detach().cpu()
    return out


class SGGMeanRecallAccumulator:
    """Aggregate SGG R@K and mR@K with image-level macro averaging.

    R@K is computed per image first and then averaged across images that have
    at least one GT relation. mR@K follows the common SGG mean-recall protocol:
    for every image, each predicate class present in that image contributes one
    class recall value; those values are averaged per class and then across
    predicate classes. The count-style fields are kept for diagnostics only.
    """

    def __init__(self, ks: Iterable[int], num_predicates: int = 59) -> None:
        self.ks = tuple(ks)
        self.num_predicates = num_predicates
        self.recall_image_count = 0
        self.image_recall_sum = {k: 0.0 for k in self.ks}
        self.class_recall_count = torch.zeros(num_predicates + 1, dtype=torch.long)
        self.class_recall_sum = {
            k: torch.zeros(num_predicates + 1, dtype=torch.float64) for k in self.ks
        }

        # Micro-style counters retained for debugging and legacy diagnostics.
        self.total_gt = 0
        self.hit_total = {k: 0 for k in self.ks}
        self.gt_per_pred = torch.zeros(num_predicates + 1, dtype=torch.long)
        self.hit_per_pred = {
            k: torch.zeros(num_predicates + 1, dtype=torch.long) for k in self.ks
        }

    def update(self, gt: SceneGraphTarget, hit_masks: Dict[int, torch.Tensor]) -> None:
        gt_rel = gt.relations.detach().cpu().long()
        num_gt = len(gt_rel)
        self.total_gt += num_gt
        gt_per_pred_img = _gt_totals_by_predicate(gt, self.num_predicates)
        self.gt_per_pred += gt_per_pred_img
        valid_pred_img = gt_per_pred_img > 0
        valid_pred_img[0] = False
        if num_gt > 0:
            self.recall_image_count += 1
            self.class_recall_count[valid_pred_img] += 1
        for k in self.ks:
            hit = hit_masks[k].detach().cpu().bool()
            if len(hit) != num_gt:
                raise ValueError(f"hit mask for K={k} has length {len(hit)}, expected {num_gt}")
            hit_count = int(hit.sum().item())
            if num_gt > 0:
                self.image_recall_sum[k] += float(hit_count) / float(num_gt)
            hit_per_pred_img = torch.zeros(self.num_predicates + 1, dtype=torch.long)
            self.hit_total[k] += int(hit.sum().item())
            for idx, is_hit in enumerate(hit.tolist()):
                if is_hit:
                    pred_id = int(gt_rel[idx, 2].item())
                    if 0 <= pred_id <= self.num_predicates:
                        hit_per_pred_img[pred_id] += 1
                        self.hit_per_pred[k][pred_id] += 1
            if bool(valid_pred_img.any()):
                recalls = torch.zeros(self.num_predicates + 1, dtype=torch.float64)
                recalls[valid_pred_img] = hit_per_pred_img[valid_pred_img].double() / gt_per_pred_img[
                    valid_pred_img
                ].double()
                self.class_recall_sum[k][valid_pred_img] += recalls[valid_pred_img]

    def summarize(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        valid_pred = self.class_recall_count > 0
        valid_pred[0] = False
        for k in self.ks:
            result[f"R@{k}"] = (
                0.0 if self.recall_image_count == 0 else self.image_recall_sum[k] / self.recall_image_count
            )
            recalls = torch.zeros(self.num_predicates + 1, dtype=torch.float64)
            recalls[valid_pred] = self.class_recall_sum[k][valid_pred] / self.class_recall_count[
                valid_pred
            ].double()
            result[f"mR@{k}"] = 0.0 if not bool(valid_pred.any()) else float(recalls[valid_pred].mean().item())
        result["recall_images"] = float(self.recall_image_count)
        result["mean_recall_classes"] = float(int(valid_pred.sum().item()))
        return result
