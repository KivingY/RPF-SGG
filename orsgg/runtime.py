"""Small runtime helpers used by the public ReCon1M evaluator."""

from __future__ import annotations

import torch
import torch.distributed as dist

from orsgg.data.recon1m import load_recon1m_cache
from orsgg.evaluation import SGGMeanRecallAccumulator


def predicate_score_weights(
    cache_path: str,
    *,
    max_images: int | None,
    num_predicates: int,
    power: float,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.ones(num_predicates + 1, dtype=torch.float32, device=device)
    if power <= 0:
        return weights
    dataset = load_recon1m_cache(cache_path)
    samples = dataset.samples[:max_images] if max_images else dataset.samples
    counts = torch.ones(num_predicates + 1, dtype=torch.float32)
    for sample in samples:
        if sample.relations.numel() == 0:
            continue
        predicates = sample.relations[:, 2].long().cpu()
        valid = (predicates > 0) & (predicates <= num_predicates)
        if bool(valid.any()):
            counts.scatter_add_(
                0,
                predicates[valid],
                torch.ones_like(predicates[valid], dtype=torch.float32),
            )
    positive = counts[1:].pow(-power)
    weights[1:] = (positive / positive.mean().clamp_min(1e-6)).to(device)
    return weights


def distributed_sgg_summary(
    accumulator: SGGMeanRecallAccumulator,
    *,
    num_images: int,
    device: torch.device,
) -> dict[str, float]:
    counts = torch.tensor(
        [accumulator.total_gt, num_images, accumulator.recall_image_count],
        dtype=torch.long,
        device=device,
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    image_recall_sum = torch.tensor(
        [accumulator.image_recall_sum[k] for k in accumulator.ks],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(image_recall_sum, op=dist.ReduceOp.SUM)
    class_count = accumulator.class_recall_count.to(device=device, dtype=torch.long)
    class_sum = torch.stack(
        [accumulator.class_recall_sum[k] for k in accumulator.ks], dim=0
    ).to(device=device, dtype=torch.float64)
    dist.all_reduce(class_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(class_sum, op=dist.ReduceOp.SUM)

    valid = class_count > 0
    valid[0] = False
    summary: dict[str, float] = {}
    for index, k in enumerate(accumulator.ks):
        summary[f"R@{k}"] = float(image_recall_sum[index] / counts[2].clamp_min(1))
        per_class = torch.zeros_like(class_sum[index])
        per_class[valid] = class_sum[index, valid] / class_count[valid].double()
        summary[f"mR@{k}"] = float(per_class[valid].mean()) if bool(valid.any()) else 0.0
    summary["eval_images"] = float(counts[1])
    summary["recall_images"] = float(counts[2])
    summary["mean_recall_classes"] = float(valid.sum())
    return summary
