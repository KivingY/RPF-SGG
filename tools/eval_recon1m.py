#!/usr/bin/env python
"""Evaluate the released RPF-SGG checkpoint on the ReCon1M SGDet test set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orsgg.data import ReCon1MImageDataset, pad_images_collate
from orsgg.evaluation import SGGMeanRecallAccumulator, fast_evaluate_sgdet_graph_constraint
from orsgg.modeling import ORSGGDirectedNet
from orsgg.runtime import distributed_sgg_summary, predicate_score_weights


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def init_distributed(device_arg: str) -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed evaluation requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    return distributed, rank, world_size, device


def load_model(checkpoint_path: Path, device: torch.device) -> ORSGGDirectedNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model_config = dict(checkpoint["model_config"])
    # The release checkpoint contains the complete backbone. Constructing it
    # from config avoids a network request to the gated upstream checkpoint.
    model_config["detector_dinov3_pretrained"] = False
    model = ORSGGDirectedNet(**model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "recon1m_eval.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--test-cache", type=Path, default=None)
    parser.add_argument("--train-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "recon1m_sgdet.json")
    parser.add_argument("--max-images", type=int, default=None, help="Debug only; omit for paper results.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    checkpoint_path = args.checkpoint or ROOT / config["checkpoint"]
    test_cache = args.test_cache or ROOT / data_cfg["test_cache"]
    train_cache = args.train_cache or ROOT / data_cfg["train_cache"]

    torch.multiprocessing.set_sharing_strategy("file_system")
    distributed, rank, world_size, device = init_distributed(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = ReCon1MImageDataset(
        test_cache,
        max_size=int(eval_cfg["max_size"]),
        max_images=args.max_images,
    )
    eval_dataset = (
        Subset(dataset, list(range(rank, len(dataset), world_size)))
        if distributed
        else dataset
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=int(eval_cfg["batch_size_per_gpu"]),
        shuffle=False,
        num_workers=int(eval_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(eval_cfg["num_workers"]) > 0,
        collate_fn=pad_images_collate,
    )

    model = load_model(checkpoint_path, device)
    prior_weights = predicate_score_weights(
        str(train_cache),
        max_images=None,
        num_predicates=model.num_predicates,
        power=float(eval_cfg["predicate_prior_power"]),
        device=device,
    )
    ks = tuple(int(k) for k in eval_cfg["ks"])
    accumulator = SGGMeanRecallAccumulator(ks, num_predicates=model.num_predicates)
    num_images = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"]
        image_sizes = [(target.width, target.height) for target in targets]
        predictions = model.predict(
            images,
            image_sizes=image_sizes,
            top_objects=int(eval_cfg["top_objects"]),
            max_pairs=int(eval_cfg["max_pairs"]),
            score_thr=0.0,
            object_score_power=float(eval_cfg["object_score_power"]),
            binding_topk=int(eval_cfg["binding_topk"]),
            predicate_temperature=float(eval_cfg["predicate_temperature"]),
            quality_power=float(eval_cfg["quality_power"]),
            binding_score_power=float(eval_cfg["binding_score_power"]),
            target_aware_blend=1.0,
            predicate_score_weights=prior_weights,
        )
        for target, prediction in zip(targets, predictions):
            hits = fast_evaluate_sgdet_graph_constraint(
                target,
                prediction,
                ks=ks,
                num_predicates=model.num_predicates,
                iou_thr=float(eval_cfg["iou_threshold"]),
            )
            accumulator.update(target, hits)
            num_images += 1
        if rank == 0 and batch_index % 50 == 0:
            print(f"evaluated {num_images} local images", flush=True)

    if distributed:
        summary = distributed_sgg_summary(accumulator, num_images=num_images, device=device)
    else:
        summary = accumulator.summarize()
        summary["eval_images"] = float(num_images)

    if rank == 0:
        metrics = {
            key: round(value * 100.0, 4) if key.startswith(("R@", "mR@")) else value
            for key, value in summary.items()
        }
        result = {
            "dataset": "ReCon1M",
            "task": "SGDet",
            "metrics_percent": metrics,
        }
        print(json.dumps(result, indent=2))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"saved {args.output}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
