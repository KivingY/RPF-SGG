from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class SceneGraphTarget:
    """Ground-truth scene graph for one image.

    Attributes:
        image_id: ReCon1M patch id.
        file_name: Image filename relative to the split image directory.
        image_path: Absolute path to the image.
        width: Image width.
        height: Image height.
        boxes: Tensor [N, 8] with OBB corners.
        labels: Tensor [N] with 1-based object category ids.
        relations: Tensor [M, 3] as subject index, object index, predicate id.
        flags: Optional tensor [N] from DOTA-style txt annotation. Flag 2
            corresponds to OBBs crossing crop bounds.
    """

    image_id: int
    file_name: str
    image_path: Path
    width: int
    height: int
    boxes: torch.Tensor
    labels: torch.Tensor
    relations: torch.Tensor
    flags: Optional[torch.Tensor] = None


@dataclass
class SceneGraphPrediction:
    """Predicted scene graph for one image.

    rel_scores can be either:
        [R, C + 1], where column 0 is background and positive predicates use
        dataset ids 1..C, or
        [R, C], if predicate_offset=1 is supplied to the evaluator.
    """

    boxes: torch.Tensor
    labels: torch.Tensor
    object_scores: Optional[torch.Tensor]
    rel_pairs: torch.Tensor
    rel_scores: torch.Tensor
