from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from orsgg.data.recon1m import ReCon1MSplit, load_recon1m_cache
from orsgg.structures import SceneGraphTarget


class ReCon1MImageDataset(Dataset):
    """Image-level ReCon1M dataset used by the public SGDet evaluator."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        max_size: int | None = 1024,
        random_scales: Sequence[int] | None = None,
        max_images: int | None = None,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
    ) -> None:
        self.dataset: ReCon1MSplit = load_recon1m_cache(cache_path)
        self.samples = self.dataset.samples[:max_images] if max_images else self.dataset.samples
        self.max_size = max_size
        self.random_scales = tuple(int(scale) for scale in random_scales or () if int(scale) > 0)
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_max_size(self) -> int | None:
        if not self.random_scales:
            return self.max_size
        index = int(torch.randint(len(self.random_scales), (), dtype=torch.long).item())
        return self.random_scales[index]

    def _resize(
        self,
        image: Image.Image,
        target: SceneGraphTarget,
        *,
        max_size: int | None,
    ) -> tuple[Image.Image, SceneGraphTarget]:
        if max_size is None:
            return image, target
        width, height = image.size
        longest = max(width, height)
        if longest <= max_size:
            return image, target
        scale = max_size / float(longest)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        boxes = target.boxes.clone()
        boxes[:, 0::2] *= scale
        boxes[:, 1::2] *= scale
        return image, replace(target, width=new_w, height=new_h, boxes=boxes)

    def _hflip(self, image: Image.Image, target: SceneGraphTarget) -> tuple[Image.Image, SceneGraphTarget]:
        image = ImageOps.mirror(image)
        boxes = target.boxes.clone()
        boxes[:, 0::2] = float(target.width) - boxes[:, 0::2]
        return image, replace(target, boxes=boxes)

    def _vflip(self, image: Image.Image, target: SceneGraphTarget) -> tuple[Image.Image, SceneGraphTarget]:
        image = ImageOps.flip(image)
        boxes = target.boxes.clone()
        boxes[:, 1::2] = float(target.height) - boxes[:, 1::2]
        return image, replace(target, boxes=boxes)

    def __getitem__(self, index: int | tuple[int, int]) -> Dict[str, Any]:
        item_max_size = self._sample_max_size()
        if isinstance(index, tuple):
            index, item_max_size = int(index[0]), int(index[1])
        target = self.samples[index]
        image = Image.open(target.image_path).convert("RGB")
        image, target = self._resize(image, target, max_size=item_max_size)
        if self.hflip_prob > 0 and float(torch.rand(()).item()) < self.hflip_prob:
            image, target = self._hflip(image, target)
        if self.vflip_prob > 0 and float(torch.rand(()).item()) < self.vflip_prob:
            image, target = self._vflip(image, target)
        tensor = F.to_tensor(image)
        return {"image": tensor, "target": target}


def pad_images_collate(
    batch: Sequence[Dict[str, Any]],
    *,
    size_divisor: int = 32,
) -> Dict[str, Any]:
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    max_h = max(int(img.shape[1]) for img in images)
    max_w = max(int(img.shape[2]) for img in images)
    if size_divisor > 1:
        max_h = (max_h + size_divisor - 1) // size_divisor * size_divisor
        max_w = (max_w + size_divisor - 1) // size_divisor * size_divisor
    padded = images[0].new_zeros((len(images), 3, max_h, max_w))
    mask = torch.ones((len(images), max_h, max_w), dtype=torch.bool)
    for i, image in enumerate(images):
        _, h, w = image.shape
        padded[i, :, :h, :w] = image
        mask[i, :h, :w] = False
    return {"images": padded, "mask": mask, "targets": targets}
