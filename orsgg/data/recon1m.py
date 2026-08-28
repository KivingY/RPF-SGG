from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from orsgg.structures import SceneGraphTarget


OBJECT_CATEGORIES: Dict[int, str] = {}
REL_CATEGORIES: List[str] = []


@dataclass
class ReCon1MSplit:
    root: Path
    split: str
    categories: Dict[int, str]
    rel_categories: List[str]
    samples: List[SceneGraphTarget]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[SceneGraphTarget]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> SceneGraphTarget:
        return self.samples[index]


def _split_json_name(split: str) -> str:
    return f"{split}_r.json"


def _load_flags_from_annfile(path: Path) -> torch.Tensor:
    flags: List[int] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Malformed annotation line in {path}: {line!r}")
        flags.append(int(float(fields[-1])))
    return torch.tensor(flags, dtype=torch.long)


def load_recon1m_split(
    root: str | Path,
    split: str,
    *,
    load_flags: bool = False,
    max_images: Optional[int] = None,
) -> ReCon1MSplit:
    """Load one ReCon1M-cropped split from aggregate JSON files.

    The loader keeps annotation order unchanged because relation triplets refer
    to 0-based object row indices within each image.
    """

    root = Path(root)
    split_dir = root / split
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    ann_json = json.loads((split_dir / _split_json_name(split)).read_text())
    rel_json = json.loads((split_dir / "rel_r.json").read_text())
    rel_by_image = rel_json[split]

    categories = {int(c["id"]): c["name"] for c in ann_json["categories"]}
    rel_categories = list(rel_json["rel_categories"])
    global OBJECT_CATEGORIES, REL_CATEGORIES
    OBJECT_CATEGORIES = categories
    REL_CATEGORIES = rel_categories

    images = ann_json["images"]
    if max_images is not None:
        images = images[:max_images]
    image_ids = {int(im["id"]) for im in images}

    anns_by_image: Dict[int, List[dict]] = {image_id: [] for image_id in image_ids}
    for ann in ann_json["annotations"]:
        image_id = int(ann["image_id"])
        if image_id in anns_by_image:
            anns_by_image[image_id].append(ann)

    samples: List[SceneGraphTarget] = []
    for im in images:
        image_id = int(im["id"])
        file_name = im["file_name"]
        anns = anns_by_image[image_id]
        boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32)
        labels = torch.tensor([int(ann["category_id"]) for ann in anns], dtype=torch.long)
        relations = torch.tensor(rel_by_image[str(image_id)], dtype=torch.long)
        if relations.numel() == 0:
            relations = relations.reshape(0, 3)

        flags = None
        if load_flags:
            annfile = split_dir / "annfiles" / f"{Path(file_name).stem}.txt"
            flags = _load_flags_from_annfile(annfile)
            if len(flags) != len(labels):
                raise ValueError(
                    f"Flag count mismatch for {file_name}: {len(flags)} != {len(labels)}"
                )

        samples.append(
            SceneGraphTarget(
                image_id=image_id,
                file_name=file_name,
                image_path=split_dir / "images" / file_name,
                width=int(im["width"]),
                height=int(im["height"]),
                boxes=boxes,
                labels=labels,
                relations=relations,
                flags=flags,
            )
        )

    return ReCon1MSplit(
        root=root,
        split=split,
        categories=categories,
        rel_categories=rel_categories,
        samples=samples,
    )


def save_recon1m_cache(dataset: ReCon1MSplit, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_recon1m_cache(path: str | Path) -> ReCon1MSplit:
    with Path(path).open("rb") as f:
        return pickle.load(f)
