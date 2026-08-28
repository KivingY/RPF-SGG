from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch

_EXT = None
_EXT_ERROR: Optional[Exception] = None


def obb5_to_poly(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h, theta] boxes to [x1,y1,...,x4,y4] polygons.

    theta is in radians. Corners are returned in counter-clockwise order in the
    image coordinate system convention used by the trigonometric formula.
    """

    if boxes.numel() == 0:
        return boxes.new_zeros((boxes.shape[0], 8))
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    dx = w * 0.5
    dy = h * 0.5
    rel = boxes.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=boxes.dtype)
    x_rel = rel[:, 0].view(1, 4) * dx.view(-1, 1)
    y_rel = rel[:, 1].view(1, 4) * dy.view(-1, 1)
    xs = cx.view(-1, 1) + x_rel * cos_t.view(-1, 1) - y_rel * sin_t.view(-1, 1)
    ys = cy.view(-1, 1) + x_rel * sin_t.view(-1, 1) + y_rel * cos_t.view(-1, 1)
    return torch.stack([xs, ys], dim=-1).reshape(-1, 8)


def _as_poly(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[1] not in {5, 8}:
        raise ValueError(f"Expected boxes with shape [N,5] or [N,8], got {tuple(boxes.shape)}")
    if boxes.shape[1] == 8:
        return boxes
    return obb5_to_poly(boxes)


def _sort_poly(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _poly_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    return abs(area) * 0.5


def _inside(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> bool:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-9


def _line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def _clip_polygon(
    subject: list[tuple[float, float]],
    clip: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    output = subject
    for i, a in enumerate(clip):
        b = clip[(i + 1) % len(clip)]
        input_poly = output
        output = []
        if not input_poly:
            break
        s = input_poly[-1]
        for e in input_poly:
            if _inside(e, a, b):
                if not _inside(s, a, b):
                    output.append(_line_intersection(s, e, a, b))
                output.append(e)
            elif _inside(s, a, b):
                output.append(_line_intersection(s, e, a, b))
            s = e
    return output


def obb_iou_python(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Slow, explicit pairwise OBB IoU reference implementation."""

    b1 = _as_poly(boxes1).detach().cpu().double()
    b2 = _as_poly(boxes2).detach().cpu().double()
    out = torch.zeros((b1.shape[0], b2.shape[0]), dtype=torch.float64)
    polys1 = [_sort_poly([(float(row[i]), float(row[i + 1])) for i in range(0, 8, 2)]) for row in b1]
    polys2 = [_sort_poly([(float(row[i]), float(row[i + 1])) for i in range(0, 8, 2)]) for row in b2]
    areas1 = [_poly_area(poly) for poly in polys1]
    areas2 = [_poly_area(poly) for poly in polys2]
    for i, poly1 in enumerate(polys1):
        for j, poly2 in enumerate(polys2):
            inter_poly = _clip_polygon(poly1, poly2)
            inter = _poly_area(inter_poly)
            union = areas1[i] + areas2[j] - inter
            out[i, j] = 0.0 if union <= 0 else inter / union
    return out.to(device=boxes1.device, dtype=boxes1.dtype)


def _load_ext():
    global _EXT, _EXT_ERROR
    if _EXT is not None or _EXT_ERROR is not None:
        return _EXT
    try:
        from torch.utils.cpp_extension import load

        python_bin = str(Path(sys.executable).resolve().parent)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if python_bin not in path_parts:
            os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

        this_dir = Path(__file__).resolve().parent
        sources = [
            str(this_dir / "csrc" / "rotated_iou_ext.cpp"),
            str(this_dir / "csrc" / "rotated_iou_cpu.cpp"),
        ]
        extra_cflags = ["-O3"]
        if torch.cuda.is_available():
            sources.append(str(this_dir / "csrc" / "rotated_iou_cuda.cu"))
            extra_cflags.append("-DWITH_CUDA")
        _EXT = load(
            name="orsgg_rotated_iou",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=["-O3"],
            verbose=bool(int(os.environ.get("ORSGG_VERBOSE_BUILD", "0"))),
        )
    except Exception as exc:  # pragma: no cover - exercised in missing-compiler envs
        _EXT_ERROR = exc
        _EXT = None
    return _EXT


def obb_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, *, use_ext: bool = True) -> torch.Tensor:
    """Pairwise OBB IoU with extension acceleration when available."""

    poly1 = _as_poly(boxes1).contiguous().to(dtype=torch.float32)
    poly2 = _as_poly(boxes2).contiguous().to(dtype=torch.float32)
    if use_ext:
        ext = _load_ext()
        if ext is not None:
            return ext.obb_iou(poly1, poly2)
    return obb_iou_python(poly1, poly2)


def extension_error() -> Optional[Exception]:
    _load_ext()
    return _EXT_ERROR
