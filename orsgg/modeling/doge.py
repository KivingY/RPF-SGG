from __future__ import annotations

import torch


def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def obb_parameters(boxes: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract differentiable-ish OBB parameters from 8-point boxes.

    ReCon1M annotations are ordered quadrilaterals. We use opposite
    edge averages to estimate long/short sides and orientation. The function is
    intended for geometry features and evaluation utilities, not as a canonical
    OBB conversion for loss computation.
    """

    if boxes.ndim != 2 or boxes.shape[1] != 8:
        raise ValueError(f"Expected [N,8] boxes, got {tuple(boxes.shape)}")
    pts = boxes.reshape(-1, 4, 2)
    center = pts.mean(dim=1)
    next_pts = torch.roll(pts, shifts=-1, dims=1)
    edges = next_pts - pts
    edge_lengths = torch.linalg.norm(edges, dim=-1).clamp_min(1e-6)

    len_a = 0.5 * (edge_lengths[:, 0] + edge_lengths[:, 2])
    len_b = 0.5 * (edge_lengths[:, 1] + edge_lengths[:, 3])
    use_a = len_a >= len_b
    long_len = torch.where(use_a, len_a, len_b)
    short_len = torch.where(use_a, len_b, len_a)
    long_vec = torch.where(use_a[:, None], edges[:, 0], edges[:, 1])
    theta = torch.atan2(long_vec[:, 1], long_vec[:, 0])

    x = pts[:, :, 0]
    y = pts[:, :, 1]
    area = 0.5 * torch.abs((x * torch.roll(y, shifts=-1, dims=1) - y * torch.roll(x, shifts=-1, dims=1)).sum(dim=1))

    return {
        "center": center,
        "long": long_len,
        "short": short_len,
        "theta": theta,
        "area": area.clamp_min(1e-6),
    }


def directed_obb_geometry(
    boxes: torch.Tensor,
    pairs: torch.Tensor,
    *,
    image_size: tuple[int, int] | None = None,
    flags: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build directed OBB geometry features for subject-object pairs.

    Args:
        boxes: [N, 8] OBB corners.
        pairs: [R, 2] long tensor, subject/object indices.
        image_size: Optional `(width, height)` for normalization.
        flags: Optional [N] annotation flags. `flag=2` indicates crop-crossing
            OBBs.

    Returns:
        Tensor [R, 28] geometry features.
    """

    if pairs.numel() == 0:
        return boxes.new_zeros((0, 28))
    pairs = pairs.long()
    params = obb_parameters(boxes)
    s = pairs[:, 0]
    o = pairs[:, 1]
    cs = params["center"][s]
    co = params["center"][o]
    long_s = params["long"][s]
    long_o = params["long"][o]
    short_s = params["short"][s]
    short_o = params["short"][o]
    area_s = params["area"][s]
    area_o = params["area"][o]
    theta_s = params["theta"][s]
    theta_o = params["theta"][o]

    if image_size is None:
        max_xy = boxes.reshape(-1, 2).amax(dim=0).clamp_min(1.0)
        width, height = max_xy[0], max_xy[1]
    else:
        width = boxes.new_tensor(float(image_size[0])).clamp_min(1.0)
        height = boxes.new_tensor(float(image_size[1])).clamp_min(1.0)
    diag = torch.sqrt(width * width + height * height).clamp_min(1.0)

    delta = co - cs
    dx = delta[:, 0] / width
    dy = delta[:, 1] / height
    dist = torch.linalg.norm(delta, dim=1) / diag
    pair_angle = torch.atan2(delta[:, 1], delta[:, 0])
    dtheta = theta_o - theta_s
    subj_axis_to_pair = pair_angle - theta_s
    obj_axis_to_pair = pair_angle - theta_o

    flag_s = boxes.new_zeros(len(pairs))
    flag_o = boxes.new_zeros(len(pairs))
    if flags is not None:
        flags = flags.to(device=boxes.device)
        flag_s = (flags[s] == 2).to(dtype=boxes.dtype)
        flag_o = (flags[o] == 2).to(dtype=boxes.dtype)

    feats = [
        cs[:, 0] / width,
        cs[:, 1] / height,
        co[:, 0] / width,
        co[:, 1] / height,
        dx,
        dy,
        dist,
        _safe_log(long_s / long_o),
        _safe_log(short_s / short_o),
        _safe_log(area_s / area_o),
        long_s / diag,
        short_s / diag,
        long_o / diag,
        short_o / diag,
        torch.sin(theta_s),
        torch.cos(theta_s),
        torch.sin(theta_o),
        torch.cos(theta_o),
        torch.sin(dtheta),
        torch.cos(dtheta),
        torch.sin(pair_angle),
        torch.cos(pair_angle),
        torch.sin(subj_axis_to_pair),
        torch.cos(subj_axis_to_pair),
        torch.sin(obj_axis_to_pair),
        torch.cos(obj_axis_to_pair),
        flag_s,
        flag_o,
    ]
    return torch.stack(feats, dim=1)
