from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from orsgg.modeling.doge import obb_parameters
from orsgg.ops.rotated_iou import obb5_to_poly
from orsgg.structures import SceneGraphTarget


def _gaussian2d(radius: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))


def _draw_gaussian(heatmap: torch.Tensor, cls: int, cx: int, cy: int, radius: int) -> None:
    h, w = heatmap.shape[-2:]
    left = min(cx, radius)
    right = min(w - cx - 1, radius)
    top = min(cy, radius)
    bottom = min(h - cy - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    gaussian = _gaussian2d(radius, sigma=max(radius / 3, 1.0), device=heatmap.device)
    patch = heatmap[cls, cy - top : cy + bottom + 1, cx - left : cx + right + 1]
    gpatch = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    torch.maximum(patch, gpatch, out=patch)


def _centernet_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = logits.float()
    target = target.float()
    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1 - 1e-4)
    pos = target.eq(1).float()
    neg = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg
    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)
        pos_loss = pos_loss * weights
        num_pos = (pos * weights).sum().clamp_min(1.0)
    else:
        num_pos = pos.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def _peak_nms(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(heat, kernel_size=kernel, stride=1, padding=pad)
    return heat * pooled.eq(heat)


def build_detector_targets(
    targets: Sequence[SceneGraphTarget],
    *,
    feat_h: int,
    feat_w: int,
    stride: int,
    num_classes: int,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    min_object_scale: float = 0.0,
    max_object_scale: float = float("inf"),
    relation_object_weight_power: float = 0.0,
    small_object_weight_power: float = 0.0,
    small_object_ref_scale: float = 32.0,
    small_object_max_weight: float = 3.0,
) -> Dict[str, torch.Tensor]:
    batch = len(targets)
    heatmap = torch.zeros((batch, num_classes, feat_h, feat_w), device=device)
    reg = torch.zeros((batch, 6, feat_h, feat_w), device=device)
    box = torch.zeros((batch, 8, feat_h, feat_w), device=device)
    reg_mask = torch.zeros((batch, 1, feat_h, feat_w), device=device)
    reg_weight = torch.zeros((batch, 1, feat_h, feat_w), device=device)
    weights = class_weights.to(device=device, dtype=torch.float32) if class_weights is not None else None

    for b, target in enumerate(targets):
        boxes = target.boxes.to(device)
        labels = target.labels.to(device)
        params = obb_parameters(boxes)
        centers = params["center"] / stride
        long_side = params["long"] / stride
        short_side = params["short"] / stride
        theta = params["theta"]
        object_scale = torch.sqrt((params["long"] * params["short"]).clamp_min(1e-6))
        relation_counts = torch.zeros((len(labels),), dtype=torch.float32, device=device)
        if relation_object_weight_power > 0 and target.relations.numel() > 0:
            rel = target.relations.to(device=device, dtype=torch.long)
            valid = (rel[:, 0] >= 0) & (rel[:, 0] < len(labels))
            valid = valid & (rel[:, 1] >= 0) & (rel[:, 1] < len(labels))
            rel = rel[valid]
            if len(rel) > 0:
                ones = torch.ones((len(rel),), dtype=torch.float32, device=device)
                relation_counts.scatter_add_(0, rel[:, 0], ones)
                relation_counts.scatter_add_(0, rel[:, 1], ones)
        for i in range(len(labels)):
            scale_i = float(object_scale[i].item())
            if scale_i < float(min_object_scale) or scale_i >= float(max_object_scale):
                continue
            cx_f, cy_f = centers[i]
            cx = int(torch.floor(cx_f).item())
            cy = int(torch.floor(cy_f).item())
            if cx < 0 or cx >= feat_w or cy < 0 or cy >= feat_h:
                continue
            cls = int(labels[i].item()) - 1
            if cls < 0 or cls >= num_classes:
                continue
            radius = int(max(1, min(4, 0.5 * min(long_side[i].item(), short_side[i].item()))))
            _draw_gaussian(heatmap[b], cls, cx, cy, radius)
            reg[b, :, cy, cx] = torch.stack(
                [
                    cx_f - cx,
                    cy_f - cy,
                    torch.log(long_side[i].clamp_min(1e-4)),
                    torch.log(short_side[i].clamp_min(1e-4)),
                    torch.sin(theta[i]),
                    torch.cos(theta[i]),
                ]
            )
            canon_box5 = torch.stack(
                [
                    params["center"][i, 0],
                    params["center"][i, 1],
                    params["long"][i],
                    params["short"][i],
                    theta[i],
                ]
            )
            box[b, :, cy, cx] = obb5_to_poly(canon_box5[None])[0]
            reg_mask[b, 0, cy, cx] = 1.0
            obj_weight = boxes.new_tensor(1.0)
            if weights is not None:
                obj_weight = obj_weight * weights[cls]
            if relation_object_weight_power > 0:
                obj_weight = obj_weight * (1.0 + relation_counts[i]).pow(float(relation_object_weight_power))
            if small_object_weight_power > 0:
                scale_tensor = object_scale[i].clamp_min(1.0)
                small_weight = (float(small_object_ref_scale) / scale_tensor).clamp(
                    min=1.0,
                    max=float(small_object_max_weight),
                )
                obj_weight = obj_weight * small_weight.pow(float(small_object_weight_power))
            reg_weight[b, 0, cy, cx] = obj_weight
    return {"heatmap": heatmap, "reg": reg, "box": box, "reg_mask": reg_mask, "reg_weight": reg_weight}


def _decode_positive_boxes(reg_pred: torch.Tensor, mask: torch.Tensor, stride: int) -> torch.Tensor:
    pos = mask[:, 0].bool()
    if not bool(pos.any()):
        return reg_pred.new_zeros((0, 8))
    byx = pos.nonzero(as_tuple=False)
    b = byx[:, 0]
    y = byx[:, 1].to(dtype=reg_pred.dtype)
    x = byx[:, 2].to(dtype=reg_pred.dtype)
    r = reg_pred.permute(0, 2, 3, 1)[b, byx[:, 1], byx[:, 2]]
    cx = (x + r[:, 0]) * stride
    cy = (y + r[:, 1]) * stride
    long_side = torch.exp(r[:, 2]).clamp(max=1e4) * stride
    short_side = torch.exp(r[:, 3]).clamp(max=1e4) * stride
    theta = torch.atan2(r[:, 4], r[:, 5])
    boxes5 = torch.stack([cx, cy, long_side, short_side, theta], dim=1)
    return obb5_to_poly(boxes5)


def _decode_positive_xywhr(reg_map: torch.Tensor, mask: torch.Tensor, stride: int) -> torch.Tensor:
    pos = mask[:, 0].bool()
    if not bool(pos.any()):
        return reg_map.new_zeros((0, 5))
    byx = pos.nonzero(as_tuple=False)
    b = byx[:, 0]
    y = byx[:, 1].to(dtype=reg_map.dtype)
    x = byx[:, 2].to(dtype=reg_map.dtype)
    r = reg_map.permute(0, 2, 3, 1)[b, byx[:, 1], byx[:, 2]]
    cx = (x + r[:, 0]) * stride
    cy = (y + r[:, 1]) * stride
    long_side = torch.exp(r[:, 2]).clamp(max=1e4) * stride
    short_side = torch.exp(r[:, 3]).clamp(max=1e4) * stride
    theta = torch.atan2(r[:, 4], r[:, 5])
    return torch.stack([cx, cy, long_side, short_side, theta], dim=1)


def _xywhr_to_xy_sigma(xywhr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = xywhr.shape
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-4, max=1e5).reshape(-1, 2)
    angle = xywhr[..., 4].reshape(-1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rot = torch.stack((cos_a, -sin_a, sin_a, cos_a), dim=-1).reshape(-1, 2, 2)
    scale = 0.5 * torch.diag_embed(wh)
    sigma = rot.bmm(scale.square()).bmm(rot.transpose(1, 2)).reshape(shape[:-1] + (2, 2))
    return xy, sigma


def _gwd_loss_from_xywhr(pred_xywhr: torch.Tensor, target_xywhr: torch.Tensor) -> torch.Tensor:
    if pred_xywhr.numel() == 0:
        return pred_xywhr.new_zeros((0,))
    with torch.autocast(device_type=pred_xywhr.device.type, enabled=False):
        pred_xywhr = pred_xywhr.float()
        target_xywhr = target_xywhr.float()
        xy_p, sigma_p = _xywhr_to_xy_sigma(pred_xywhr)
        xy_t, sigma_t = _xywhr_to_xy_sigma(target_xywhr)
        xy_distance = (xy_p - xy_t).square().sum(dim=-1)
        whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        whr_distance = whr_distance + sigma_t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        trace_term = sigma_p.bmm(sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        det_sqrt = (sigma_p.det() * sigma_t.det()).clamp_min(0.0).sqrt()
        whr_distance = whr_distance - 2.0 * (trace_term + 2.0 * det_sqrt).clamp_min(0.0).sqrt()
        distance = (xy_distance + whr_distance.clamp_min(0.0)).clamp_min(0.0).sqrt()
        scale = 2.0 * det_sqrt.sqrt().sqrt().clamp_min(1e-4)
        distance = distance / scale
        return 1.0 - 1.0 / (1.0 + torch.log1p(distance))


class TinyBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
        )
        self.out_channels = width * 2
        self.stride = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, 3, groups=channels),
            ConvBNAct(channels, hidden, 1),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class LiteFPNBackbone(nn.Module):
    """Lightweight multi-scale backbone with stride-4 fused features.

    The design keeps the detector fast while giving the head access to
    stride-2 local detail, stride-4 dense features, and stride-8 context.
    """

    def __init__(self, in_channels: int = 3, width: int = 64) -> None:
        super().__init__()
        c2 = width
        c3 = width * 2
        c4 = width * 4
        out = width * 2
        self.stem = ConvBNAct(in_channels, c2, 3, stride=2)
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(c2),
            DepthwiseSeparableBlock(c2),
        )
        self.down3 = ConvBNAct(c2, c3, 3, stride=2)
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(c3),
            DepthwiseSeparableBlock(c3),
        )
        self.down4 = ConvBNAct(c3, c4, 3, stride=2)
        self.stage4 = nn.Sequential(
            DepthwiseSeparableBlock(c4),
            DepthwiseSeparableBlock(c4),
        )
        self.lat2 = ConvBNAct(c2, out, 1)
        self.lat3 = ConvBNAct(c3, out, 1)
        self.lat4 = ConvBNAct(c4, out, 1)
        self.smooth = nn.Sequential(
            DepthwiseSeparableBlock(out),
            DepthwiseSeparableBlock(out),
        )
        self.out_channels = out
        self.stride = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p2 = self.stage2(self.stem(x))
        p3 = self.stage3(self.down3(p2))
        p4 = self.stage4(self.down4(p3))
        p2_down = F.max_pool2d(self.lat2(p2), kernel_size=2, stride=2)
        p3_lat = self.lat3(p3)
        p4_up = F.interpolate(self.lat4(p4), size=p3_lat.shape[-2:], mode="nearest")
        return self.smooth(p2_down + p3_lat + p4_up)


class DINOv3ConvNeXtTinyFPNBackbone(nn.Module):
    """DINOv3 ConvNeXt-Tiny backbone with a trainable stride-4 FPN adapter."""

    def __init__(
        self,
        *,
        out_channels: int = 192,
        pretrained: bool = True,
        freeze: bool = True,
        trainable_stages: int = 0,
        trainable_blocks: int = 0,
        model_id: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.freeze = bool(freeze)
        self.trainable_stages = max(0, min(4, int(trainable_stages)))
        self.trainable_blocks = max(0, int(trainable_blocks))
        self.trainable_block_prefixes: tuple[str, ...] = ()
        self.out_channels = int(out_channels)
        self.stride = 4
        out_features = ["stage1", "stage2", "stage3", "stage4"]
        if pretrained:
            try:
                from transformers import AutoBackbone
            except ImportError as exc:
                raise ImportError(
                    "DINOv3 backbone requires transformers. Install it with "
                    "`python -m pip install transformers huggingface_hub safetensors`."
                ) from exc
            try:
                offline = os.environ.get("HF_HUB_OFFLINE", "") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"
                if offline:
                    from huggingface_hub import snapshot_download
                    from transformers.models.dinov3_convnext.modeling_dinov3_convnext import (
                        DINOv3ConvNextBackbone,
                    )

                    pretrained_path = snapshot_download(model_id, local_files_only=True)
                    self.backbone = DINOv3ConvNextBackbone.from_pretrained(
                        pretrained_path,
                        out_features=out_features,
                        local_files_only=True,
                    )
                else:
                    self.backbone = AutoBackbone.from_pretrained(
                        model_id,
                        out_features=out_features,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load gated DINOv3 checkpoint {model_id!r}. "
                    "Set HF_TOKEN after accepting the model's upstream license."
                ) from exc
        else:
            try:
                from transformers.models.dinov3_convnext.configuration_dinov3_convnext import (
                    DINOv3ConvNextConfig,
                )
                from transformers.models.dinov3_convnext.modeling_dinov3_convnext import (
                    DINOv3ConvNextBackbone,
                )
            except ImportError as exc:
                raise ImportError("Installed transformers version does not include DINOv3 ConvNeXt.") from exc
            config = DINOv3ConvNextConfig(out_features=out_features)
            self.backbone = DINOv3ConvNextBackbone(config)

        channels = list(getattr(self.backbone, "channels", [96, 192, 384, 768]))
        if len(channels) != 4:
            raise ValueError(f"Expected four DINOv3 feature channel sizes, got {channels}")
        self.lateral = nn.ModuleList([ConvBNAct(ch, self.out_channels, 1) for ch in channels])
        self.smooth = nn.Sequential(
            DepthwiseSeparableBlock(self.out_channels),
            DepthwiseSeparableBlock(self.out_channels),
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.trainable_block_prefixes = self._select_trainable_block_prefixes(self.trainable_blocks)
        self._set_backbone_trainability()

    def _select_trainable_block_prefixes(self, count: int) -> tuple[str, ...]:
        if count <= 0:
            return ()
        block_ids: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        for name, _ in self.backbone.named_parameters():
            parts = name.split(".")
            if len(parts) < 5 or parts[0] != "model" or parts[1] != "stages" or parts[3] != "layers":
                continue
            try:
                stage_idx = int(parts[2])
                block_idx = int(parts[4])
            except ValueError:
                continue
            prefix = ".".join(parts[:5]) + "."
            if prefix in seen:
                continue
            block_ids.append((stage_idx, block_idx, prefix))
            seen.add(prefix)
        block_ids.sort(key=lambda item: (item[0], item[1]))
        selected = block_ids[-count:] if count < len(block_ids) else block_ids
        return tuple(prefix for _, _, prefix in selected)

    def _param_is_trainable(self, param_name: str) -> bool:
        if not self.freeze:
            return True
        if self.trainable_block_prefixes:
            return any(param_name.startswith(prefix) for prefix in self.trainable_block_prefixes)
        if self.trainable_stages <= 0:
            return False
        first_trainable = 4 - self.trainable_stages
        for stage_idx in range(first_trainable, 4):
            if param_name.startswith(f"model.stages.{stage_idx}."):
                return True
        return False

    def _set_backbone_trainability(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad_(self._param_is_trainable(name))
        self.fully_frozen_backbone = not any(param.requires_grad for param in self.backbone.parameters())
        if self.fully_frozen_backbone:
            self.backbone.eval()
        elif self.freeze:
            self._eval_frozen_modules()

    def _eval_frozen_modules(self) -> None:
        for module in self.backbone.modules():
            params = list(module.parameters(recurse=True))
            if params and not any(param.requires_grad for param in params):
                module.eval()

    def train(self, mode: bool = True) -> "DINOv3ConvNeXtTinyFPNBackbone":
        super().train(mode)
        if self.fully_frozen_backbone:
            self.backbone.eval()
        elif self.freeze:
            self._eval_frozen_modules()
        return self

    def _backbone_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.pixel_mean.to(dtype=x.dtype)) / self.pixel_std.to(dtype=x.dtype)
        if self.fully_frozen_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=x)
        else:
            outputs = self.backbone(pixel_values=x)
        features = list(outputs.feature_maps)
        if len(features) != 4:
            raise RuntimeError(f"DINOv3 backbone returned {len(features)} features, expected 4")
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._backbone_feature_maps(x)
        fused = self.lateral[0](features[0])
        target_size = fused.shape[-2:]
        for idx, feat in enumerate(features[1:], start=1):
            fused = fused + F.interpolate(
                self.lateral[idx](feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.smooth(fused)


class DINOv3ConvNeXtTinyPyramidBackbone(DINOv3ConvNeXtTinyFPNBackbone):
    """DINOv3 ConvNeXt-Tiny FPN that keeps P2/P3/P4/P5 features.

    The previous detector fused every scale back to a single stride-4 map. That
    is fast, but dense remote-sensing patches suffer from heatmap collisions and
    large/small object scale conflict. This backbone exposes the pyramid so the
    detector can assign each OBB to a scale-appropriate level.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # The parent class owns a single-scale smoothing tower. The pyramid
        # detector uses per-level smoothing instead, so keep the inherited tower
        # out of DDP's trainable parameter set.
        self.smooth = nn.Identity()
        self.strides = [4, 8, 16, 32]
        self.stride = self.strides[0]
        self.pyramid_smooth = nn.ModuleList(
            [
                nn.Sequential(DepthwiseSeparableBlock(self.out_channels), DepthwiseSeparableBlock(self.out_channels))
                for _ in self.strides
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self._backbone_feature_maps(x)
        laterals = [lat(feat) for lat, feat in zip(self.lateral, features)]
        for idx in range(len(laterals) - 2, -1, -1):
            laterals[idx] = laterals[idx] + F.interpolate(
                laterals[idx + 1],
                size=laterals[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return [smooth(feat) for smooth, feat in zip(self.pyramid_smooth, laterals)]


class OBBDetectorHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 60,
        hidden: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(max(1, int(num_layers))):
            layers.extend(
                [
                    nn.Conv2d(current, hidden, 3, padding=1),
                    nn.GroupNorm(16, hidden),
                    nn.SiLU(inplace=True),
                ]
            )
            current = hidden
        self.shared = nn.Sequential(*layers)
        self.cls = nn.Conv2d(hidden, num_classes, 1)
        self.reg = nn.Conv2d(hidden, 6, 1)
        nn.init.constant_(self.cls.bias, -4.6)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.shared(feat)
        return {"heatmap_logits": self.cls(x), "reg": self.reg(x)}


class ContextOBBHead(nn.Module):
    """Global OBB proposals for large relation-context objects.

    Center-point heatmaps struggle with stuff-like remote-sensing entities such
    as road, harbor, water, and block. This tiny head predicts a small fixed
    number of OBB queries for each configured context class from pooled image
    features.
    """

    def __init__(
        self,
        in_channels: int,
        num_context_classes: int,
        *,
        num_queries_per_class: int = 1,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.num_context_classes = num_context_classes
        self.num_queries_per_class = max(1, int(num_queries_per_class))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
        )
        self.cls = nn.Linear(hidden, num_context_classes * self.num_queries_per_class)
        self.reg = nn.Linear(hidden, num_context_classes * self.num_queries_per_class * 6)
        nn.init.constant_(self.cls.bias, -2.0)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pool(feat).flatten(1)
        x = self.mlp(pooled)
        logits = self.cls(x).reshape(feat.shape[0], self.num_context_classes, self.num_queries_per_class)
        reg = self.reg(x).reshape(feat.shape[0], self.num_context_classes, self.num_queries_per_class, 6)
        return {"context_logits": logits, "context_reg": reg}


@dataclass
class DecodedDetections:
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


class TinyOBBDetector(nn.Module):
    """Anchor-free oriented object detector."""

    def __init__(
        self,
        num_classes: int = 60,
        width: int = 64,
        box_loss_weight: float = 0.0,
        gwd_loss_weight: float = 0.0,
        backbone: str = "tiny",
        context_class_ids: Sequence[int] | None = None,
        context_loss_weight: float = 1.0,
        context_queries_per_class: int = 1,
        context_target_mode: str = "largest",
        dinov3_pretrained: bool = True,
        freeze_backbone: bool = True,
        dinov3_trainable_stages: int = 0,
        dinov3_trainable_blocks: int = 0,
        head_hidden: int = 128,
        head_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_loss_weight = float(box_loss_weight)
        self.gwd_loss_weight = float(gwd_loss_weight)
        self.backbone_name = backbone
        self.context_class_ids = [int(class_id) for class_id in (context_class_ids or [])]
        self.context_loss_weight = float(context_loss_weight)
        self.context_queries_per_class = max(1, int(context_queries_per_class))
        if context_target_mode not in {"largest", "relation"}:
            raise ValueError(f"Unsupported context target mode: {context_target_mode}")
        self.context_target_mode = context_target_mode
        if backbone == "tiny":
            self.backbone = TinyBackbone(width=width)
        elif backbone == "lite_fpn":
            self.backbone = LiteFPNBackbone(width=width)
        elif backbone == "dinov3_convnext_tiny":
            self.backbone = DINOv3ConvNeXtTinyFPNBackbone(
                out_channels=width,
                pretrained=dinov3_pretrained,
                freeze=freeze_backbone,
                trainable_stages=dinov3_trainable_stages,
                trainable_blocks=dinov3_trainable_blocks,
            )
        else:
            raise ValueError(f"Unsupported detector backbone: {backbone}")
        self.head = OBBDetectorHead(
            self.backbone.out_channels,
            num_classes=num_classes,
            hidden=head_hidden,
            num_layers=head_layers,
        )
        self.context_head = (
            ContextOBBHead(
                self.backbone.out_channels,
                len(self.context_class_ids),
                num_queries_per_class=self.context_queries_per_class,
                hidden=head_hidden,
            )
            if self.context_class_ids
            else None
        )
        self.stride = self.backbone.stride

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(images)
        outputs = self.head(feat)
        outputs["features"] = feat
        if self.context_head is not None:
            outputs.update(self.context_head(feat))
        return outputs

    def _context_targets(
        self,
        targets: Sequence[SceneGraphTarget],
        *,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        batch = len(targets)
        num_context = len(self.context_class_ids)
        num_queries = self.context_queries_per_class
        presence = torch.zeros((batch, num_context, num_queries), device=device)
        reg = torch.zeros((batch, num_context, num_queries, 6), device=device)
        for b, target in enumerate(targets):
            boxes = target.boxes.to(device)
            labels = target.labels.to(device)
            if len(labels) == 0:
                continue
            params = obb_parameters(boxes)
            areas = params["long"] * params["short"]
            relation_counts = torch.zeros(len(labels), device=device)
            if self.context_target_mode == "relation" and target.relations.numel() > 0:
                relations = target.relations.to(device=device).long()
                valid = (relations[:, 0] >= 0) & (relations[:, 0] < len(labels))
                valid = valid & (relations[:, 1] >= 0) & (relations[:, 1] < len(labels))
                relations = relations[valid]
                if len(relations) > 0:
                    ones = torch.ones(len(relations), device=device)
                    relation_counts.scatter_add_(0, relations[:, 0], ones)
                    relation_counts.scatter_add_(0, relations[:, 1], ones)
            image_w = max(float(target.width), 1.0)
            image_h = max(float(target.height), 1.0)
            max_dim = max(image_w, image_h)
            for j, class_id in enumerate(self.context_class_ids):
                idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
                if len(idx) == 0:
                    continue
                if self.context_target_mode == "relation":
                    scale = areas[idx].max().clamp_min(1.0) + 1.0
                    target_score = relation_counts[idx] * scale + areas[idx] / scale
                    order = torch.argsort(target_score, descending=True)
                else:
                    order = torch.argsort(areas[idx], descending=True)
                selected = idx[order[:num_queries]]
                for query_idx, best in enumerate(selected.tolist()):
                    best_idx = int(best)
                    presence[b, j, query_idx] = 1.0
                    center = params["center"][best_idx]
                    long_side = params["long"][best_idx].clamp_min(1e-4)
                    short_side = params["short"][best_idx].clamp_min(1e-4)
                    theta = params["theta"][best_idx]
                    reg[b, j, query_idx] = torch.stack(
                        [
                            (center[0] / image_w).clamp(0.0, 1.0),
                            (center[1] / image_h).clamp(0.0, 1.0),
                            torch.log(long_side / max_dim),
                            torch.log(short_side / max_dim),
                            torch.sin(theta),
                            torch.cos(theta),
                        ]
                    )
        return {"presence": presence, "reg": reg}

    def _context_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[SceneGraphTarget],
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["context_logits"].float()
        raw_reg = outputs["context_reg"].float()
        tgt = self._context_targets(targets, device=logits.device)
        presence = tgt["presence"]
        pred_reg = torch.stack(
            [
                torch.sigmoid(raw_reg[..., 0]),
                torch.sigmoid(raw_reg[..., 1]),
                raw_reg[..., 2],
                raw_reg[..., 3],
                raw_reg[..., 4],
                raw_reg[..., 5],
            ],
            dim=-1,
        )
        cls_loss = F.binary_cross_entropy_with_logits(logits, presence)
        reg_loss = (
            F.smooth_l1_loss(pred_reg, tgt["reg"], reduction="none") * presence[..., None]
        ).sum() / presence.sum().clamp_min(1.0)
        return {
            "loss_context": cls_loss + reg_loss,
            "loss_context_cls": cls_loss,
            "loss_context_reg": reg_loss,
        }

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[SceneGraphTarget],
        *,
        class_weights: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["heatmap_logits"].float()
        reg_pred = outputs["reg"].float()
        tgt = build_detector_targets(
            targets,
            feat_h=logits.shape[-2],
            feat_w=logits.shape[-1],
            stride=self.stride,
            num_classes=self.num_classes,
            device=logits.device,
            class_weights=class_weights,
        )
        heatmap_loss = _centernet_focal_loss(logits, tgt["heatmap"], class_weights=class_weights)
        mask = tgt["reg_mask"]
        reg_weight = tgt["reg_weight"]
        reg_loss = (
            F.smooth_l1_loss(reg_pred, tgt["reg"], reduction="none") * mask * reg_weight
        ).sum() / reg_weight.sum().clamp_min(1.0)
        pred_boxes = _decode_positive_boxes(reg_pred, mask, self.stride)
        if pred_boxes.numel() == 0:
            box_loss = reg_pred.sum() * 0
            gwd_loss = reg_pred.sum() * 0
        else:
            pos = mask[:, 0].bool()
            byx = pos.nonzero(as_tuple=False)
            target_boxes = tgt["box"].permute(0, 2, 3, 1)[byx[:, 0], byx[:, 1], byx[:, 2]]
            box_loss = F.smooth_l1_loss(pred_boxes / self.stride, target_boxes / self.stride, reduction="mean")
            pred_xywhr = _decode_positive_xywhr(reg_pred, mask, self.stride)
            target_xywhr = _decode_positive_xywhr(tgt["reg"], mask, self.stride)
            gwd_loss = _gwd_loss_from_xywhr(pred_xywhr, target_xywhr).mean()
        losses = {
            "loss": heatmap_loss + reg_loss + self.box_loss_weight * box_loss + self.gwd_loss_weight * gwd_loss,
            "loss_heatmap": heatmap_loss,
            "loss_reg": reg_loss,
            "loss_box": box_loss,
            "loss_gwd": gwd_loss,
            "num_pos": mask.sum().detach(),
        }
        if self.context_head is not None:
            context_losses = self._context_loss(outputs, targets)
            losses.update(context_losses)
            losses["loss"] = losses["loss"] + self.context_loss_weight * context_losses["loss_context"]
        return losses

    @torch.no_grad()
    def decode(
        self,
        outputs: Dict[str, torch.Tensor],
        *,
        topk: int = 300,
        image_sizes: Sequence[tuple[int, int]] | None = None,
        peak_nms: bool = True,
        multilabel: bool = True,
        labels_per_cell: int | None = None,
        context_score_thr: float = 0.05,
    ) -> List[DecodedDetections]:
        heat = torch.sigmoid(outputs["heatmap_logits"])
        if peak_nms:
            heat = _peak_nms(heat)
        reg = outputs["reg"]
        batch, num_classes, h, w = heat.shape
        per_cell_k = None
        if multilabel:
            if labels_per_cell is not None and labels_per_cell < num_classes:
                per_cell_k = max(1, labels_per_cell)
                cell_scores, cell_labels = torch.topk(heat, k=per_cell_k, dim=1)
                flat = cell_scores.permute(0, 2, 3, 1).reshape(batch, -1)
                label_flat = cell_labels.permute(0, 2, 3, 1).reshape(batch, -1)
            else:
                flat = heat.reshape(batch, -1)
                label_flat = None
        else:
            cell_scores, cell_labels = heat.max(dim=1)
            flat = cell_scores.reshape(batch, -1)
            label_flat = cell_labels.reshape(batch, -1)
        k = min(topk, flat.shape[1])
        scores, indices = torch.topk(flat, k=k, dim=1)
        decoded: List[DecodedDetections] = []
        for b in range(batch):
            idx = indices[b]
            if multilabel:
                if label_flat is None:
                    labels = (idx // (h * w)).long() + 1
                    rem = idx % (h * w)
                else:
                    labels = label_flat[b, idx].long() + 1
                    rem = idx // int(per_cell_k)
            else:
                labels = label_flat[b, idx].long() + 1
                rem = idx
            ys = (rem // w).float()
            xs = (rem % w).float()
            reg_b = reg[b].permute(1, 2, 0).reshape(-1, 6)
            r = reg_b[rem]
            cx = (xs + r[:, 0]) * self.stride
            cy = (ys + r[:, 1]) * self.stride
            long_side = torch.exp(r[:, 2]).clamp(max=1e4) * self.stride
            short_side = torch.exp(r[:, 3]).clamp(max=1e4) * self.stride
            theta = torch.atan2(r[:, 4], r[:, 5])
            boxes5 = torch.stack([cx, cy, long_side, short_side, theta], dim=1)
            boxes8 = obb5_to_poly(boxes5)
            if image_sizes is not None:
                img_w, img_h = image_sizes[b]
                keep = (cx >= 0) & (cx < img_w) & (cy >= 0) & (cy < img_h)
                boxes8 = boxes8[keep]
                labels = labels[keep]
                scores_b = scores[b][keep]
            else:
                scores_b = scores[b]
            if self.context_class_ids and "context_logits" in outputs and image_sizes is not None:
                context_scores = torch.sigmoid(outputs["context_logits"][b])
                context_reg = outputs["context_reg"][b]
                keep_context = context_scores >= context_score_thr
                if bool(keep_context.any()):
                    class_ids = torch.tensor(self.context_class_ids, dtype=torch.long, device=heat.device)
                    class_ids = class_ids[:, None].expand(-1, self.context_queries_per_class)
                    img_w, img_h = image_sizes[b]
                    max_dim = float(max(img_w, img_h))
                    raw = context_reg[keep_context]
                    ctx_cx = torch.sigmoid(raw[:, 0]) * float(img_w)
                    ctx_cy = torch.sigmoid(raw[:, 1]) * float(img_h)
                    ctx_long = torch.exp(raw[:, 2]).clamp(max=2.0) * max_dim
                    ctx_short = torch.exp(raw[:, 3]).clamp(max=2.0) * max_dim
                    ctx_theta = torch.atan2(raw[:, 4], raw[:, 5])
                    ctx_boxes = obb5_to_poly(torch.stack([ctx_cx, ctx_cy, ctx_long, ctx_short, ctx_theta], dim=1))
                    boxes8 = torch.cat([boxes8, ctx_boxes], dim=0)
                    labels = torch.cat([labels, class_ids[keep_context]], dim=0)
                    scores_b = torch.cat([scores_b, context_scores[keep_context]], dim=0)
                    if len(scores_b) > topk:
                        order = torch.argsort(scores_b, descending=True)[:topk]
                        boxes8 = boxes8[order]
                        labels = labels[order]
                        scores_b = scores_b[order]
            decoded.append(DecodedDetections(boxes=boxes8, labels=labels, scores=scores_b))
        return decoded


class MultiScaleOBBDetector(TinyOBBDetector):
    """Multi-scale OBB detector with scale-assigned CenterNet heads.

    It keeps the public detector API used by ORSGGDirectedNet: forward returns a
    high-resolution feature map under ``features``, ``loss`` trains the detector,
    and ``decode`` returns scored OBB detections. Internally, detection heads are
    attached to P2/P3/P4/P5 to reduce dense-scene heatmap conflicts.
    """

    def __init__(
        self,
        num_classes: int = 60,
        width: int = 192,
        box_loss_weight: float = 0.0,
        gwd_loss_weight: float = 0.0,
        backbone: str = "dinov3_convnext_tiny_pyramid",
        context_class_ids: Sequence[int] | None = None,
        context_loss_weight: float = 1.0,
        context_queries_per_class: int = 1,
        context_target_mode: str = "largest",
        dinov3_pretrained: bool = True,
        freeze_backbone: bool = True,
        dinov3_trainable_stages: int = 0,
        dinov3_trainable_blocks: int = 0,
        head_hidden: int = 128,
        head_layers: int = 2,
        relation_object_weight_power: float = 0.25,
        small_object_weight_power: float = 0.5,
    ) -> None:
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.box_loss_weight = float(box_loss_weight)
        self.gwd_loss_weight = float(gwd_loss_weight)
        self.backbone_name = backbone
        self.context_class_ids = [int(class_id) for class_id in (context_class_ids or [])]
        self.context_loss_weight = float(context_loss_weight)
        self.context_queries_per_class = max(1, int(context_queries_per_class))
        if context_target_mode not in {"largest", "relation"}:
            raise ValueError(f"Unsupported context target mode: {context_target_mode}")
        self.context_target_mode = context_target_mode
        if backbone != "dinov3_convnext_tiny_pyramid":
            raise ValueError(f"MultiScaleOBBDetector only supports dinov3_convnext_tiny_pyramid, got {backbone}")
        self.backbone = DINOv3ConvNeXtTinyPyramidBackbone(
            out_channels=width,
            pretrained=dinov3_pretrained,
            freeze=freeze_backbone,
            trainable_stages=dinov3_trainable_stages,
            trainable_blocks=dinov3_trainable_blocks,
        )
        self.strides = list(self.backbone.strides)
        self.scale_ranges = [(0.0, 48.0), (48.0, 128.0), (128.0, 256.0), (256.0, float("inf"))]
        self.stride = self.strides[0]
        self.out_channels = self.backbone.out_channels
        self.heads = nn.ModuleList(
            [
                OBBDetectorHead(
                    self.backbone.out_channels,
                    num_classes=num_classes,
                    hidden=head_hidden,
                    num_layers=head_layers,
                )
                for _ in self.strides
            ]
        )
        self.context_head = (
            ContextOBBHead(
                self.backbone.out_channels,
                len(self.context_class_ids),
                num_queries_per_class=self.context_queries_per_class,
                hidden=head_hidden,
            )
            if self.context_class_ids
            else None
        )
        self.relation_object_weight_power = float(relation_object_weight_power)
        self.small_object_weight_power = float(small_object_weight_power)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        features = self.backbone(images)
        level_outputs = [head(feat) for head, feat in zip(self.heads, features)]
        outputs: Dict[str, torch.Tensor | list[torch.Tensor]] = {
            "pyramid_features": features,
            "features": features[0],
            "feature_strides": self.strides,
            "heatmap_logits": [item["heatmap_logits"] for item in level_outputs],
            "reg": [item["reg"] for item in level_outputs],
        }
        if self.context_head is not None:
            outputs.update(self.context_head(features[-1]))
        return outputs

    def loss(
        self,
        outputs: Dict[str, torch.Tensor | list[torch.Tensor]],
        targets: Sequence[SceneGraphTarget],
        *,
        class_weights: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        logits_levels = outputs["heatmap_logits"]
        reg_levels = outputs["reg"]
        if not isinstance(logits_levels, list) or not isinstance(reg_levels, list):
            raise TypeError("MultiScaleOBBDetector expected list heatmap/reg outputs")
        heatmap_losses = []
        reg_losses = []
        box_losses = []
        gwd_losses = []
        num_pos_total = logits_levels[0].new_tensor(0.0)
        for level_idx, (logits, reg_pred, stride, scale_range) in enumerate(
            zip(logits_levels, reg_levels, self.strides, self.scale_ranges)
        ):
            logits = logits.float()
            reg_pred = reg_pred.float()
            tgt = build_detector_targets(
                targets,
                feat_h=logits.shape[-2],
                feat_w=logits.shape[-1],
                stride=int(stride),
                num_classes=self.num_classes,
                device=logits.device,
                class_weights=class_weights,
                min_object_scale=scale_range[0],
                max_object_scale=scale_range[1],
                relation_object_weight_power=self.relation_object_weight_power,
                small_object_weight_power=self.small_object_weight_power if level_idx == 0 else 0.0,
                small_object_ref_scale=48.0,
                small_object_max_weight=3.0,
            )
            heatmap_losses.append(_centernet_focal_loss(logits, tgt["heatmap"], class_weights=class_weights))
            mask = tgt["reg_mask"]
            reg_weight = tgt["reg_weight"]
            reg_losses.append(
                (F.smooth_l1_loss(reg_pred, tgt["reg"], reduction="none") * mask * reg_weight).sum()
                / reg_weight.sum().clamp_min(1.0)
            )
            pred_boxes = _decode_positive_boxes(reg_pred, mask, int(stride))
            if pred_boxes.numel() == 0:
                box_losses.append(reg_pred.sum() * 0)
                gwd_losses.append(reg_pred.sum() * 0)
            else:
                pos = mask[:, 0].bool()
                byx = pos.nonzero(as_tuple=False)
                target_boxes = tgt["box"].permute(0, 2, 3, 1)[byx[:, 0], byx[:, 1], byx[:, 2]]
                box_losses.append(F.smooth_l1_loss(pred_boxes / float(stride), target_boxes / float(stride), reduction="mean"))
                pred_xywhr = _decode_positive_xywhr(reg_pred, mask, int(stride))
                target_xywhr = _decode_positive_xywhr(tgt["reg"], mask, int(stride))
                gwd_losses.append(_gwd_loss_from_xywhr(pred_xywhr, target_xywhr).mean())
            num_pos_total = num_pos_total + mask.sum().detach()
        heatmap_loss = torch.stack(heatmap_losses).mean()
        reg_loss = torch.stack(reg_losses).mean()
        box_loss = torch.stack(box_losses).mean()
        gwd_loss = torch.stack(gwd_losses).mean()
        losses = {
            "loss": heatmap_loss + reg_loss + self.box_loss_weight * box_loss + self.gwd_loss_weight * gwd_loss,
            "loss_heatmap": heatmap_loss,
            "loss_reg": reg_loss,
            "loss_box": box_loss,
            "loss_gwd": gwd_loss,
            "num_pos": num_pos_total,
        }
        if self.context_head is not None:
            context_losses = self._context_loss(outputs, targets)  # type: ignore[arg-type]
            losses.update(context_losses)
            losses["loss"] = losses["loss"] + self.context_loss_weight * context_losses["loss_context"]
        return losses

    @torch.no_grad()
    def decode(
        self,
        outputs: Dict[str, torch.Tensor | list[torch.Tensor]],
        *,
        topk: int = 300,
        image_sizes: Sequence[tuple[int, int]] | None = None,
        peak_nms: bool = True,
        multilabel: bool = True,
        labels_per_cell: int | None = None,
        context_score_thr: float = 0.05,
    ) -> List[DecodedDetections]:
        logits_levels = outputs["heatmap_logits"]
        reg_levels = outputs["reg"]
        if not isinstance(logits_levels, list) or not isinstance(reg_levels, list):
            raise TypeError("MultiScaleOBBDetector expected list heatmap/reg outputs")
        batch = logits_levels[0].shape[0]
        per_image_boxes: list[list[torch.Tensor]] = [[] for _ in range(batch)]
        per_image_labels: list[list[torch.Tensor]] = [[] for _ in range(batch)]
        per_image_scores: list[list[torch.Tensor]] = [[] for _ in range(batch)]
        per_level_topk = max(topk, topk // max(len(self.strides), 1))
        for logits, reg, stride in zip(logits_levels, reg_levels, self.strides):
            heat = torch.sigmoid(logits)
            if peak_nms:
                heat = _peak_nms(heat)
            batch_l, num_classes, h, w = heat.shape
            per_cell_k = None
            if multilabel:
                if labels_per_cell is not None and labels_per_cell < num_classes:
                    per_cell_k = max(1, labels_per_cell)
                    cell_scores, cell_labels = torch.topk(heat, k=per_cell_k, dim=1)
                    flat = cell_scores.permute(0, 2, 3, 1).reshape(batch_l, -1)
                    label_flat = cell_labels.permute(0, 2, 3, 1).reshape(batch_l, -1)
                else:
                    flat = heat.reshape(batch_l, -1)
                    label_flat = None
            else:
                cell_scores, cell_labels = heat.max(dim=1)
                flat = cell_scores.reshape(batch_l, -1)
                label_flat = cell_labels.reshape(batch_l, -1)
            k = min(per_level_topk, flat.shape[1])
            scores, indices = torch.topk(flat, k=k, dim=1)
            for b in range(batch_l):
                idx = indices[b]
                if multilabel:
                    if label_flat is None:
                        labels = (idx // (h * w)).long() + 1
                        rem = idx % (h * w)
                    else:
                        labels = label_flat[b, idx].long() + 1
                        rem = idx // int(per_cell_k)
                else:
                    labels = label_flat[b, idx].long() + 1
                    rem = idx
                ys = (rem // w).float()
                xs = (rem % w).float()
                reg_b = reg[b].permute(1, 2, 0).reshape(-1, 6)
                r = reg_b[rem]
                cx = (xs + r[:, 0]) * int(stride)
                cy = (ys + r[:, 1]) * int(stride)
                long_side = torch.exp(r[:, 2]).clamp(max=1e4) * int(stride)
                short_side = torch.exp(r[:, 3]).clamp(max=1e4) * int(stride)
                theta = torch.atan2(r[:, 4], r[:, 5])
                boxes8 = obb5_to_poly(torch.stack([cx, cy, long_side, short_side, theta], dim=1))
                scores_b = scores[b]
                if image_sizes is not None:
                    img_w, img_h = image_sizes[b]
                    keep = (cx >= 0) & (cx < img_w) & (cy >= 0) & (cy < img_h)
                    boxes8 = boxes8[keep]
                    labels = labels[keep]
                    scores_b = scores_b[keep]
                per_image_boxes[b].append(boxes8)
                per_image_labels[b].append(labels)
                per_image_scores[b].append(scores_b)
        decoded: List[DecodedDetections] = []
        for b in range(batch):
            boxes8 = torch.cat(per_image_boxes[b], dim=0) if per_image_boxes[b] else logits_levels[0].new_zeros((0, 8))
            labels = torch.cat(per_image_labels[b], dim=0) if per_image_labels[b] else torch.zeros((0,), dtype=torch.long, device=logits_levels[0].device)
            scores_b = torch.cat(per_image_scores[b], dim=0) if per_image_scores[b] else logits_levels[0].new_zeros((0,))
            if self.context_class_ids and "context_logits" in outputs and image_sizes is not None:
                context_scores = torch.sigmoid(outputs["context_logits"][b])  # type: ignore[index]
                context_reg = outputs["context_reg"][b]  # type: ignore[index]
                keep_context = context_scores >= context_score_thr
                if bool(keep_context.any()):
                    class_ids = torch.tensor(self.context_class_ids, dtype=torch.long, device=boxes8.device)
                    class_ids = class_ids[:, None].expand(-1, self.context_queries_per_class)
                    img_w, img_h = image_sizes[b]
                    max_dim = float(max(img_w, img_h))
                    raw = context_reg[keep_context]
                    ctx_cx = torch.sigmoid(raw[:, 0]) * float(img_w)
                    ctx_cy = torch.sigmoid(raw[:, 1]) * float(img_h)
                    ctx_long = torch.exp(raw[:, 2]).clamp(max=2.0) * max_dim
                    ctx_short = torch.exp(raw[:, 3]).clamp(max=2.0) * max_dim
                    ctx_theta = torch.atan2(raw[:, 4], raw[:, 5])
                    ctx_boxes = obb5_to_poly(torch.stack([ctx_cx, ctx_cy, ctx_long, ctx_short, ctx_theta], dim=1))
                    boxes8 = torch.cat([boxes8, ctx_boxes], dim=0)
                    labels = torch.cat([labels, class_ids[keep_context]], dim=0)
                    scores_b = torch.cat([scores_b, context_scores[keep_context]], dim=0)
            if len(scores_b) > topk:
                order = torch.argsort(scores_b, descending=True)[:topk]
                boxes8 = boxes8[order]
                labels = labels[order]
                scores_b = scores_b[order]
            decoded.append(DecodedDetections(boxes=boxes8, labels=labels, scores=scores_b))
        return decoded
