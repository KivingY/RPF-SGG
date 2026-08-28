from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from orsgg.modeling.doge import obb_parameters
from orsgg.modeling.obb_detector import MultiScaleOBBDetector, TinyOBBDetector
from orsgg.ops import obb_iou
from orsgg.structures import SceneGraphPrediction, SceneGraphTarget


RELATION_GEOMETRY_MODES = (
    "semantic_only",
    "global_center",
    "global_obb",
    "subject_aligned_center",
    "subject_aligned_obb",
    "subject_aligned_obb_anisotropic",
)


def _sample_center_features(feature_map: torch.Tensor, boxes: torch.Tensor, stride: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return feature_map.new_zeros((0, feature_map.shape[0]))
    params = obb_parameters(boxes)
    centers = params["center"]
    _, feat_h, feat_w = feature_map.shape
    feat_x = (centers[:, 0] / float(stride)).clamp(0, max(feat_w - 1, 0))
    feat_y = (centers[:, 1] / float(stride)).clamp(0, max(feat_h - 1, 0))
    grid_x = feat_x / float(max(feat_w - 1, 1)) * 2.0 - 1.0 if feat_w > 1 else torch.zeros_like(feat_x)
    grid_y = feat_y / float(max(feat_h - 1, 1)) * 2.0 - 1.0 if feat_h > 1 else torch.zeros_like(feat_y)
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, len(boxes), 1, 2)
    sampled = F.grid_sample(feature_map[None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


def _sample_obb5_features(feature_map: torch.Tensor, boxes: torch.Tensor, stride: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return feature_map.new_zeros((0, feature_map.shape[0]))
    params = obb_parameters(boxes)
    centers = params["center"]
    theta = params["theta"]
    long_side = params["long"]
    short_side = params["short"]
    long_axis = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * (0.25 * long_side)[:, None]
    short_axis = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=1) * (0.25 * short_side)[:, None]
    sample_points = torch.stack(
        [
            centers,
            centers + long_axis,
            centers - long_axis,
            centers + short_axis,
            centers - short_axis,
        ],
        dim=1,
    )
    _, feat_h, feat_w = feature_map.shape
    feat_xy = sample_points / float(stride)
    feat_x = feat_xy[..., 0].clamp(0, max(feat_w - 1, 0))
    feat_y = feat_xy[..., 1].clamp(0, max(feat_h - 1, 0))
    if feat_w > 1:
        grid_x = feat_x / float(feat_w - 1) * 2.0 - 1.0
    else:
        grid_x = torch.zeros_like(feat_x)
    if feat_h > 1:
        grid_y = feat_y / float(feat_h - 1) * 2.0 - 1.0
    else:
        grid_y = torch.zeros_like(feat_y)
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, len(boxes), 5, 2)
    sampled = F.grid_sample(feature_map[None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0].permute(1, 2, 0).mean(dim=1).contiguous()


def _sample_object_features(
    feature_map: torch.Tensor,
    boxes: torch.Tensor,
    stride: int,
    pooling: str,
) -> torch.Tensor:
    if pooling == "center":
        return _sample_center_features(feature_map, boxes, stride)
    if pooling == "obb5":
        return _sample_obb5_features(feature_map, boxes, stride)
    raise ValueError(f"Unsupported relation feature pooling: {pooling}")


def _object_geometry_features(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 6))
    params = obb_parameters(boxes)
    img_w, img_h = image_size
    width = boxes.new_tensor(float(max(img_w, 1)))
    height = boxes.new_tensor(float(max(img_h, 1)))
    max_dim = torch.maximum(width, height)
    center = params["center"]
    long_side = params["long"].clamp_min(1e-4)
    short_side = params["short"].clamp_min(1e-4)
    theta = params["theta"]
    return torch.stack(
        [
            (center[:, 0] / width).clamp(0.0, 1.0),
            (center[:, 1] / height).clamp(0.0, 1.0),
            torch.log(long_side / max_dim),
            torch.log(short_side / max_dim),
            torch.sin(theta),
            torch.cos(theta),
        ],
        dim=1,
    )


def _pair_geometry_features(geom: torch.Tensor, subj_idx: torch.Tensor, obj_idx: torch.Tensor) -> torch.Tensor:
    subj = geom[subj_idx]
    obj = geom[obj_idx]
    delta = obj[:, :2] - subj[:, :2]
    size_delta = obj[:, 2:4] - subj[:, 2:4]
    sin_s, cos_s = subj[:, 4], subj[:, 5]
    sin_o, cos_o = obj[:, 4], obj[:, 5]
    sin_delta = sin_o * cos_s - cos_o * sin_s
    cos_delta = cos_o * cos_s + sin_o * sin_s
    return torch.cat(
        [
            delta,
            delta.abs(),
            size_delta,
            sin_delta[:, None],
            cos_delta[:, None],
        ],
        dim=1,
    )


def _address_geometry_dim(mode: str) -> int:
    if mode in {"semantic_only", "global_center"}:
        return 0
    if mode == "subject_aligned_center":
        return 2
    if mode in {"global_obb", "subject_aligned_obb", "subject_aligned_obb_anisotropic"}:
        return 6
    raise ValueError(f"Unsupported relation geometry mode: {mode}")


def _address_geometry_features(
    boxes: torch.Tensor,
    image_size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    """Return directed subject-object geometry for endpoint addressing.

    The output is indexed as ``[subject, object, feature]``. Subject-aligned
    modes rotate displacement by the negative subject angle and normalize it
    by the subject long/short axes. The doubled relative-angle representation
    respects the pi-periodicity of oriented boxes.
    """

    dim = _address_geometry_dim(mode)
    num_objects = len(boxes)
    if dim == 0:
        return boxes.new_zeros((num_objects, num_objects, 0))
    if num_objects == 0:
        return boxes.new_zeros((0, 0, dim))

    params = obb_parameters(boxes)
    centers = params["center"].float()
    long_side = params["long"].float().clamp_min(1.0e-4)
    short_side = params["short"].float().clamp_min(1.0e-4)
    theta = params["theta"].float()
    delta = centers[None, :, :] - centers[:, None, :]
    delta_x = delta[..., 0]
    delta_y = delta[..., 1]
    subject_theta = theta[:, None]
    object_theta = theta[None, :]
    relative_theta = object_theta - subject_theta
    log_long_ratio = torch.log(long_side[None, :] / long_side[:, None]).clamp(-5.0, 5.0)
    log_short_ratio = torch.log(short_side[None, :] / short_side[:, None]).clamp(-5.0, 5.0)
    angle_sin = torch.sin(2.0 * relative_theta)
    angle_cos = torch.cos(2.0 * relative_theta)

    if mode == "global_obb":
        image_width = float(max(int(image_size[0]), 1))
        image_height = float(max(int(image_size[1]), 1))
        global_x = (delta_x / image_width).clamp(-2.0, 2.0)
        global_y = (delta_y / image_height).clamp(-2.0, 2.0)
        return torch.stack(
            [global_x, global_y, log_long_ratio, log_short_ratio, angle_sin, angle_cos],
            dim=-1,
        ).to(dtype=boxes.dtype)

    cos_subject = torch.cos(subject_theta)
    sin_subject = torch.sin(subject_theta)
    local_x = ((cos_subject * delta_x + sin_subject * delta_y) / long_side[:, None]).clamp(-10.0, 10.0)
    local_y = ((-sin_subject * delta_x + cos_subject * delta_y) / short_side[:, None]).clamp(-10.0, 10.0)
    if mode == "subject_aligned_center":
        return torch.stack([local_x, local_y], dim=-1).to(dtype=boxes.dtype)
    return torch.stack(
        [local_x, local_y, log_long_ratio, log_short_ratio, angle_sin, angle_cos],
        dim=-1,
    ).to(dtype=boxes.dtype)


def _build_outgoing_slot_targets(
    num_objects: int,
    relations: torch.Tensor,
    *,
    num_predicates: int,
    num_slots: int,
    device: torch.device,
    relation_quality: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    pred_targets = torch.zeros((num_objects, num_slots), dtype=torch.long, device=device)
    bind_targets = torch.full((num_objects, num_slots), -1, dtype=torch.long, device=device)
    quality_targets = torch.zeros((num_objects, num_slots), dtype=torch.float32, device=device)
    if num_objects == 0 or relations.numel() == 0:
        return {"pred": pred_targets, "bind": bind_targets, "quality": quality_targets}
    rel = relations.to(device=device, dtype=torch.long)
    rel_quality = (
        torch.ones((len(rel),), dtype=torch.float32, device=device)
        if relation_quality is None or len(relation_quality) != len(rel)
        else relation_quality.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    )
    valid = (rel[:, 0] >= 0) & (rel[:, 0] < num_objects)
    valid = valid & (rel[:, 1] >= 0) & (rel[:, 1] < num_objects)
    valid = valid & (rel[:, 0] != rel[:, 1])
    valid = valid & (rel[:, 2] > 0) & (rel[:, 2] <= num_predicates)
    rel = rel[valid]
    rel_quality = rel_quality[valid]
    if len(rel) == 0:
        return {"pred": pred_targets, "bind": bind_targets, "quality": quality_targets}
    used = torch.zeros((num_objects,), dtype=torch.long, device=device)
    for rel_idx, (subj, obj, pred) in enumerate(rel.tolist()):
        slot = int(used[int(subj)].item())
        if slot >= num_slots:
            continue
        pred_targets[int(subj), slot] = int(pred)
        bind_targets[int(subj), slot] = int(obj)
        quality_targets[int(subj), slot] = rel_quality[rel_idx]
        used[int(subj)] += 1
    return {"pred": pred_targets, "bind": bind_targets, "quality": quality_targets}


def _slot_relation_assignments(cost: torch.Tensor, slot_assignment: str) -> list[tuple[int, int]]:
    if cost.ndim != 2:
        raise ValueError(f"Expected 2D assignment cost, got {tuple(cost.shape)}")
    num_slots, rel_count = cost.shape
    if num_slots == 0 or rel_count == 0:
        return []
    if slot_assignment == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError("Hungarian slot assignment requires scipy.") from exc
        cost_np = cost.detach().cpu().float().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        return [(int(slot), int(rel_idx)) for slot, rel_idx in zip(row_ind, col_ind)]
    if slot_assignment == "greedy":
        order = torch.argsort(cost.reshape(-1))
        used_slots: set[int] = set()
        used_rels: set[int] = set()
        assignments: list[tuple[int, int]] = []
        for flat_idx in order.tolist():
            slot = int(flat_idx // rel_count)
            rel_idx = int(flat_idx % rel_count)
            if slot in used_slots or rel_idx in used_rels:
                continue
            assignments.append((slot, rel_idx))
            used_slots.add(slot)
            used_rels.add(rel_idx)
            if len(used_slots) >= num_slots or len(used_rels) >= rel_count:
                break
        return assignments
    raise ValueError(f"Unsupported slot_assignment: {slot_assignment}")


def _build_dynamic_outgoing_slot_targets(
    num_objects: int,
    relations: torch.Tensor,
    *,
    num_predicates: int,
    num_slots: int,
    device: torch.device,
    pred_logits: torch.Tensor,
    binding_logits: torch.Tensor,
    target_centers: torch.Tensor,
    boxes: torch.Tensor,
    image_size: tuple[int, int],
    relation_quality: torch.Tensor | None = None,
    slot_assignment: str = "greedy",
    use_center_assignment_cost: bool = True,
) -> dict[str, torch.Tensor]:
    pred_targets = torch.zeros((num_objects, num_slots), dtype=torch.long, device=device)
    bind_targets = torch.full((num_objects, num_slots), -1, dtype=torch.long, device=device)
    quality_targets = torch.zeros((num_objects, num_slots), dtype=torch.float32, device=device)
    if num_objects == 0 or relations.numel() == 0:
        return {"pred": pred_targets, "bind": bind_targets, "quality": quality_targets}
    rel = relations.to(device=device, dtype=torch.long)
    rel_quality = (
        torch.ones((len(rel),), dtype=torch.float32, device=device)
        if relation_quality is None or len(relation_quality) != len(rel)
        else relation_quality.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    )
    valid = (rel[:, 0] >= 0) & (rel[:, 0] < num_objects)
    valid = valid & (rel[:, 1] >= 0) & (rel[:, 1] < num_objects)
    valid = valid & (rel[:, 0] != rel[:, 1])
    valid = valid & (rel[:, 2] > 0) & (rel[:, 2] <= num_predicates)
    rel = rel[valid]
    rel_quality = rel_quality[valid]
    if len(rel) == 0:
        return {"pred": pred_targets, "bind": bind_targets, "quality": quality_targets}

    object_centers = _object_geometry_features(boxes.to(device=device, dtype=torch.float32), image_size)[:, :2]
    pred_log_prob = F.log_softmax(pred_logits.float(), dim=-1).detach()
    bind_log_prob = F.log_softmax(binding_logits.float(), dim=-1).detach()
    center_cost = torch.cdist(
        target_centers.detach().reshape(-1, 2).float(),
        object_centers.float(),
        p=1,
    ).reshape(num_objects, num_slots, num_objects)

    for subj in rel[:, 0].unique(sorted=False).tolist():
        subj_idx = int(subj)
        subj_rel = rel[rel[:, 0] == subj_idx]
        if len(subj_rel) == 0:
            continue
        subj_quality = rel_quality[rel[:, 0] == subj_idx]
        pred_ids = subj_rel[:, 2].long()
        obj_ids = subj_rel[:, 1].long()
        cost = -pred_log_prob[subj_idx][:, pred_ids]
        cost = cost - bind_log_prob[subj_idx][:, obj_ids]
        if use_center_assignment_cost:
            cost = cost + 2.0 * center_cost[subj_idx][:, obj_ids]
        for slot, rel_idx in _slot_relation_assignments(cost, slot_assignment):
            pred_targets[subj_idx, slot] = int(pred_ids[rel_idx].item())
            bind_targets[subj_idx, slot] = int(obj_ids[rel_idx].item())
            quality_targets[subj_idx, slot] = subj_quality[rel_idx]
    return {"pred": pred_targets, "bind": bind_targets, "quality": quality_targets}


class DirectedRelationHead(nn.Module):
    """Subject-conditioned outgoing relation slots.

    The head predicts a small set of directed relation slots for every subject
    object. Predicate classification is performed on subject-slot tokens, and
    the object endpoint is selected through a learned target binding field.
    """

    def __init__(
        self,
        *,
        visual_dim: int,
        num_object_classes: int = 60,
        num_predicates: int = 59,
        num_slots: int = 4,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        spatial_bind_weight: float = 8.0,
        geometry_mode: str = "global_center",
        target_aware_predicate: bool = False,
        ranker: bool = False,
        memory_layers: int = 0,
        memory_heads: int = 8,
        memory_ffn_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_predicates = int(num_predicates)
        self.num_slots = max(1, int(num_slots))
        self.spatial_bind_weight = float(spatial_bind_weight)
        if geometry_mode not in RELATION_GEOMETRY_MODES:
            raise ValueError(f"Unsupported relation geometry mode: {geometry_mode}")
        self.geometry_mode = geometry_mode
        self.address_geometry_dim = _address_geometry_dim(geometry_mode)
        self.target_aware_predicate = bool(target_aware_predicate)
        self.ranker_enabled = bool(ranker)
        self.memory_layers = max(0, int(memory_layers))
        self.obj_embed = nn.Embedding(num_object_classes + 1, embed_dim)
        self.subject_proj = nn.Sequential(
            nn.Linear(visual_dim + embed_dim + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.object_key_proj = nn.Sequential(
            nn.Linear(visual_dim + embed_dim + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if self.memory_layers > 0:
            if hidden_dim % int(memory_heads) != 0:
                raise ValueError(f"relation hidden_dim={hidden_dim} must be divisible by memory_heads={memory_heads}")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(memory_heads),
                dim_feedforward=int(memory_ffn_dim) if int(memory_ffn_dim) > 0 else hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.memory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.memory_layers)
            self.memory_subject_norm = nn.LayerNorm(hidden_dim)
            self.memory_key_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.memory_key_norm = nn.LayerNorm(hidden_dim)
        else:
            self.memory_encoder = None
            self.memory_subject_norm = None
            self.memory_key_proj = None
            self.memory_key_norm = None
        self.slot_embed = nn.Embedding(self.num_slots, hidden_dim)
        self.slot_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.predicate = nn.Linear(hidden_dim, num_predicates + 1)
        if self.target_aware_predicate:
            self.pair_predicate = nn.Sequential(
                nn.Linear(hidden_dim * 4 + 8, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_predicates + 1),
            )
        else:
            self.pair_predicate = None
        if self.ranker_enabled:
            self.rank_score = nn.Sequential(
                nn.Linear(hidden_dim * 4 + 8, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.rank_score = None
        self.quality = nn.Linear(hidden_dim, 1)
        self.target_query = nn.Linear(hidden_dim, hidden_dim)
        self.target_delta = nn.Linear(hidden_dim, 2)
        if self.geometry_mode != "global_center":
            # The legacy center-offset branch is retained in the state dict for
            # checkpoint/API compatibility, but oriented/semantic addressing
            # does not consume it. Freezing prevents DDP unused-parameter
            # failures without enabling global-center supervision implicitly.
            for parameter in self.target_delta.parameters():
                parameter.requires_grad_(False)
        if self.address_geometry_dim > 0:
            self.geometry_mu = nn.Linear(hidden_dim, self.address_geometry_dim)
            nn.init.normal_(self.geometry_mu.weight, std=0.01)
            nn.init.zeros_(self.geometry_mu.bias)
        else:
            self.geometry_mu = None
        if self.geometry_mode == "subject_aligned_obb_anisotropic":
            self.geometry_log_scale = nn.Linear(hidden_dim, self.address_geometry_dim)
            nn.init.zeros_(self.geometry_log_scale.weight)
            # Local displacements are expressed in subject-axis units and can
            # naturally span several object widths; ratio/angle coordinates
            # are already O(1). Start from that parameterization prior and let
            # every slot learn its own anisotropy.
            nn.init.constant_(self.geometry_log_scale.bias, 0.0)
            with torch.no_grad():
                self.geometry_log_scale.bias[:2].fill_(math.log(2.0))
        else:
            self.geometry_log_scale = None
        nn.init.constant_(self.predicate.bias, -2.0)
        if self.pair_predicate is not None:
            nn.init.constant_(self.pair_predicate[-1].bias, -2.0)
        if self.rank_score is not None:
            nn.init.constant_(self.rank_score[-1].bias, 0.0)
        nn.init.constant_(self.quality.bias, -2.0)

    def _pair_features(
        self,
        *,
        slot_tokens: torch.Tensor,
        object_keys: torch.Tensor,
        object_geom: torch.Tensor,
        subj_idx: torch.Tensor,
        slot_idx: torch.Tensor,
        obj_idx: torch.Tensor,
    ) -> torch.Tensor:
        subj_slot = slot_tokens[subj_idx, slot_idx]
        obj_key = object_keys[obj_idx]
        pair_geom = _pair_geometry_features(object_geom.float(), subj_idx, obj_idx).to(dtype=subj_slot.dtype)
        return torch.cat(
            [
                subj_slot,
                obj_key,
                subj_slot * obj_key,
                (subj_slot - obj_key).abs(),
                pair_geom,
            ],
            dim=1,
        )

    def pair_predicate_logits(
        self,
        *,
        slot_tokens: torch.Tensor,
        object_keys: torch.Tensor,
        object_geom: torch.Tensor,
        subj_idx: torch.Tensor,
        slot_idx: torch.Tensor,
        obj_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.pair_predicate is None:
            raise RuntimeError("Target-aware predicate head is disabled")
        pair_features = self._pair_features(
            slot_tokens=slot_tokens,
            object_keys=object_keys,
            object_geom=object_geom,
            subj_idx=subj_idx,
            slot_idx=slot_idx,
            obj_idx=obj_idx,
        )
        return self.pair_predicate(pair_features)

    def pair_rank_logits(
        self,
        *,
        slot_tokens: torch.Tensor,
        object_keys: torch.Tensor,
        object_geom: torch.Tensor,
        subj_idx: torch.Tensor,
        slot_idx: torch.Tensor,
        obj_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.rank_score is None:
            raise RuntimeError("Relation ranker is disabled")
        pair_features = self._pair_features(
            slot_tokens=slot_tokens,
            object_keys=object_keys,
            object_geom=object_geom,
            subj_idx=subj_idx,
            slot_idx=slot_idx,
            obj_idx=obj_idx,
        )
        return self.rank_score(pair_features).squeeze(-1)

    def forward(
        self,
        *,
        object_features: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        num_objects = len(labels)
        if num_objects == 0:
            device = object_features.device
            return {
                "predicate_logits": object_features.new_zeros((0, self.num_slots, self.num_predicates + 1)),
                "quality_logits": object_features.new_zeros((0, self.num_slots)),
                "binding_logits": object_features.new_zeros((0, self.num_slots, 0)),
                "target_centers": object_features.new_zeros((0, self.num_slots, 2)),
                "slot_tokens": object_features.new_zeros((0, self.num_slots, self.slot_embed.embedding_dim)),
                "object_keys": object_features.new_zeros((0, self.slot_embed.embedding_dim)),
                "object_geom": object_features.new_zeros((0, 6)),
                "address_geometry": object_features.new_zeros((0, 0, self.address_geometry_dim)),
                "geometry_mu": object_features.new_zeros((0, self.num_slots, self.address_geometry_dim)),
                "geometry_scale": object_features.new_zeros((0, self.num_slots, self.address_geometry_dim)),
            }
        geom = _object_geometry_features(boxes, image_size).to(device=object_features.device, dtype=object_features.dtype)
        label_emb = self.obj_embed(labels.long().clamp(min=0)).to(dtype=object_features.dtype)
        packed = torch.cat([object_features, label_emb, geom], dim=1)
        subject = self.subject_proj(packed)
        keys = self.object_key_proj(packed)
        if self.memory_encoder is not None:
            memory = self.memory_encoder(subject[None]).squeeze(0)
            subject = self.memory_subject_norm(subject + memory)
            keys = self.memory_key_norm(keys + self.memory_key_proj(memory))
        slot_tokens = subject[:, None, :] + self.slot_embed.weight[None, :, :].to(dtype=subject.dtype)
        slot_tokens = self.slot_mlp(slot_tokens)
        predicate_logits = self.predicate(slot_tokens)
        quality_logits = self.quality(slot_tokens).squeeze(-1)
        target_query = self.target_query(slot_tokens)
        appearance_logits = torch.einsum("nkd,md->nkm", target_query, keys) / math.sqrt(float(keys.shape[-1]))
        delta = torch.tanh(self.target_delta(slot_tokens)) * 0.75
        subject_centers = geom[:, :2]
        target_centers = (subject_centers[:, None, :] + delta).clamp(0.0, 1.0)
        address_geometry = _address_geometry_features(
            boxes.to(device=object_features.device, dtype=torch.float32),
            image_size,
            self.geometry_mode,
        ).to(dtype=appearance_logits.dtype)
        geometry_mu = appearance_logits.new_zeros((num_objects, self.num_slots, self.address_geometry_dim))
        geometry_scale = appearance_logits.new_ones((num_objects, self.num_slots, self.address_geometry_dim))
        if self.geometry_mode == "semantic_only":
            binding_logits = appearance_logits
        elif self.geometry_mode == "global_center":
            distance = torch.cdist(target_centers.reshape(-1, 2).float(), subject_centers.float(), p=1)
            distance = distance.reshape(num_objects, self.num_slots, num_objects).to(dtype=appearance_logits.dtype)
            binding_logits = appearance_logits - self.spatial_bind_weight * distance
        else:
            if self.geometry_mu is None:
                raise RuntimeError(f"Geometry prototype head is missing for mode {self.geometry_mode}")
            geometry_mu = self.geometry_mu(slot_tokens)
            residual = (address_geometry[:, None, :, :] - geometry_mu[:, :, None, :]).abs()
            if self.geometry_log_scale is not None:
                geometry_scale = torch.exp(self.geometry_log_scale(slot_tokens).clamp(-3.0, 3.0))
                geometry_components = (
                    residual / geometry_scale[:, :, None, :]
                    + torch.log(geometry_scale[:, :, None, :])
                )
            else:
                geometry_components = residual
            # Keep lambda calibrated to the legacy two-coordinate center
            # potential instead of letting a six-dimensional OBB encoding
            # receive three times the energy solely because it has more terms.
            geometry_energy = 2.0 * geometry_components.mean(dim=-1)
            binding_logits = appearance_logits - self.spatial_bind_weight * geometry_energy
        eye = torch.eye(num_objects, dtype=torch.bool, device=object_features.device)
        binding_logits = binding_logits.masked_fill(eye[:, None, :], -1.0e4)
        return {
            "predicate_logits": predicate_logits,
            "quality_logits": quality_logits,
            "binding_logits": binding_logits,
            "target_centers": target_centers,
            "slot_tokens": slot_tokens,
            "object_keys": keys,
            "object_geom": geom,
            "address_geometry": address_geometry,
            "geometry_mu": geometry_mu,
            "geometry_scale": geometry_scale,
        }


class ORSGGDirectedNet(nn.Module):
    """One-stage OBB SGG with node-centric directed relation generation."""

    def __init__(
        self,
        *,
        num_object_classes: int = 60,
        num_predicates: int = 59,
        detector_width: int = 64,
        detector_backbone: str = "tiny",
        detector_context_class_ids: Sequence[int] | None = None,
        detector_context_loss_weight: float = 1.0,
        detector_context_queries_per_class: int = 1,
        detector_context_target_mode: str = "largest",
        detector_dinov3_pretrained: bool = True,
        detector_freeze_backbone: bool = True,
        detector_dinov3_trainable_stages: int = 0,
        detector_dinov3_trainable_blocks: int = 0,
        detector_box_loss_weight: float = 0.0,
        detector_gwd_loss_weight: float = 0.0,
        detector_decode_labels_per_cell: int = 0,
        detector_head_hidden: int = 128,
        detector_head_layers: int = 2,
        relation_slots: int = 4,
        relation_feature_pooling: str = "center",
        relation_geometry_mode: str = "global_center",
        relation_hidden_dim: int = 256,
        relation_memory_layers: int = 0,
        relation_memory_heads: int = 8,
        relation_memory_ffn_dim: int = 0,
        relation_loss_weight: float = 1.0,
        relation_bind_loss_weight: float = 1.0,
        relation_quality_loss_weight: float = 0.5,
        relation_offset_loss_weight: float = 2.0,
        relation_rank_loss_weight: float = 0.5,
        relation_target_aware_predicate: bool = False,
        relation_ranker: bool = False,
        relation_predicate_negatives_per_positive: int = 3,
        relation_target_aware_aux_weight: float = 0.25,
    ) -> None:
        super().__init__()
        detector_cls = MultiScaleOBBDetector if detector_backbone == "dinov3_convnext_tiny_pyramid" else TinyOBBDetector
        self.detector = detector_cls(
            num_classes=num_object_classes,
            width=detector_width,
            backbone=detector_backbone,
            context_class_ids=detector_context_class_ids,
            context_loss_weight=detector_context_loss_weight,
            context_queries_per_class=detector_context_queries_per_class,
            context_target_mode=detector_context_target_mode,
            dinov3_pretrained=detector_dinov3_pretrained,
            freeze_backbone=detector_freeze_backbone,
            dinov3_trainable_stages=detector_dinov3_trainable_stages,
            dinov3_trainable_blocks=detector_dinov3_trainable_blocks,
            box_loss_weight=detector_box_loss_weight,
            gwd_loss_weight=detector_gwd_loss_weight,
            head_hidden=detector_head_hidden,
            head_layers=detector_head_layers,
        )
        if relation_feature_pooling not in {"center", "obb5"}:
            raise ValueError(f"Unsupported relation_feature_pooling: {relation_feature_pooling}")
        self.relation_feature_pooling = relation_feature_pooling
        self.detector_decode_labels_per_cell = max(0, int(detector_decode_labels_per_cell))
        self.relation_head = DirectedRelationHead(
            visual_dim=self.detector.backbone.out_channels,
            num_object_classes=num_object_classes,
            num_predicates=num_predicates,
            num_slots=relation_slots,
            hidden_dim=relation_hidden_dim,
            geometry_mode=relation_geometry_mode,
            target_aware_predicate=relation_target_aware_predicate,
            ranker=relation_ranker,
            memory_layers=relation_memory_layers,
            memory_heads=relation_memory_heads,
            memory_ffn_dim=relation_memory_ffn_dim,
        )
        self.num_predicates = int(num_predicates)
        self.relation_slots = int(relation_slots)
        self.relation_geometry_mode = relation_geometry_mode
        self.relation_loss_weight = float(relation_loss_weight)
        self.relation_bind_loss_weight = float(relation_bind_loss_weight)
        self.relation_quality_loss_weight = float(relation_quality_loss_weight)
        self.relation_offset_loss_weight = float(relation_offset_loss_weight)
        self.relation_rank_loss_weight = float(relation_rank_loss_weight)
        self.relation_target_aware_predicate = bool(relation_target_aware_predicate)
        self.relation_ranker = bool(relation_ranker)
        self.relation_predicate_negatives_per_positive = max(0, int(relation_predicate_negatives_per_positive))
        self.relation_target_aware_aux_weight = float(relation_target_aware_aux_weight)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.detector(images)

    def _target_aware_pair_losses(
        self,
        *,
        out: Dict[str, torch.Tensor],
        pred_targets: torch.Tensor,
        bind_targets: torch.Tensor,
        quality_targets: torch.Tensor,
        slot_loss: torch.Tensor,
        weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = bind_targets >= 0
        if not bool(pos.any()):
            return slot_loss, slot_loss.sum() * 0.0

        device = pred_targets.device
        subj_indices: list[int] = []
        slot_indices: list[int] = []
        obj_indices: list[int] = []
        labels: list[int] = []
        rank_targets: list[float] = []
        positive_slots = torch.nonzero(pos, as_tuple=False)
        neg_per_pos = self.relation_predicate_negatives_per_positive
        for subj, slot in positive_slots.tolist():
            obj = int(bind_targets[subj, slot].item())
            pred = int(pred_targets[subj, slot].item())
            subj_indices.append(int(subj))
            slot_indices.append(int(slot))
            obj_indices.append(obj)
            labels.append(pred)
            rank_targets.append(float(quality_targets[subj, slot].detach().clamp(0.0, 1.0).item()))
            if neg_per_pos <= 0:
                continue
            scores = out["binding_logits"][subj, slot].detach().float().clone()
            scores[int(subj)] = -float("inf")
            positive_objects = bind_targets[subj][bind_targets[subj] >= 0].long()
            if len(positive_objects) > 0:
                scores[positive_objects] = -float("inf")
            valid_neg = torch.isfinite(scores)
            if not bool(valid_neg.any()):
                continue
            neg_k = min(neg_per_pos, int(valid_neg.sum().item()))
            neg_idx = torch.topk(scores, k=neg_k).indices
            for neg_obj in neg_idx.tolist():
                subj_indices.append(int(subj))
                slot_indices.append(int(slot))
                obj_indices.append(int(neg_obj))
                labels.append(0)
                rank_targets.append(0.0)
        if not subj_indices:
            return slot_loss, slot_loss.sum() * 0.0

        subj_idx = torch.tensor(subj_indices, dtype=torch.long, device=device)
        slot_idx = torch.tensor(slot_indices, dtype=torch.long, device=device)
        obj_idx = torch.tensor(obj_indices, dtype=torch.long, device=device)
        pair_labels = torch.tensor(labels, dtype=torch.long, device=device)
        pair_rank_targets = torch.tensor(rank_targets, dtype=torch.float32, device=device)
        pair_logits = self.relation_head.pair_predicate_logits(
            slot_tokens=out["slot_tokens"],
            object_keys=out["object_keys"],
            object_geom=out["object_geom"],
            subj_idx=subj_idx,
            slot_idx=slot_idx,
            obj_idx=obj_idx,
        ).float()
        pair_loss = F.cross_entropy(pair_logits, pair_labels, weight=weight)
        aux_weight = min(max(self.relation_target_aware_aux_weight, 0.0), 1.0)
        pred_loss = (1.0 - aux_weight) * pair_loss + aux_weight * slot_loss
        if self.relation_ranker:
            rank_logits = self.relation_head.pair_rank_logits(
                slot_tokens=out["slot_tokens"],
                object_keys=out["object_keys"],
                object_geom=out["object_geom"],
                subj_idx=subj_idx,
                slot_idx=slot_idx,
                obj_idx=obj_idx,
            ).float()
            rank_loss = F.binary_cross_entropy_with_logits(rank_logits, pair_rank_targets)
        else:
            rank_loss = pred_loss.sum() * 0.0
        return pred_loss, rank_loss

    def _relation_loss_for_objects(
        self,
        *,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        relations: torch.Tensor,
        image_size: tuple[int, int],
        predicate_class_weights: torch.Tensor | None,
        dynamic_slot_assignment: bool = True,
        slot_assignment: str = "greedy",
        relation_quality: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        device = feature_map.device
        if len(labels) < 2:
            zero = feature_map.sum() * 0.0
            return {
                "loss_relation": zero,
                "loss_rel_pred": zero,
                "loss_rel_bind": zero,
                "loss_rel_quality": zero,
                "loss_rel_offset": zero,
                "loss_rel_rank": zero,
                "rel_slots": zero.detach(),
                "rel_pos": zero.detach(),
            }
        object_features = _sample_object_features(
            feature_map.float(),
            boxes.to(device=device),
            self.detector.stride,
            self.relation_feature_pooling,
        )
        out = self.relation_head(
            object_features=object_features,
            labels=labels.to(device=device, dtype=torch.long),
            boxes=boxes.to(device=device, dtype=torch.float32),
            image_size=image_size,
        )
        pred_logits = out["predicate_logits"].float()
        if dynamic_slot_assignment:
            targets = _build_dynamic_outgoing_slot_targets(
                len(labels),
                relations,
                num_predicates=self.num_predicates,
                num_slots=self.relation_slots,
                device=device,
                pred_logits=pred_logits,
                binding_logits=out["binding_logits"],
                target_centers=out["target_centers"],
                boxes=boxes,
                image_size=image_size,
                relation_quality=relation_quality,
                slot_assignment=slot_assignment,
                use_center_assignment_cost=self.relation_geometry_mode == "global_center",
            )
        else:
            targets = _build_outgoing_slot_targets(
                len(labels),
                relations,
                num_predicates=self.num_predicates,
                num_slots=self.relation_slots,
                device=device,
                relation_quality=relation_quality,
            )
        pred_targets = targets["pred"]
        weight = None
        if predicate_class_weights is not None:
            weight = predicate_class_weights.to(device=device, dtype=torch.float32)
        slot_loss_pred = F.cross_entropy(
            pred_logits.reshape(-1, self.num_predicates + 1),
            pred_targets.reshape(-1),
            weight=weight,
        )
        quality_targets = targets["quality"].float()
        loss_quality = F.binary_cross_entropy_with_logits(out["quality_logits"].float(), quality_targets)
        pos = targets["bind"] >= 0
        if self.relation_target_aware_predicate:
            loss_pred, loss_rank = self._target_aware_pair_losses(
                out=out,
                pred_targets=pred_targets,
                bind_targets=targets["bind"],
                quality_targets=quality_targets,
                slot_loss=slot_loss_pred,
                weight=weight,
            )
        else:
            loss_pred = slot_loss_pred
            loss_rank = slot_loss_pred.sum() * 0.0
        if bool(pos.any()):
            binding_logits = out["binding_logits"].float()[pos]
            bind_targets = targets["bind"][pos]
            loss_bind = F.cross_entropy(binding_logits, bind_targets)
            if self.relation_geometry_mode == "semantic_only":
                loss_offset = pred_logits.sum() * 0.0
            elif self.relation_geometry_mode == "global_center":
                geom = _object_geometry_features(boxes.to(device=device, dtype=torch.float32), image_size)
                target_centers = geom[bind_targets, :2].to(device=device, dtype=out["target_centers"].dtype)
                loss_offset = F.smooth_l1_loss(out["target_centers"][pos], target_centers)
            else:
                positive_slots = torch.nonzero(pos, as_tuple=False)
                geometry_targets = out["address_geometry"][
                    positive_slots[:, 0],
                    bind_targets,
                ].to(dtype=out["geometry_mu"].dtype)
                geometry_prediction = out["geometry_mu"][pos]
                if self.relation_geometry_mode == "subject_aligned_obb_anisotropic":
                    geometry_scale = out["geometry_scale"][pos].clamp_min(0.05)
                    loss_offset = (
                        (geometry_targets - geometry_prediction).abs() / geometry_scale
                        + torch.log(geometry_scale)
                    ).mean()
                else:
                    loss_offset = F.smooth_l1_loss(geometry_prediction, geometry_targets)
        else:
            loss_bind = pred_logits.sum() * 0.0
            loss_offset = pred_logits.sum() * 0.0
        loss_relation = (
            loss_pred
            + self.relation_bind_loss_weight * loss_bind
            + self.relation_quality_loss_weight * loss_quality
            + self.relation_offset_loss_weight * loss_offset
            + self.relation_rank_loss_weight * loss_rank
        )
        return {
            "loss_relation": loss_relation,
            "loss_rel_pred": loss_pred,
            "loss_rel_bind": loss_bind,
            "loss_rel_quality": loss_quality,
            "loss_rel_offset": loss_offset,
            "loss_rel_rank": loss_rank,
            "rel_slots": torch.tensor(float(len(labels) * self.relation_slots), device=device),
            "rel_pos": torch.tensor(float(pos.sum().item()), device=device),
        }

    def _relation_loss_from_targets(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[SceneGraphTarget],
        *,
        predicate_class_weights: torch.Tensor | None,
        dynamic_slot_assignment: bool,
        slot_assignment: str,
    ) -> Dict[str, torch.Tensor]:
        features = outputs["features"]
        losses: list[Dict[str, torch.Tensor]] = []
        for image_idx, target in enumerate(targets):
            losses.append(
                self._relation_loss_for_objects(
                    feature_map=features[image_idx],
                    boxes=target.boxes,
                    labels=target.labels,
                    relations=target.relations,
                    image_size=(target.width, target.height),
                    predicate_class_weights=predicate_class_weights,
                    dynamic_slot_assignment=dynamic_slot_assignment,
                    slot_assignment=slot_assignment,
                )
            )
        keys = [
            "loss_relation",
            "loss_rel_pred",
            "loss_rel_bind",
            "loss_rel_quality",
            "loss_rel_offset",
            "loss_rel_rank",
            "rel_slots",
            "rel_pos",
        ]
        return {key: torch.stack([item[key] for item in losses]).mean() for key in keys}

    def _relation_loss_from_detections(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[SceneGraphTarget],
        *,
        predicate_class_weights: torch.Tensor | None,
        top_objects: int,
        match_iou: float,
        dynamic_slot_assignment: bool,
        slot_assignment: str,
    ) -> Dict[str, torch.Tensor]:
        features = outputs["features"]
        image_sizes = [(target.width, target.height) for target in targets]
        labels_per_cell = self.detector_decode_labels_per_cell or None
        detections = self.detector.decode(
            outputs,
            topk=top_objects,
            image_sizes=image_sizes,
            labels_per_cell=labels_per_cell,
        )
        losses: list[Dict[str, torch.Tensor]] = []
        total_matched = 0
        total_detections = 0
        for image_idx, (det, target) in enumerate(zip(detections, targets)):
            device = features.device
            det_boxes = det.boxes.to(device=device, dtype=torch.float32)
            det_labels = det.labels.to(device=device, dtype=torch.long)
            det_scores = det.scores.to(device=device, dtype=torch.float32)
            total_detections += len(det_labels)
            if len(det_labels) < 2 or len(target.labels) == 0:
                losses.append(
                    self._relation_loss_for_objects(
                        feature_map=features[image_idx],
                        boxes=det_boxes,
                        labels=det_labels,
                        relations=torch.zeros((0, 3), dtype=torch.long, device=device),
                        image_size=(target.width, target.height),
                        predicate_class_weights=predicate_class_weights,
                        dynamic_slot_assignment=dynamic_slot_assignment,
                        slot_assignment=slot_assignment,
                    )
                )
                continue
            gt_boxes = target.boxes.to(device=device, dtype=torch.float32)
            gt_labels = target.labels.to(device=device, dtype=torch.long)
            ious = obb_iou(det_boxes, gt_boxes)
            ious = ious.masked_fill(det_labels[:, None] != gt_labels[None, :], -1.0)
            best_iou, best_gt = ious.max(dim=1)
            det_to_gt = best_gt.long()
            det_to_gt[best_iou < match_iou] = -1
            gt_to_det = torch.full((len(gt_labels),), -1, dtype=torch.long, device=device)
            gt_to_score = torch.full((len(gt_labels),), -1.0, dtype=torch.float32, device=device)
            gt_to_iou = torch.zeros((len(gt_labels),), dtype=torch.float32, device=device)
            for det_idx in range(len(det_to_gt)):
                gt_idx = int(det_to_gt[det_idx].item())
                if gt_idx < 0:
                    continue
                if float(det_scores[det_idx].item()) > float(gt_to_score[gt_idx].item()):
                    gt_to_det[gt_idx] = det_idx
                    gt_to_score[gt_idx] = det_scores[det_idx]
                    gt_to_iou[gt_idx] = best_iou[det_idx].clamp(0.0, 1.0)
            total_matched += int((gt_to_det >= 0).sum().item())
            mapped_rel = []
            mapped_quality = []
            if target.relations.numel() > 0:
                rel = target.relations.to(device=device, dtype=torch.long)
                valid = (rel[:, 0] >= 0) & (rel[:, 0] < len(gt_to_det))
                valid = valid & (rel[:, 1] >= 0) & (rel[:, 1] < len(gt_to_det))
                valid = valid & (rel[:, 2] > 0) & (rel[:, 2] <= self.num_predicates)
                rel = rel[valid]
                for subj, obj, pred in rel.tolist():
                    det_subj = int(gt_to_det[int(subj)].item())
                    det_obj = int(gt_to_det[int(obj)].item())
                    if det_subj >= 0 and det_obj >= 0 and det_subj != det_obj:
                        mapped_rel.append([det_subj, det_obj, int(pred)])
                        endpoint_quality = torch.minimum(gt_to_iou[int(subj)], gt_to_iou[int(obj)])
                        mapped_quality.append(float(endpoint_quality.item()))
            rel_tensor = (
                torch.tensor(mapped_rel, dtype=torch.long, device=device)
                if mapped_rel
                else torch.zeros((0, 3), dtype=torch.long, device=device)
            )
            rel_quality = (
                torch.tensor(mapped_quality, dtype=torch.float32, device=device)
                if mapped_quality
                else torch.zeros((0,), dtype=torch.float32, device=device)
            )
            losses.append(
                self._relation_loss_for_objects(
                    feature_map=features[image_idx],
                    boxes=det_boxes,
                    labels=det_labels,
                    relations=rel_tensor,
                    image_size=(target.width, target.height),
                    predicate_class_weights=predicate_class_weights,
                    dynamic_slot_assignment=dynamic_slot_assignment,
                    slot_assignment=slot_assignment,
                    relation_quality=rel_quality,
                )
            )
        keys = [
            "loss_relation",
            "loss_rel_pred",
            "loss_rel_bind",
            "loss_rel_quality",
            "loss_rel_offset",
            "loss_rel_rank",
            "rel_slots",
            "rel_pos",
        ]
        out = {key: torch.stack([item[key] for item in losses]).mean() for key in keys}
        device = features.device
        out["rel_matched_det"] = torch.tensor(float(total_matched), device=device)
        out["rel_total_det"] = torch.tensor(float(total_detections), device=device)
        return out

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[SceneGraphTarget],
        *,
        detector_class_weights: torch.Tensor | None = None,
        predicate_class_weights: torch.Tensor | None = None,
        detector_loss_weight: float = 1.0,
        relation_train_mode: str = "gt",
        relation_detected_top_objects: int = 100,
        relation_detected_match_iou: float = 0.25,
        relation_detected_loss_weight: float = 0.5,
        dynamic_slot_assignment: bool = True,
        relation_slot_assignment: str = "greedy",
    ) -> Dict[str, torch.Tensor]:
        if relation_slot_assignment not in {"greedy", "hungarian"}:
            raise ValueError(f"Unsupported relation_slot_assignment: {relation_slot_assignment}")
        losses = self.detector.loss(outputs, targets, class_weights=detector_class_weights)
        detector_loss = losses["loss"]
        if relation_train_mode == "gt":
            relation_losses = self._relation_loss_from_targets(
                outputs,
                targets,
                predicate_class_weights=predicate_class_weights,
                dynamic_slot_assignment=dynamic_slot_assignment,
                slot_assignment=relation_slot_assignment,
            )
        elif relation_train_mode == "detected":
            relation_losses = self._relation_loss_from_detections(
                outputs,
                targets,
                predicate_class_weights=predicate_class_weights,
                top_objects=relation_detected_top_objects,
                match_iou=relation_detected_match_iou,
                dynamic_slot_assignment=dynamic_slot_assignment,
                slot_assignment=relation_slot_assignment,
            )
        elif relation_train_mode == "mixed":
            gt_losses = self._relation_loss_from_targets(
                outputs,
                targets,
                predicate_class_weights=predicate_class_weights,
                dynamic_slot_assignment=dynamic_slot_assignment,
                slot_assignment=relation_slot_assignment,
            )
            det_losses = self._relation_loss_from_detections(
                outputs,
                targets,
                predicate_class_weights=predicate_class_weights,
                top_objects=relation_detected_top_objects,
                match_iou=relation_detected_match_iou,
                dynamic_slot_assignment=dynamic_slot_assignment,
                slot_assignment=relation_slot_assignment,
            )
            det_weight = float(relation_detected_loss_weight)
            relation_losses = {}
            loss_keys = [
                "loss_relation",
                "loss_rel_pred",
                "loss_rel_bind",
                "loss_rel_quality",
                "loss_rel_offset",
                "loss_rel_rank",
            ]
            metric_keys = ["rel_slots", "rel_pos"]
            for key in loss_keys:
                relation_losses[key] = gt_losses[key] + det_weight * det_losses[key]
            for key in metric_keys:
                relation_losses[key] = gt_losses[key] + det_losses[key]
            relation_losses["rel_matched_det"] = det_losses["rel_matched_det"]
            relation_losses["rel_total_det"] = det_losses["rel_total_det"]
        else:
            raise ValueError(f"Unsupported relation train mode: {relation_train_mode}")
        losses.update(relation_losses)
        losses["loss"] = (
            float(detector_loss_weight) * detector_loss
            + self.relation_loss_weight * relation_losses["loss_relation"]
        )
        return losses

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        *,
        image_sizes: Sequence[tuple[int, int]],
        top_objects: int = 200,
        max_pairs: int = 10000,
        score_thr: float = 0.0,
        object_score_power: float = 1.0,
        binding_topk: int = 1,
        predicate_temperature: float = 1.0,
        quality_power: float = 1.0,
        binding_score_power: float = 1.0,
        rank_score_power: float = 1.0,
        target_aware_blend: float = 1.0,
        predicate_score_weights: torch.Tensor | None = None,
        detector_labels_per_cell: int | None = None,
    ) -> List[SceneGraphPrediction]:
        self.eval()
        det_outputs = self.detector(images)
        labels_per_cell = self.detector_decode_labels_per_cell if detector_labels_per_cell is None else int(detector_labels_per_cell)
        detections = self.detector.decode(
            det_outputs,
            topk=top_objects,
            image_sizes=image_sizes,
            labels_per_cell=labels_per_cell if labels_per_cell > 0 else None,
        )
        features = det_outputs["features"]
        predictions: List[SceneGraphPrediction] = []
        for image_idx, (det, image_size) in enumerate(zip(detections, image_sizes)):
            keep = det.scores >= score_thr
            boxes = det.boxes[keep]
            labels = det.labels[keep]
            scores = det.scores[keep]
            object_scores = scores.clamp_min(1e-6).pow(object_score_power)
            n = len(labels)
            if n < 2:
                predictions.append(
                    SceneGraphPrediction(
                        boxes=boxes,
                        labels=labels,
                        object_scores=object_scores,
                        rel_pairs=torch.zeros((0, 2), dtype=torch.long, device=images.device),
                        rel_scores=torch.zeros((0, self.num_predicates + 1), dtype=torch.float32, device=images.device),
                    )
                )
                continue
            object_features = _sample_object_features(
                features[image_idx].float(),
                boxes,
                self.detector.stride,
                self.relation_feature_pooling,
            )
            out = self.relation_head(
                object_features=object_features,
                labels=labels,
                boxes=boxes,
                image_size=image_size,
            )
            bind_probs = torch.softmax(out["binding_logits"].float(), dim=-1)
            bind_k = min(max(int(binding_topk), 1), max(n - 1, 1))
            bind_scores, bind_idx = torch.topk(bind_probs, k=bind_k, dim=-1)
            quality = torch.sigmoid(out["quality_logits"].float())
            subj_idx = (
                torch.arange(n, device=images.device)[:, None, None]
                .expand(n, self.relation_slots, bind_k)
                .reshape(-1)
            )
            slot_idx = (
                torch.arange(self.relation_slots, device=images.device)[None, :, None]
                .expand(n, self.relation_slots, bind_k)
                .reshape(-1)
            )
            obj_idx = bind_idx.reshape(-1)
            if self.relation_target_aware_predicate:
                predicate_logits = self.relation_head.pair_predicate_logits(
                    slot_tokens=out["slot_tokens"],
                    object_keys=out["object_keys"],
                    object_geom=out["object_geom"],
                    subj_idx=subj_idx,
                    slot_idx=slot_idx,
                    obj_idx=obj_idx,
                ).float()
                temperature = max(float(predicate_temperature), 1e-3)
                pair_scores = torch.softmax(predicate_logits / temperature, dim=-1)
                slot_scores = torch.softmax(out["predicate_logits"].float() / temperature, dim=-1)
                slot_scores = (
                    slot_scores[:, :, None, :]
                    .expand(n, self.relation_slots, bind_k, self.num_predicates + 1)
                    .reshape(-1, self.num_predicates + 1)
                )
                blend = min(max(float(target_aware_blend), 0.0), 1.0)
                rel_scores = blend * pair_scores + (1.0 - blend) * slot_scores
            else:
                temperature = max(float(predicate_temperature), 1e-3)
                predicate_scores = torch.softmax(out["predicate_logits"].float() / temperature, dim=-1)
                rel_scores = (
                    predicate_scores[:, :, None, :]
                    .expand(n, self.relation_slots, bind_k, self.num_predicates + 1)
                    .reshape(-1, self.num_predicates + 1)
                )
            quality_gate = quality.clamp_min(1e-6).pow(float(quality_power))
            bind_gate = bind_scores.clamp_min(1e-6).pow(float(binding_score_power))
            rel_gate = (quality_gate[:, :, None] * bind_gate).reshape(-1)
            if self.relation_ranker:
                rank_logits = self.relation_head.pair_rank_logits(
                    slot_tokens=out["slot_tokens"],
                    object_keys=out["object_keys"],
                    object_geom=out["object_geom"],
                    subj_idx=subj_idx,
                    slot_idx=slot_idx,
                    obj_idx=obj_idx,
                ).float()
                rank_gate = torch.sigmoid(rank_logits).clamp_min(1e-6).pow(float(rank_score_power))
                rel_gate = rel_gate * rank_gate
            rel_scores = rel_scores.clone()
            rel_scores[:, 1:] = rel_scores[:, 1:] * rel_gate[:, None]
            if predicate_score_weights is not None:
                weights = predicate_score_weights.to(device=rel_scores.device, dtype=rel_scores.dtype)
                rel_scores = rel_scores * weights.view(1, -1)
            valid = subj_idx != obj_idx
            pairs = torch.stack([subj_idx[valid], obj_idx[valid]], dim=1)
            rel_scores = rel_scores[valid]
            if len(pairs) > max_pairs:
                pair_rank = rel_scores[:, 1:].max(dim=1).values
                order = torch.topk(pair_rank, k=max_pairs).indices
                pairs = pairs[order]
                rel_scores = rel_scores[order]
            predictions.append(
                SceneGraphPrediction(
                    boxes=boxes,
                    labels=labels,
                    object_scores=object_scores,
                    rel_pairs=pairs.long(),
                    rel_scores=rel_scores,
                )
            )
        return predictions
