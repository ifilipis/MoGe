from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import lsmr
import utils3d

from .geometry_torch import normalized_view_plane_uv, recover_focal_shift
from .panorama import grad_equation, poisson_equation


def fov_x_from_pinhole(focal_length_mm: float, sensor_width_mm: float) -> float:
    return float(np.rad2deg(2.0 * np.arctan(sensor_width_mm / (2.0 * focal_length_mm))))


def internal_points_to_depth(
    points_signal: torch.Tensor,
    fov_x: Optional[float] = None,
    metric_scale: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert MoGe's raw affine point-map signal to metric depth."""
    if points_signal.dim() == 3:
        points_signal = points_signal.unsqueeze(0)
        omit_batch_dim = True
    else:
        omit_batch_dim = False

    batch_size, height, width, _ = points_signal.shape
    aspect_ratio = width / height
    points_camera = points_signal.float().clone()
    mask_binary = None if mask is None else mask.bool()
    if mask_binary is not None and mask_binary.dim() == 2:
        mask_binary = mask_binary.unsqueeze(0)

    if fov_x is None:
        focal, shift = recover_focal_shift(points_camera, mask_binary)
    else:
        focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(
            torch.deg2rad(torch.as_tensor(fov_x, device=points_camera.device, dtype=points_camera.dtype) / 2)
        )
        if focal.ndim == 0:
            focal = focal[None].expand(batch_size)
        _, shift = recover_focal_shift(points_camera, mask_binary, focal=focal)

    fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
    fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
    intrinsics = utils3d.pt.intrinsics_from_focal_center(
        fx,
        fy,
        torch.tensor(0.5, device=points_camera.device, dtype=points_camera.dtype),
        torch.tensor(0.5, device=points_camera.device, dtype=points_camera.dtype),
    )

    points_camera[..., 2] += shift[..., None, None]
    depth = points_camera[..., 2].clone()
    if metric_scale is not None:
        if metric_scale.dim() == 0:
            metric_scale = metric_scale[None]
        depth = depth * metric_scale[:, None, None].float()
        points_camera = points_camera * metric_scale[:, None, None, None].float()

    points_projected = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)

    if omit_batch_dim:
        return depth.squeeze(0), points_projected.squeeze(0), intrinsics.squeeze(0)
    return depth, points_projected, intrinsics


def optimize_internal_point_depth_residual(
    points_signal: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    fov_x: Optional[float] = None,
    metric_scale: Optional[torch.Tensor] = None,
    model_mask: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    fit_mask: Optional[torch.Tensor] = None,
    fit_weight: Optional[torch.Tensor] = None,
    num_steps: int = 300,
    lr: float = 0.05,
    data_weight: float = 100.0,
    smooth_weight: float = 0.05,
    anchor_weight: float = 0.01,
    optimize_scale: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimize a residual on MoGe's internal point-map signal, not final depth."""
    if points_signal.dim() == 4:
        if points_signal.shape[0] != 1:
            raise ValueError("Guided internal optimization currently expects a single image.")
        points_signal = points_signal.squeeze(0)
    base_points = points_signal.float().detach()
    with torch.no_grad():
        pred_depth, _, _ = internal_points_to_depth(base_points, fov_x=fov_x, metric_scale=metric_scale, mask=model_mask)
    pred_depth = pred_depth.clamp_min(eps)
    target_depth = target_depth.float().clamp_min(eps)
    target_mask = target_mask.bool() & torch.isfinite(target_depth) & (target_depth > eps)
    if valid_mask is not None:
        target_mask = target_mask & valid_mask.bool()
    if fit_mask is None:
        fit_mask = target_mask
    else:
        fit_mask = fit_mask.bool() & target_mask
    if fit_weight is None:
        fit_weight = torch.ones_like(pred_depth, dtype=torch.float32)
    else:
        fit_weight = fit_weight.float().clamp_min(0)

    if not target_mask.any():
        raise ValueError("Guided depth mask has no finite positive target depth pixels.")
    if not fit_mask.any():
        raise ValueError("Guided depth fit mask has no finite positive target depth pixels.")

    log_pred = pred_depth.log().detach()
    log_target = target_depth.log().detach()
    initial_log_scale = (log_pred[fit_mask].median() - log_target[fit_mask].median()).detach()
    z_residual = torch.zeros_like(pred_depth, requires_grad=True)
    log_scale = initial_log_scale.clone().requires_grad_(optimize_scale)
    params = [z_residual, log_scale] if optimize_scale else [z_residual]
    optimizer = torch.optim.Adam(params, lr=lr)

    outside_mask = ~target_mask
    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        corrected_points = base_points.clone()
        corrected_points[..., 2] = corrected_points[..., 2] + z_residual
        corrected_depth, _, _ = internal_points_to_depth(corrected_points, fov_x=fov_x, metric_scale=metric_scale, mask=model_mask)
        corrected_depth = corrected_depth.clamp_min(eps)
        scaled_log_target = log_target + log_scale
        corrected_log_depth = corrected_depth.log()
        residual_error = (corrected_log_depth[fit_mask] - scaled_log_target[fit_mask]).square()
        residual_weight = fit_weight[fit_mask]
        loss = data_weight * (residual_error * residual_weight).sum() / residual_weight.sum().clamp_min(eps)
        if smooth_weight > 0:
            loss = loss + smooth_weight * (
                (z_residual[:, 1:] - z_residual[:, :-1]).square().mean()
                + (z_residual[1:, :] - z_residual[:-1, :]).square().mean()
            )
        if anchor_weight > 0 and outside_mask.any():
            loss = loss + anchor_weight * z_residual[outside_mask].square().mean()
        loss.backward()
        optimizer.step()

    corrected_points = base_points.clone()
    corrected_points[..., 2] = corrected_points[..., 2] + z_residual.detach()
    corrected_depth, corrected_projected_points, intrinsics = internal_points_to_depth(
        corrected_points,
        fov_x=fov_x,
        metric_scale=metric_scale,
        mask=model_mask,
    )
    return corrected_depth, corrected_projected_points, intrinsics, z_residual.detach(), log_scale.detach().exp()


def _forward_points_from_neck_features(model, neck_features, bias: torch.Tensor, bias_level: int, height: int, width: int):
    biased_features = list(neck_features)
    biased_features[bias_level] = biased_features[bias_level] + bias.to(dtype=biased_features[bias_level].dtype)
    points = model.points_head(biased_features)[-1]
    points = torch.nn.functional.interpolate(points, (height, width), mode="bilinear", align_corners=False, antialias=False)
    points = points.permute(0, 2, 3, 1)
    return model._remap_points(points)


def optimize_neck_feature_bias(
    model,
    image: torch.Tensor,
    num_tokens: int,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    fov_x: Optional[float] = None,
    fit_mask: Optional[torch.Tensor] = None,
    fit_weight: Optional[torch.Tensor] = None,
    bias_level: int = -1,
    num_steps: int = 1000,
    lr: float = 0.05,
    data_weight: float = 100.0,
    smooth_weight: float = 0.05,
    anchor_weight: float = 0.1,
    optimize_scale: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimize a trainable bias on a frozen internal neck feature map."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[0] != 1:
        raise ValueError("Guided neck-bias optimization currently expects a single image.")

    device = image.device
    dtype = image.dtype
    _, _, height, width = image.shape
    aspect_ratio = width / height
    base_h, base_w = round((num_tokens / aspect_ratio) ** 0.5), round((num_tokens * aspect_ratio) ** 0.5)

    with torch.no_grad():
        features, cls_token = model.encoder(image, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2 ** level,
                height=base_h * 2 ** level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(1, -1, -1, -1)
            features[level] = uv if features[level] is None else torch.concat([features[level], uv], dim=1)
        neck_features = [feature.detach() for feature in model.neck(features)]
        metric_scale = model.scale_head(cls_token).squeeze(1).exp().float() if hasattr(model, "scale_head") else None
        mask = model.mask_head(neck_features)[-1].sigmoid().squeeze(1) if hasattr(model, "mask_head") else None
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.unsqueeze(1), (height, width), mode="bilinear", align_corners=False).squeeze(1)
            mask_binary = mask > 0.5
        else:
            mask_binary = None
        native_points_signal = _forward_points_from_neck_features(
            model, neck_features, torch.zeros_like(neck_features[bias_level]), bias_level, height, width
        ).float()
        native_depth, native_projected_points, native_intrinsics = internal_points_to_depth(
            native_points_signal.squeeze(0),
            fov_x=fov_x,
            metric_scale=metric_scale,
            mask=mask_binary.squeeze(0) if mask_binary is not None else None,
        )

    target_depth = target_depth.float().clamp_min(eps)
    target_mask = target_mask.bool() & torch.isfinite(target_depth) & (target_depth > eps)
    if fit_mask is None:
        fit_mask = target_mask
    else:
        fit_mask = fit_mask.bool() & target_mask
    if fit_weight is None:
        fit_weight = torch.ones_like(target_depth, dtype=torch.float32)
    else:
        fit_weight = fit_weight.float().clamp_min(0)
    if not target_mask.any():
        raise ValueError("Guided depth mask has no finite positive target depth pixels.")
    if not fit_mask.any():
        raise ValueError("Guided depth fit mask has no finite positive target depth pixels.")

    log_native = native_depth.clamp_min(eps).log().detach()
    log_target = target_depth.log().detach()
    initial_log_scale = (log_native[fit_mask].median() - log_target[fit_mask].median()).detach()

    bias = torch.zeros_like(neck_features[bias_level], dtype=torch.float32, requires_grad=True)
    log_scale = initial_log_scale.clone().requires_grad_(optimize_scale)
    params = [bias, log_scale] if optimize_scale else [bias]
    optimizer = torch.optim.Adam(params, lr=lr)
    outside_mask = ~target_mask

    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        points_signal = _forward_points_from_neck_features(model, neck_features, bias, bias_level, height, width).float()
        depth, _, _ = internal_points_to_depth(
            points_signal.squeeze(0),
            fov_x=fov_x,
            metric_scale=metric_scale,
            mask=mask_binary.squeeze(0) if mask_binary is not None else None,
        )
        log_depth = depth.clamp_min(eps).log()
        scaled_log_target = log_target + log_scale
        residual_error = (log_depth[fit_mask] - scaled_log_target[fit_mask]).square()
        residual_weight = fit_weight[fit_mask]
        loss = data_weight * (residual_error * residual_weight).sum() / residual_weight.sum().clamp_min(eps)
        if smooth_weight > 0:
            loss = loss + smooth_weight * (
                (bias[..., :, 1:] - bias[..., :, :-1]).square().mean()
                + (bias[..., 1:, :] - bias[..., :-1, :]).square().mean()
            )
        if anchor_weight > 0 and outside_mask.any():
            depth_delta = log_depth - log_native
            loss = loss + anchor_weight * depth_delta[outside_mask].square().mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        points_signal = _forward_points_from_neck_features(model, neck_features, bias, bias_level, height, width).float()
        optimized_depth, optimized_points, optimized_intrinsics = internal_points_to_depth(
            points_signal.squeeze(0),
            fov_x=fov_x,
            metric_scale=metric_scale,
            mask=mask_binary.squeeze(0) if mask_binary is not None else None,
        )

    return (
        native_depth,
        native_projected_points,
        native_intrinsics,
        optimized_depth,
        optimized_points,
        optimized_intrinsics,
        bias.detach(),
        log_scale.detach().exp(),
    )


def optimize_encoder_feature_bias(
    model,
    image: torch.Tensor,
    num_tokens: int,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    fov_x: Optional[float] = None,
    fit_mask: Optional[torch.Tensor] = None,
    fit_weight: Optional[torch.Tensor] = None,
    num_steps: int = 1000,
    lr: float = 0.05,
    data_weight: float = 100.0,
    smooth_weight: float = 0.05,
    anchor_weight: float = 0.1,
    optimize_scale: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimize a trainable bias on the frozen DINO encoder feature before MoGe's neck."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[0] != 1:
        raise ValueError("Guided encoder-bias optimization currently expects a single image.")

    device = image.device
    dtype = image.dtype
    _, _, height, width = image.shape
    aspect_ratio = width / height
    base_h, base_w = round((num_tokens / aspect_ratio) ** 0.5), round((num_tokens * aspect_ratio) ** 0.5)

    with torch.no_grad():
        encoder_feature, cls_token = model.encoder(image, base_h, base_w, return_class_token=True)
        encoder_feature = encoder_feature.detach()
        cls_token = cls_token.detach()
        metric_scale = model.scale_head(cls_token).squeeze(1).exp().float() if hasattr(model, "scale_head") else None

    def forward_from_encoder_feature(biased_encoder_feature: torch.Tensor):
        features = [biased_encoder_feature.to(dtype=dtype), None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2 ** level,
                height=base_h * 2 ** level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(1, -1, -1, -1)
            features[level] = uv if features[level] is None else torch.concat([features[level], uv], dim=1)
        neck_features = model.neck(features)
        points = model.points_head(neck_features)[-1]
        points = torch.nn.functional.interpolate(points, (height, width), mode="bilinear", align_corners=False, antialias=False)
        points = points.permute(0, 2, 3, 1)
        points = model._remap_points(points).float()
        mask_binary = None
        if hasattr(model, "mask_head"):
            mask = model.mask_head(neck_features)[-1].sigmoid()
            mask = torch.nn.functional.interpolate(mask, (height, width), mode="bilinear", align_corners=False).squeeze(1)
            mask_binary = mask > 0.5
        depth, projected_points, intrinsics = internal_points_to_depth(
            points.squeeze(0),
            fov_x=fov_x,
            metric_scale=metric_scale,
            mask=mask_binary.squeeze(0) if mask_binary is not None else None,
        )
        return depth, projected_points, intrinsics, mask_binary

    with torch.no_grad():
        native_depth, native_points, native_intrinsics, native_mask = forward_from_encoder_feature(encoder_feature)

    target_depth = target_depth.float().clamp_min(eps)
    target_mask = target_mask.bool() & torch.isfinite(target_depth) & (target_depth > eps)
    if fit_mask is None:
        fit_mask = target_mask
    else:
        fit_mask = fit_mask.bool() & target_mask
    if fit_weight is None:
        fit_weight = torch.ones_like(target_depth, dtype=torch.float32)
    else:
        fit_weight = fit_weight.float().clamp_min(0)
    if not target_mask.any():
        raise ValueError("Guided depth mask has no finite positive target depth pixels.")
    if not fit_mask.any():
        raise ValueError("Guided depth fit mask has no finite positive target depth pixels.")

    log_native = native_depth.clamp_min(eps).log().detach()
    log_target = target_depth.log().detach()
    initial_log_scale = (log_native[fit_mask].median() - log_target[fit_mask].median()).detach()

    bias = torch.zeros_like(encoder_feature, dtype=torch.float32, requires_grad=True)
    log_scale = initial_log_scale.clone().requires_grad_(optimize_scale)
    params = [bias, log_scale] if optimize_scale else [bias]
    optimizer = torch.optim.Adam(params, lr=lr)
    outside_mask = ~target_mask

    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        depth, _, _, _ = forward_from_encoder_feature(encoder_feature + bias.to(dtype=encoder_feature.dtype))
        log_depth = depth.clamp_min(eps).log()
        scaled_log_target = log_target + log_scale
        residual_error = (log_depth[fit_mask] - scaled_log_target[fit_mask]).square()
        residual_weight = fit_weight[fit_mask]
        loss = data_weight * (residual_error * residual_weight).sum() / residual_weight.sum().clamp_min(eps)
        if smooth_weight > 0:
            loss = loss + smooth_weight * (
                (bias[..., :, 1:] - bias[..., :, :-1]).square().mean()
                + (bias[..., 1:, :] - bias[..., :-1, :]).square().mean()
            )
        if anchor_weight > 0 and outside_mask.any():
            depth_delta = log_depth - log_native
            loss = loss + anchor_weight * depth_delta[outside_mask].square().mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        optimized_depth, optimized_points, optimized_intrinsics, _ = forward_from_encoder_feature(encoder_feature + bias.to(dtype=encoder_feature.dtype))

    return (
        native_depth,
        native_points,
        native_intrinsics,
        optimized_depth,
        optimized_points,
        optimized_intrinsics,
        bias.detach(),
        log_scale.detach().exp(),
    )


def _convstack_forward_with_layer_biases(convstack, in_features, layer_biases):
    out_features = []
    for i in range(len(convstack.res_blocks)):
        feature = convstack.input_blocks[i](in_features[i])
        if i == 0:
            x = feature
        elif feature is not None:
            x = x + feature
        x = convstack.res_blocks[i](x)
        if i in layer_biases:
            x = x + layer_biases[i].to(dtype=x.dtype)
        out_features.append(convstack.output_blocks[i](x))
        if i < len(convstack.res_blocks) - 1:
            x = convstack.resamplers[i](x)
    return out_features


def optimize_moge_layer_signals(
    model,
    image: torch.Tensor,
    num_tokens: int,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    fov_x: Optional[float] = None,
    fit_mask: Optional[torch.Tensor] = None,
    fit_weight: Optional[torch.Tensor] = None,
    bias_levels: Tuple[int, ...] = (0, 1, 2),
    num_steps: int = 200,
    lr: float = 0.05,
    data_weight: float = 100.0,
    smooth_weight: float = 0.05,
    anchor_weight: float = 0.1,
    normal_weight: float = 1.0,
    target_normal_weight: float = 1.0,
    bias_l2_weight: float = 1e-4,
    optimize_scale: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], torch.Tensor]:
    """Optimize additive trainable signals inside frozen MoGe neck layers."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[0] != 1:
        raise ValueError("Guided MoGe-layer optimization currently expects a single image.")

    device = image.device
    dtype = image.dtype
    _, _, height, width = image.shape
    aspect_ratio = width / height
    base_h, base_w = round((num_tokens / aspect_ratio) ** 0.5), round((num_tokens * aspect_ratio) ** 0.5)
    bias_levels = tuple(i if i >= 0 else len(model.neck.res_blocks) + i for i in bias_levels)

    with torch.no_grad():
        encoder_feature, cls_token = model.encoder(image, base_h, base_w, return_class_token=True)
        encoder_feature = encoder_feature.detach()
        cls_token = cls_token.detach()
        metric_scale = model.scale_head(cls_token).squeeze(1).exp().float() if hasattr(model, "scale_head") else None

        base_features = [encoder_feature, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2 ** level,
                height=base_h * 2 ** level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(1, -1, -1, -1)
            base_features[level] = uv if base_features[level] is None else torch.concat([base_features[level], uv], dim=1)

        neck_features = model.neck(base_features)
        native_points = model.points_head(neck_features)[-1]
        native_points = torch.nn.functional.interpolate(native_points, (height, width), mode="bilinear", align_corners=False, antialias=False)
        native_points = model._remap_points(native_points.permute(0, 2, 3, 1)).float()
        native_mask_binary = None
        if hasattr(model, "mask_head"):
            native_mask = model.mask_head(neck_features)[-1].sigmoid()
            native_mask = torch.nn.functional.interpolate(native_mask, (height, width), mode="bilinear", align_corners=False).squeeze(1)
            native_mask_binary = native_mask > 0.5
        native_depth, native_projected_points, native_intrinsics = internal_points_to_depth(
            native_points.squeeze(0),
            fov_x=fov_x,
            metric_scale=metric_scale,
            mask=native_mask_binary.squeeze(0) if native_mask_binary is not None else None,
        )
        if hasattr(model, "normal_head"):
            native_normals = model.normal_head(neck_features)[-1]
            native_normals = torch.nn.functional.interpolate(native_normals, (height, width), mode="bilinear", align_corners=False, antialias=False)
            native_normals = torch.nn.functional.normalize(native_normals.permute(0, 2, 3, 1).float().squeeze(0), dim=-1).detach()
        else:
            native_normals = utils3d.pt.point_map_to_normal_map(native_projected_points).detach()

        bias_shapes = {}
        x = None
        for i in range(len(model.neck.res_blocks)):
            feature = model.neck.input_blocks[i](base_features[i])
            if i == 0:
                x = feature
            elif feature is not None:
                x = x + feature
            x = model.neck.res_blocks[i](x)
            if i in bias_levels:
                bias_shapes[i] = x.shape
            if i < len(model.neck.res_blocks) - 1:
                x = model.neck.resamplers[i](x)

    def forward_with_biases(layer_biases):
        neck_features_biased = _convstack_forward_with_layer_biases(model.neck, base_features, layer_biases)
        points = model.points_head(neck_features_biased)[-1]
        points = torch.nn.functional.interpolate(points, (height, width), mode="bilinear", align_corners=False, antialias=False)
        points = model._remap_points(points.permute(0, 2, 3, 1)).float()
        mask_binary = None
        if hasattr(model, "mask_head"):
            mask = model.mask_head(neck_features_biased)[-1].sigmoid()
            mask = torch.nn.functional.interpolate(mask, (height, width), mode="bilinear", align_corners=False).squeeze(1)
            mask_binary = mask > 0.5
        depth, projected_points, intrinsics = internal_points_to_depth(
            points.squeeze(0),
            fov_x=fov_x,
            metric_scale=metric_scale,
            mask=mask_binary.squeeze(0) if mask_binary is not None else None,
        )
        return depth, projected_points, intrinsics

    target_depth = target_depth.float().clamp_min(eps)
    target_mask = target_mask.bool() & torch.isfinite(target_depth) & (target_depth > eps)
    if fit_mask is None:
        fit_mask = target_mask
    else:
        fit_mask = fit_mask.bool() & target_mask
    if fit_weight is None:
        fit_weight = torch.ones_like(target_depth, dtype=torch.float32)
    else:
        fit_weight = fit_weight.float().clamp_min(0)
    if not target_mask.any():
        raise ValueError("Guided depth mask has no finite positive target depth pixels.")
    if not fit_mask.any():
        raise ValueError("Guided depth fit mask has no finite positive target depth pixels.")

    log_native = native_depth.clamp_min(eps).log().detach()
    log_target = target_depth.log().detach()
    initial_log_scale = (log_native[fit_mask].median() - log_target[fit_mask].median()).detach()

    layer_biases = {
        i: torch.zeros(bias_shapes[i], dtype=torch.float32, device=device, requires_grad=True)
        for i in bias_levels
    }
    log_scale = initial_log_scale.clone().requires_grad_(optimize_scale)
    params = list(layer_biases.values()) + ([log_scale] if optimize_scale else [])
    optimizer = torch.optim.Adam(params, lr=lr)
    outside_mask = ~target_mask

    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        depth, _, _ = forward_with_biases(layer_biases)
        log_depth = depth.clamp_min(eps).log()
        scaled_log_target = log_target + log_scale
        residual_error = (log_depth[fit_mask] - scaled_log_target[fit_mask]).square()
        residual_weight = fit_weight[fit_mask]
        loss = data_weight * (residual_error * residual_weight).sum() / residual_weight.sum().clamp_min(eps)
        if smooth_weight > 0:
            smooth_loss = 0
            for bias in layer_biases.values():
                smooth_loss = smooth_loss + (
                    (bias[..., :, 1:] - bias[..., :, :-1]).square().mean()
                    + (bias[..., 1:, :] - bias[..., :-1, :]).square().mean()
                )
            loss = loss + smooth_weight * smooth_loss / max(len(layer_biases), 1)
        if bias_l2_weight > 0:
            loss = loss + bias_l2_weight * sum(bias.square().mean() for bias in layer_biases.values()) / max(len(layer_biases), 1)
        if anchor_weight > 0 and outside_mask.any():
            depth_delta = log_depth - log_native
            loss = loss + anchor_weight * depth_delta[outside_mask].square().mean()
        if normal_weight > 0:
            optimized_points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=native_intrinsics)
            optimized_normals = utils3d.pt.point_map_to_normal_map(optimized_points)
            normal_deviation = 1 - (optimized_normals * native_normals).sum(dim=-1).clamp(-1, 1)
            normal_hf = (
                (optimized_normals[:, 1:] - native_normals[:, 1:] - optimized_normals[:, :-1] + native_normals[:, :-1]).square().sum(dim=-1).mean()
                + (optimized_normals[1:, :] - native_normals[1:, :] - optimized_normals[:-1, :] + native_normals[:-1, :]).square().sum(dim=-1).mean()
            )
            loss = loss + normal_weight * (normal_deviation.mean() + normal_hf)
        if target_normal_weight > 0:
            scaled_target_depth = (log_target + log_scale).exp()
            target_points = utils3d.pt.depth_map_to_point_map(scaled_target_depth, intrinsics=native_intrinsics)
            target_normals, target_normal_valid = utils3d.pt.point_map_to_normal_map(target_points, mask=target_mask)
            optimized_points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=native_intrinsics)
            optimized_target_normals, optimized_target_normal_valid = utils3d.pt.point_map_to_normal_map(optimized_points, mask=target_mask)
            target_normal_valid = target_normal_valid & optimized_target_normal_valid & target_mask
            if target_normal_valid.any():
                target_normal_cos = (optimized_target_normals * target_normals).sum(dim=-1).clamp(-1, 1)
                target_normal_error = 1 - target_normal_cos[target_normal_valid]
                target_normal_weights = fit_weight[target_normal_valid].clamp_min(eps)
                loss = loss + target_normal_weight * (target_normal_error * target_normal_weights).sum() / target_normal_weights.sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        optimized_depth, optimized_points, optimized_intrinsics = forward_with_biases(layer_biases)

    return (
        native_depth,
        native_projected_points,
        native_intrinsics,
        optimized_depth,
        optimized_points,
        optimized_intrinsics,
        native_mask_binary.squeeze(0) if native_mask_binary is not None else torch.ones_like(native_depth, dtype=torch.bool),
        {i: bias.detach() for i, bias in layer_biases.items()},
        log_scale.detach().exp(),
    )


def poisson_match_depth(
    inferred_depth: np.ndarray,
    replacement_depth: np.ndarray,
    replacement_mask: np.ndarray,
    anchor_weight: float = 20.0,
    outside_anchor_weight: float = 0.05,
    eps: float = 1e-6,
) -> np.ndarray:
    """Poisson-match log depth, then exactly restore the replacement region."""
    inferred_depth = np.asarray(inferred_depth, dtype=np.float32)
    replacement_depth = np.asarray(replacement_depth, dtype=np.float32)
    replacement_mask = np.asarray(replacement_mask, dtype=bool)

    height, width = inferred_depth.shape
    valid_inferred = np.isfinite(inferred_depth) & (inferred_depth > eps)
    valid_replacement = np.isfinite(replacement_depth) & (replacement_depth > eps)
    replacement_mask = replacement_mask & valid_replacement
    if not replacement_mask.any():
        raise ValueError("Poisson replacement mask has no finite positive depth pixels.")

    safe_inferred = np.where(valid_inferred, inferred_depth, np.nan)
    if np.isnan(safe_inferred).any():
        fill_value = np.nanmedian(safe_inferred)
        if not np.isfinite(fill_value):
            fill_value = float(np.median(replacement_depth[replacement_mask]))
        safe_inferred = np.nan_to_num(safe_inferred, nan=fill_value, posinf=fill_value, neginf=fill_value)

    guidance_depth = np.where(replacement_mask, replacement_depth, safe_inferred).clip(eps)
    guidance_log_depth = np.log(guidance_depth)

    grad_x = guidance_log_depth[:, :-1] - guidance_log_depth[:, 1:]
    grad_y = guidance_log_depth[:-1, :] - guidance_log_depth[1:, :]
    padded = np.pad(guidance_log_depth, ((1, 1), (1, 1)), mode="edge")
    laplacian = cv2.filter2D(
        padded,
        ddepth=-1,
        kernel=np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    )[1:-1, 1:-1]

    grad_mask = np.ones(grad_x.size + grad_y.size, dtype=bool)
    laplacian_mask = np.ones(height * width, dtype=bool)
    anchor_mask = replacement_mask.reshape(-1)
    outside_anchor_mask = (~replacement_mask & valid_inferred).reshape(-1)

    identity = diags(np.ones(height * width, dtype=np.float32), format="csr")
    equations = [
        grad_equation(width, height, wrap_x=False, wrap_y=False)[grad_mask],
        poisson_equation(width, height, wrap_x=False, wrap_y=False)[laplacian_mask],
        identity[anchor_mask] * anchor_weight,
    ]
    values = [
        np.concatenate([grad_x.reshape(-1), grad_y.reshape(-1)]),
        laplacian.reshape(-1),
        np.log(replacement_depth.clip(eps)).reshape(-1)[anchor_mask] * anchor_weight,
    ]
    if outside_anchor_weight > 0 and outside_anchor_mask.any():
        equations.append(identity[outside_anchor_mask] * outside_anchor_weight)
        values.append(np.log(safe_inferred.clip(eps)).reshape(-1)[outside_anchor_mask] * outside_anchor_weight)

    x, *_ = lsmr(
        vstack(equations),
        np.concatenate(values),
        atol=1e-5,
        btol=1e-5,
        x0=np.log(guidance_depth).reshape(-1),
        show=False,
    )
    matched_depth = np.exp(x.reshape(height, width)).astype(np.float32)
    matched_depth[replacement_mask] = replacement_depth[replacement_mask]
    return matched_depth
