import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import itertools
import json
import warnings

import click


@click.command(help='Inference script')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), help='Input image or folder path. "jpg" and "png" are supported.')
@click.option('--fov_x', 'fov_x_', type=float, default=None, help='If camera parameters are known, set the horizontal field of view in degrees. Otherwise, MoGe will estimate it.')
@click.option('--fov_y', 'fov_y_', type=float, default=None, help='If camera parameters are known, set the vertical field of view in degrees.')
@click.option('--output', '-o', 'output_path', default='./output', type=click.Path(), help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default=None, help='Pretrained model name or path. If not provided, the corresponding default model will be chosen.')
@click.option('--version', 'model_version', type=click.Choice(['v1', 'v2']), default='v2', help='Model version. Defaults to "v2"')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Use fp16 precision for much faster inference.')
@click.option('--resize', 'resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).')
@click.option('--guided_depth', 'guided_depth_path', type=click.Path(exists=True), default=None, help='Depth map used to guide frozen-model residual optimization.')
@click.option('--guided_mask', 'guided_mask_path', type=click.Path(exists=True), default=None, help='Binary mask for the guided depth region.')
@click.option('--guided_steps', type=int, default=200, help='Optimizer steps for guided internal feature-bias optimization.')
@click.option('--guided_lr', type=float, default=0.05, help='Learning rate for guided depth residual optimization.')
@click.option('--guided_depth_weight', type=float, default=100.0, help='Masked-region weight for fitting user-provided depth.')
@click.option('--guided_smooth_weight', type=float, default=0.05, help='Smoothness weight for guided depth residuals.')
@click.option('--guided_anchor_weight', type=float, default=0.1, help='Outside-mask residual anchor weight for guided depth optimization.')
@click.option('--guided_normal_weight', type=float, default=1.0, help='Whole-image weight for preserving MoGe native normals and their high-frequency structure.')
@click.option('--guided_target_normal_weight', type=float, default=1.0, help='Masked-region weight for fitting normals derived from the user-provided depth.')
@click.option('--guided_fit_erode_px', type=int, default=12, help='Pixels to erode from guided mask before applying full target loss.')
@click.option('--guided_outer_weight', type=float, default=0.0, help='Target-loss weight for the excluded outer mask band.')
@click.option('--guided_layer_levels', type=str, default='0,1,2', help='Comma-separated MoGe neck layer indices receiving trainable additive signals.')
@click.option('--poisson_anchor_weight', type=float, default=20.0, help='Masked-region anchor weight for final Poisson depth matching.')
@click.option('--focal_length_mm', type=float, default=None, help='Known focal length in millimeters for simple pinhole FoV conversion.')
@click.option('--sensor_width_mm', type=float, default=36.0, help='Camera sensor width in millimeters for simple pinhole FoV conversion.')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level for inference. \
Higher value means more tokens and the finer details will be captured, but inference can be slower. \
Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size. \
`resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.')
@click.option('--num_tokens', type=int, default=None, help='number of tokens used for inference. A integer in the (suggested) range of `[1200, 2500]`. \
`resolution_level` will be ignored if `num_tokens` is provided. Default: None')
@click.option('--threshold', type=float, default=0.04, help='Threshold for removing edges. Defaults to 0.01. Smaller value removes more edges. "inf" means no thresholding.')
@click.option('--maps', 'save_maps_', is_flag=True, help='Whether to save the output maps (image, point map, depth map, normal map, mask) and fov.')
@click.option('--glb', 'save_glb_', is_flag=True, help='Whether to save the output as a.glb file. The color will be saved as a texture.')
@click.option('--ply', 'save_ply_', is_flag=True, help='Whether to save the output as a.ply file. The color will be saved as vertex colors.')
@click.option('--show', 'show', is_flag=True, help='Whether show the output in a window. Note that this requires pyglet<2 installed as required by trimesh.')
def main(
    input_path: str,
    fov_x_: float,
    fov_y_: float,
    output_path: str,
    pretrained_model_name_or_path: str,
    model_version: str,
    device_name: str,
    use_fp16: bool,
    resize_to: int,
    guided_depth_path: str,
    guided_mask_path: str,
    guided_steps: int,
    guided_lr: float,
    guided_depth_weight: float,
    guided_smooth_weight: float,
    guided_anchor_weight: float,
    guided_normal_weight: float,
    guided_target_normal_weight: float,
    guided_fit_erode_px: int,
    guided_outer_weight: float,
    guided_layer_levels: str,
    poisson_anchor_weight: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    resolution_level: int,
    num_tokens: int,
    threshold: float,
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    show: bool,
):  
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    import click
    import matplotlib

    from moge.model import import_model_class_by_version
    from moge.utils.io import save_glb, save_ply
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
    from moge.utils.guided_depth import fov_x_from_pinhole, optimize_moge_layer_signals, poisson_match_depth
    import utils3d

    def image_to_float_tensor(image_: np.ndarray) -> np.ndarray:
        if np.issubdtype(image_.dtype, np.integer):
            return image_.astype(np.float32) / np.iinfo(image_.dtype).max
        return image_.astype(np.float32).clip(0, 1)

    def read_depth_map(path: Union[str, os.PathLike]) -> np.ndarray:
        depth_ = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth_ is None:
            try:
                import OpenEXR
                exr_file = OpenEXR.File(str(path))
                channels = exr_file.channels()
                depth_ = channels[next(iter(channels))].pixels
            except Exception as exc:
                raise ValueError(f'Could not read guided depth map {path}') from exc
        if depth_.ndim == 3:
            depth_ = depth_[..., 0]
        return depth_.astype(np.float32)

    def read_feather_mask(path: Union[str, os.PathLike]) -> np.ndarray:
        mask_ = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask_ is None:
            raise ValueError(f'Could not read guided mask {path}')
        if mask_.ndim == 3:
            mask_ = mask_[..., 0]
        if np.issubdtype(mask_.dtype, np.integer):
            return (mask_.astype(np.float32) / np.iinfo(mask_.dtype).max).clip(0, 1)
        return mask_.astype(np.float32).clip(0, 1)

    def fov_x_from_fov_y(fov_y: float, width_: int, height_: int) -> float:
        return float(np.rad2deg(2 * np.arctan((width_ / height_) * np.tan(np.deg2rad(fov_y) / 2))))

    def colorize_depth_like_reference(depth_: np.ndarray, reference_depth_: np.ndarray, mask_: np.ndarray = None, cmap: str = 'Spectral') -> np.ndarray:
        reference_depth_ = np.where(reference_depth_ > 0, reference_depth_, np.nan)
        ref_disp = 1 / reference_depth_
        min_disp, max_disp = np.nanquantile(ref_disp, 0.001), np.nanquantile(ref_disp, 0.99)

        if mask_ is None:
            depth_ = np.where(depth_ > 0, depth_, np.nan)
        else:
            depth_ = np.where((depth_ > 0) & mask_, depth_, np.nan)
        disp = (1 / depth_ - min_disp) / (max_disp - min_disp)
        colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
        return np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))

    device = torch.device(device_name)

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    if pretrained_model_name_or_path is None:
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[model_version]
    model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()
    if use_fp16 and guided_depth_path is None:
        model.half()

    if guided_depth_path is not None:
        if guided_mask_path is None:
            raise ValueError('--guided_mask is required when --guided_depth is provided.')
        if focal_length_mm is not None:
            fov_x_ = fov_x_from_pinhole(focal_length_mm, sensor_width_mm)
    
    if not any([save_maps_, save_glb_, save_ply_]):
        warnings.warn('No output format specified. Defaults to saving all. Please use "--maps", "--glb", or "--ply" to specify the output.')
        save_maps_ = save_glb_ = save_ply_ = True

    for image_path in (pbar := tqdm(image_paths, desc='Inference', disable=len(image_paths) <= 1)):
        if not image_path.exists():
            raise FileNotFoundError(f'File {image_path} does not exist.')
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if resize_to is not None:
            height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        fov_x_for_image = fov_x_
        if fov_x_for_image is None and fov_y_ is not None:
            fov_x_for_image = fov_x_from_fov_y(fov_y_, width, height)
        image_tensor = torch.tensor(image_to_float_tensor(image), dtype=torch.float32, device=device).permute(2, 0, 1)

        # Inference
        if guided_depth_path is None:
            output = model.infer(
                image_tensor,
                fov_x=fov_x_for_image,
                resolution_level=resolution_level,
                num_tokens=num_tokens,
                use_fp16=use_fp16,
            )
            points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
            normal = output['normal'].cpu().numpy() if 'normal' in output else None
            native_depth = depth.copy()
        else:
            if num_tokens is None:
                min_tokens, max_tokens = model.num_tokens_range
                num_tokens_ = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
            else:
                num_tokens_ = num_tokens
            normal = None

        guided_scale = None
        optimized_depth = None
        scaled_guided_depth = None
        guided_mask = None
        guided_fit_mask = None
        if guided_depth_path is not None:
            guided_depth = read_depth_map(guided_depth_path)
            guided_alpha = read_feather_mask(guided_mask_path)
            if guided_depth.shape != (height, width):
                guided_depth = cv2.resize(guided_depth, (width, height), cv2.INTER_NEAREST)
            if guided_alpha.shape != (height, width):
                guided_alpha = cv2.resize(guided_alpha, (width, height), cv2.INTER_LINEAR).clip(0, 1)
            guided_mask = guided_alpha > 0

            target_depth_t = torch.tensor(guided_depth, dtype=torch.float32, device=device)
            target_mask_t = torch.tensor(guided_mask, dtype=torch.bool, device=device)
            if guided_fit_erode_px > 0:
                kernel_size = guided_fit_erode_px * 2 + 1
                guided_fit_alpha = -cv2.dilate(-guided_alpha, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
                guided_fit_alpha = guided_fit_alpha.clip(0, 1)
                if not (guided_fit_alpha > 0).any():
                    guided_fit_alpha = guided_alpha.copy()
            else:
                guided_fit_alpha = guided_alpha.copy()
            guided_fit_mask = guided_fit_alpha > 0
            fit_weight = np.maximum(guided_fit_alpha, guided_alpha * float(guided_outer_weight)).astype(np.float32)
            fit_mask = guided_mask & (fit_weight > 0)
            layer_levels = tuple(int(item.strip()) for item in guided_layer_levels.split(',') if item.strip())
            native_depth_t, native_points_t, native_intrinsics_t, optimized_depth_t, optimized_points_t, optimized_intrinsics_t, output_mask_t, _, guided_scale_t = optimize_moge_layer_signals(
                model,
                image_tensor.to(dtype=model.dtype, device=device),
                num_tokens_,
                target_depth_t,
                target_mask_t,
                fov_x=fov_x_for_image,
                fit_mask=torch.tensor(fit_mask, dtype=torch.bool, device=device),
                fit_weight=torch.tensor(fit_weight, dtype=torch.float32, device=device),
                bias_levels=layer_levels,
                num_steps=guided_steps,
                lr=guided_lr,
                data_weight=guided_depth_weight,
                smooth_weight=guided_smooth_weight,
                anchor_weight=guided_anchor_weight,
                normal_weight=guided_normal_weight,
                target_normal_weight=guided_target_normal_weight,
                optimize_scale=True,
            )
            native_depth = native_depth_t.cpu().numpy()
            mask = output_mask_t.cpu().numpy()
            optimized_depth = optimized_depth_t.cpu().numpy()
            optimized_points = optimized_points_t.cpu().numpy()
            intrinsics = optimized_intrinsics_t.cpu().numpy()
            scaled_guided_depth = guided_depth * float(guided_scale_t.cpu())
            depth = poisson_match_depth(
                optimized_depth,
                scaled_guided_depth,
                guided_alpha,
                anchor_weight=poisson_anchor_weight,
            )
            guided_scale = float(guided_scale_t.cpu())
            depth_t = torch.tensor(depth, dtype=optimized_depth_t.dtype, device=device)
            mask_t = output_mask_t.to(device=device, dtype=torch.bool)
            depth_t = torch.where(mask_t, depth_t, torch.inf)
            points_t = utils3d.pt.depth_map_to_point_map(depth_t, intrinsics=optimized_intrinsics_t)
            points_t = torch.where(mask_t[..., None], points_t, torch.inf)
            depth = depth_t.cpu().numpy()
            points = points_t.cpu().numpy()

        save_path = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
        save_path.mkdir(exist_ok=True, parents=True)

        # Save images / maps
        if save_maps_:
            depth_exr_scale = guided_scale if guided_scale is not None else 1.0
            depth_exr = np.where(mask, depth / depth_exr_scale, np.inf).astype(np.float32)
            points_exr = np.where(mask[..., None], points, np.inf).astype(np.float32)
            cv2.imwrite(str(save_path / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth.exr'), depth_exr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            if optimized_depth is not None:
                cv2.imwrite(str(save_path / 'depth_native.exr'), np.where(mask, native_depth / depth_exr_scale, np.inf).astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(save_path / 'depth_optimized.exr'), np.where(mask, optimized_depth / depth_exr_scale, np.inf).astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(save_path / 'depth_guided_scaled.exr'), np.where(mask, scaled_guided_depth / depth_exr_scale, np.inf).astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(save_path / 'depth_native_vis.png'), cv2.cvtColor(colorize_depth_like_reference(native_depth, depth), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / 'depth_optimized_vis.png'), cv2.cvtColor(colorize_depth_like_reference(optimized_depth, depth), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / 'depth_optimized_replaced_vis.png'), cv2.cvtColor(colorize_depth_like_reference(depth, depth), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / 'guided_mask.png'), (guided_alpha * 255).round().astype(np.uint8))
                cv2.imwrite(str(save_path / 'guided_fit_mask.png'), (guided_fit_alpha * 255).round().astype(np.uint8))
            cv2.imwrite(str(save_path / 'mask.png'), (mask * 255).astype(np.uint8))
            cv2.imwrite(str(save_path / 'points.exr'), cv2.cvtColor(points_exr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            if normal is not None:
                cv2.imwrite(str(save_path / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))
            fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
            with open(save_path / 'fov.json', 'w') as f:
                fov_data = {
                    'fov_x': round(float(np.rad2deg(fov_x)), 2),
                    'fov_y': round(float(np.rad2deg(fov_y)), 2),
                }
                if guided_scale is not None:
                    fov_data['guided_depth_scale'] = guided_scale
                json.dump(fov_data, f)

        # Export mesh & visulization
        if save_glb_ or save_ply_ or show:
            mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=threshold)
            if normal is None:
                faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.np.uv_map(height, width),
                    mask=mask_cleaned,
                    tri=True
                )
                vertex_normals = None
            else:
                faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.np.build_mesh_from_map(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.np.uv_map(height, width),
                    normal,
                    mask=mask_cleaned,
                    tri=True
                )
            # When exporting the model, follow the OpenGL coordinate conventions:
            # - world coordinate system: x right, y up, z backward.
            # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
            vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
            if normal is not None:
                vertex_normals = vertex_normals * [1, -1, -1]

        if save_glb_:
            save_glb(save_path / 'mesh.glb', vertices, faces, vertex_uvs, image, vertex_normals)

        if save_ply_:
            save_ply(save_path / 'pointcloud.ply', vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)

        if show:
            import trimesh
            trimesh.Trimesh(
                vertices=vertices,
                vertex_colors=vertex_colors,
                vertex_normals=vertex_normals,
                faces=faces, 
                process=False
            ).show()  


if __name__ == '__main__':
    main()
