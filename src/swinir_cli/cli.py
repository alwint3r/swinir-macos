from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np
import requests
from PIL import Image

from swinir_cli.swinir.network_swinir_mlx import SwinIR


MODEL_RELEASE_BASE = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0"
MODEL_MEDIUM_X4 = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
MODEL_LARGE_X4 = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    large: bool


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        upscale(
            input_path=args.input,
            output_dir=args.output,
            model_dir=args.model_dir,
            large_model=args.large_model,
            device_name=args.device,
            tile=args.tile,
            tile_overlap=args.tile_overlap,
            overwrite=args.overwrite,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swinir-upscale",
        description="Upscale an image or folder of images with SwinIR real-world x4 super-resolution.",
    )
    parser.add_argument("input", type=Path, help="Image file or directory to upscale.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path.cwd() / "out",
        help="Output folder. Defaults to ./out in the current working directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.cwd() / "models",
        help="Folder used to cache downloaded and converted SwinIR weights. Defaults to ./models.",
    )
    parser.add_argument(
        "--large-model",
        action="store_true",
        help="Use the larger SwinIR-L x4 GAN model. It is slower and uses more memory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="Inference device. MLX auto defaults to the GPU when available.",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=400,
        help="Tile size for lower memory use. Use 0 to process the whole image. Must be a multiple of 8.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=32,
        help="Overlap between tiles. Must be smaller than --tile when tiling is enabled.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser


def upscale(
    input_path: Path,
    output_dir: Path,
    model_dir: Path,
    large_model: bool,
    device_name: str,
    tile: int,
    tile_overlap: int,
    overwrite: bool,
) -> None:
    image_paths = list(resolve_images(input_path))
    if not image_paths:
        raise ValueError(f"no supported images found at {input_path}")

    if tile < 0:
        raise ValueError("--tile must be 0 or a positive multiple of 8")
    if tile and tile % 8 != 0:
        raise ValueError("--tile must be a multiple of 8")
    if tile and not 0 <= tile_overlap < tile:
        raise ValueError("--tile-overlap must be at least 0 and smaller than --tile")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(device_name)
    spec = ModelSpec(MODEL_LARGE_X4, large=True) if large_model else ModelSpec(MODEL_MEDIUM_X4, large=False)
    model_path = ensure_mlx_model(model_dir, spec)

    print(f"Using device: {device}")
    print(f"Using model: {model_path}")
    model = load_model(spec, model_path)

    for path in image_paths:
        target = output_dir / f"{path.stem}_swinir_x4.png"
        if target.exists() and not overwrite:
            print(f"Skipping existing output: {target}")
            continue

        print(f"Upscaling {path} -> {target}")
        result = upscale_image(path, model, device, tile=None if tile == 0 else tile, tile_overlap=tile_overlap)
        result.save(target)


def resolve_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield input_path
        return

    if input_path.is_dir():
        for path in sorted(input_path.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path
        return

    raise FileNotFoundError(input_path)


def pick_device(name: str) -> mx.Device:
    if name == "auto":
        device = mx.gpu if mx.metal.is_available() else mx.cpu
    elif name == "gpu":
        if not mx.metal.is_available():
            raise RuntimeError("MLX GPU was requested, but Metal is not available.")
        device = mx.gpu
    else:
        device = mx.cpu

    mx.set_default_device(device)
    return device


def ensure_mlx_model(model_dir: Path, spec: ModelSpec) -> Path:
    mlx_path = model_dir / f"{Path(spec.filename).stem}.npz"
    if mlx_path.exists() and mlx_path.stat().st_size > 0:
        return mlx_path

    torch_path = ensure_torch_model(model_dir, spec.filename)
    print(f"Converting {torch_path} -> {mlx_path}")
    convert_torch_checkpoint(torch_path, mlx_path, build_model(spec))
    return mlx_path


def ensure_torch_model(model_dir: Path, filename: str) -> Path:
    path = model_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return path

    url = f"{MODEL_RELEASE_BASE}/{filename}"
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(path)

    return path


def build_model(spec: ModelSpec) -> SwinIR:
    if spec.large:
        return SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=240,
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="3conv",
        )

    return SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="1conv",
    )


def load_model(spec: ModelSpec, model_path: Path) -> SwinIR:
    model = build_model(spec)
    model.load_weights(str(model_path), strict=True)
    model.eval()
    return model


def convert_torch_checkpoint(torch_path: Path, mlx_path: Path, model: SwinIR) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "converting official .pth weights requires PyTorch; run `uv sync --extra convert` first"
        ) from exc

    checkpoint = torch.load(torch_path, map_location="cpu", weights_only=False)
    params = checkpoint.get("params_ema", checkpoint)
    target_shapes = {name: tuple(value.shape) for name, value in flatten_parameters(model.parameters()).items()}

    converted: dict[str, mx.array] = {}
    missing: list[str] = []
    for name, target_shape in target_shapes.items():
        if name not in params:
            missing.append(name)
            continue

        value = params[name].detach().cpu().numpy()
        if value.ndim == 4 and value.transpose(0, 2, 3, 1).shape == target_shape:
            converted[name] = mx.array(value.transpose(0, 2, 3, 1))
        elif value.shape == target_shape:
            converted[name] = mx.array(value)
        else:
            raise ValueError(f"shape mismatch for {name}: torch {value.shape}, mlx {target_shape}")

    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} parameters, including {missing[:5]}")

    tmp_path = mlx_path.with_name(f"{mlx_path.stem}.tmp{mlx_path.suffix}")
    mx.savez(str(tmp_path), **converted)
    tmp_path.replace(mlx_path)


def flatten_parameters(tree: object, prefix: str = "") -> dict[str, mx.array]:
    if isinstance(tree, mx.array):
        return {prefix: tree}
    if isinstance(tree, dict):
        values: dict[str, mx.array] = {}
        for key, value in tree.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            values.update(flatten_parameters(value, child_prefix))
        return values
    if isinstance(tree, list):
        values = {}
        for index, value in enumerate(tree):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            values.update(flatten_parameters(value, child_prefix))
        return values
    return {}


def upscale_image(path: Path, model: SwinIR, device: mx.Device, tile: int | None, tile_overlap: int) -> Image.Image:
    del device
    image = Image.open(path).convert("RGB")
    image_np = np.asarray(image).astype(np.float32) / 255.0
    h_old, w_old, _ = image_np.shape
    h_pad = (h_old // 8 + 1) * 8 - h_old
    w_pad = (w_old // 8 + 1) * 8 - w_old
    image_np = np.concatenate([image_np, np.flip(image_np, axis=0)], axis=0)[: h_old + h_pad, :, :]
    image_np = np.concatenate([image_np, np.flip(image_np, axis=1)], axis=1)[:, : w_old + w_pad, :]

    tensor = mx.array(image_np[None, ...])
    output = run_model(tensor, model, scale=4, tile=tile, tile_overlap=tile_overlap)
    output = output[:, : h_old * 4, : w_old * 4, :]

    mx.eval(output)
    output_np = np.clip(np.array(output[0]), 0, 1)
    output_np = (output_np * 255.0).round().astype(np.uint8)
    return Image.fromarray(output_np, mode="RGB")


def run_model(
    image: mx.array,
    model: SwinIR,
    scale: int,
    tile: int | None,
    tile_overlap: int,
) -> mx.array:
    if tile is None:
        return model(image)

    batch, height, width, channels = image.shape
    tile = min(tile, height, width)
    if tile % 8 != 0:
        tile = max(8, tile - tile % 8)
    effective_overlap = min(tile_overlap, max(0, tile - 8))

    stride = tile - effective_overlap
    h_indices = list(range(0, height - tile, stride)) + [height - tile]
    w_indices = list(range(0, width - tile, stride)) + [width - tile]
    output = np.zeros((batch, height * scale, width * scale, channels), dtype=np.float32)
    weights = np.zeros_like(output)

    for h_index in h_indices:
        for w_index in w_indices:
            in_patch = image[:, h_index : h_index + tile, w_index : w_index + tile, :]
            out_patch = model(in_patch)
            mx.eval(out_patch)
            out_patch_np = np.array(out_patch)
            output[
                :,
                h_index * scale : (h_index + tile) * scale,
                w_index * scale : (w_index + tile) * scale,
                :,
            ] += out_patch_np
            weights[
                :,
                h_index * scale : (h_index + tile) * scale,
                w_index * scale : (w_index + tile) * scale,
                :,
            ] += 1.0

    return mx.array(output / weights)


if __name__ == "__main__":
    raise SystemExit(main())
