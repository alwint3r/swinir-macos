# SwinIR Upscale CLI

This is a small MLX wrapper around [jingyunliang/SwinIR](https://github.com/jingyunliang/swinir) for real-world x2/x4 image upscaling on macOS. It runs inference with [MLX](https://ml-explore.github.io/mlx/build/html/index.html) and prefers the Apple GPU when available.

## Usage

Install dependencies and run the CLI with `uv`:

```sh
uv sync --extra convert
uv run swinir-upscale path/to/image.png
```

The `convert` extra installs PyTorch and timm so the first run can convert the official `.pth` checkpoint into MLX `.npz` weights. After the `.npz` file exists in `./models`, plain `uv sync` is enough for MLX inference.

By default, output images are written to `./out` in the current working directory:

```sh
uv run swinir-upscale path/to/image.png
uv run swinir-upscale path/to/image.png --output upscaled
uv run swinir-upscale path/to/folder --output upscaled
```

The first run downloads the official SwinIR real-world x4 model into `./models` and writes the converted MLX weights next to it.

Useful options:

```sh
uv run swinir-upscale input.jpg --device gpu
uv run swinir-upscale input.jpg --device cpu
uv run swinir-upscale input.jpg --scale 2
uv run swinir-upscale input.jpg --scale 4
uv run swinir-upscale input.jpg --tile 256
uv run swinir-upscale input.jpg --tile 0
uv run swinir-upscale input.jpg --large-model
```

`--scale 4` is the default. `--scale 2` uses the official medium real-world x2 GAN checkpoint. `--large-model` is only available with `--scale 4` because upstream does not provide a real-world SwinIR-L x2 GAN checkpoint.

`--tile` keeps memory usage lower on larger images. It defaults to `400`, which is a multiple of SwinIR's window size. `--tile 0` processes the full image at once.

## Notes

- The CLI is intentionally scoped to real-world x2/x4 super-resolution, the upstream SwinIR mode meant for existing low-quality images without a paired ground-truth image.
- The MLX model mirrors the vendored Torch SwinIR architecture. The Torch definition is retained for checkpoint/reference parity. See [LICENSE.swinir](LICENSE.swinir).
