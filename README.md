# SwinIR Upscale CLI

This is a small wrapper around [jingyunliang/SwinIR](https://github.com/jingyunliang/swinir) for real-world x4 image upscaling on macOS. It prefers Apple GPU acceleration through PyTorch MPS when available.

## Usage

Install dependencies and run the CLI with `uv`:

```sh
uv sync
uv run swinir-upscale path/to/image.png
```

By default, output images are written to `./out` in the current working directory:

```sh
uv run swinir-upscale path/to/image.png
uv run swinir-upscale path/to/image.png --output upscaled
uv run swinir-upscale path/to/folder --output upscaled
```

The first run downloads the official SwinIR real-world x4 model into `./models`.

Useful options:

```sh
uv run swinir-upscale input.jpg --device mps
uv run swinir-upscale input.jpg --tile 256
uv run swinir-upscale input.jpg --tile 0
uv run swinir-upscale input.jpg --large-model
```

`--tile` keeps memory usage lower on larger images. It defaults to `400`, which is a multiple of SwinIR's window size. `--tile 0` processes the full image at once.

## Notes

- The CLI is intentionally scoped to real-world x4 super-resolution, the upstream SwinIR mode meant for existing low-quality images without a paired ground-truth image.
- The vendored model definition is from the upstream SwinIR repository. See [LICENSE.swinir](LICENSE.swinir).
