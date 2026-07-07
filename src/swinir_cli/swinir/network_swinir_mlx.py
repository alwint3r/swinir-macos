from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def run_layers(x: mx.array, layers: list[nn.Module]) -> mx.array:
    for layer in layers:
        x = layer(x)
    return x


def window_partition(x: mx.array, window_size: int) -> mx.array:
    batch, height, width, channels = x.shape
    x = x.reshape(batch, height // window_size, window_size, width // window_size, window_size, channels)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, window_size, window_size, channels)


def window_reverse(windows: mx.array, window_size: int, height: int, width: int) -> mx.array:
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.reshape(batch, height // window_size, width // window_size, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(batch, height, width, -1)


def nearest_upsample(x: mx.array, scale: int) -> mx.array:
    x = mx.repeat(x, scale, axis=1)
    return mx.repeat(x, scale, axis=2)


def pixel_shuffle(x: mx.array, scale: int) -> mx.array:
    batch, height, width, channels = x.shape
    out_channels = channels // (scale * scale)
    x = x.reshape(batch, height, width, scale, scale, out_channels)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(batch, height * scale, width * scale, out_channels)


def calculate_attention_mask(x_size: tuple[int, int], window_size: int, shift_size: int) -> np.ndarray:
    height, width = x_size
    mask = np.zeros((1, height, width, 1), dtype=np.float32)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))

    count = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            mask[:, h_slice, w_slice, :] = count
            count += 1

    mask_windows = np.array(window_partition(mx.array(mask), window_size))
    mask_windows = mask_windows.reshape(-1, window_size * window_size)
    attention_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    attention_mask = np.where(attention_mask != 0, -100.0, 0.0).astype(np.float32)
    return attention_mask


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        table_shape = ((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        self.relative_position_bias_table = mx.zeros(table_shape)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = np.transpose(relative_coords, (1, 2, 0))
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1).astype(np.int32)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.swapaxes(-2, -1)

        bias = self.relative_position_bias_table[mx.array(self.relative_position_index.reshape(-1))]
        bias = bias.reshape(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        bias = bias.transpose(2, 0, 1)
        attn = attn + mx.expand_dims(bias, 0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.reshape(batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask[:, None, :, :][None, ...]
            attn = attn.reshape(-1, self.num_heads, tokens, tokens)

        attn = mx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(batch_windows, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must be in [0, window_size)")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.attn_mask = calculate_attention_mask(self.input_resolution, self.window_size, self.shift_size) if self.shift_size > 0 else None

    def __call__(self, x: mx.array, x_size: tuple[int, int]) -> mx.array:
        height, width = x_size
        batch, _, channels = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(batch, height, width, channels)

        if self.shift_size > 0:
            shifted_x = mx.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, channels)

        if self.input_resolution == x_size:
            mask = mx.array(self.attn_mask) if self.attn_mask is not None else None
        else:
            mask = mx.array(calculate_attention_mask(x_size, self.window_size, self.shift_size)) if self.shift_size > 0 else None
        attn_windows = self.attn(x_windows, mask=mask)

        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attn_windows, self.window_size, height, width)

        if self.shift_size > 0:
            x = mx.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(batch, height * width, channels)

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = [
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ]

    def __call__(self, x: mx.array, x_size: tuple[int, int]) -> mx.array:
        for block in self.blocks:
            x = block(x, x_size)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int | tuple[int, int] = 224, patch_size: int | tuple[int, int] = 4, embed_dim: int = 96, norm_layer: bool = False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else None

    def __call__(self, x: mx.array) -> mx.array:
        batch, height, width, channels = x.shape
        x = x.reshape(batch, height * width, channels)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size: int | tuple[int, int] = 224, patch_size: int | tuple[int, int] = 4, embed_dim: int = 96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.embed_dim = embed_dim

    def __call__(self, x: mx.array, x_size: tuple[int, int]) -> mx.array:
        batch, _, channels = x.shape
        return x.reshape(batch, x_size[0], x_size[1], channels)


class RSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        img_size: int = 224,
        patch_size: int = 4,
        resi_connection: str = "1conv",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        if resi_connection == "1conv":
            self.conv: nn.Module | list[nn.Module] = nn.Conv2d(dim, dim, 3, padding=1)
        elif resi_connection == "3conv":
            self.conv = [
                nn.Conv2d(dim, dim // 4, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(dim // 4, dim // 4, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(dim // 4, dim, 3, padding=1),
            ]
        else:
            raise ValueError(f"unsupported resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=dim, norm_layer=False)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, embed_dim=dim)

    def __call__(self, x: mx.array, x_size: tuple[int, int]) -> mx.array:
        residual = self.residual_group(x, x_size)
        residual = self.patch_unembed(residual, x_size)
        if isinstance(self.conv, list):
            residual = run_layers(residual, self.conv)
        else:
            residual = self.conv(residual)
        return self.patch_embed(residual) + x


class Upsample(nn.Module):
    def __init__(self, scale: int, num_feat: int):
        super().__init__()
        layers: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.append(nn.Conv2d(num_feat, 4 * num_feat, 3, padding=1))
                layers.append(PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_feat, 9 * num_feat, 3, padding=1))
            layers.append(PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        return run_layers(x, self.layers)


class PixelShuffle(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        return pixel_shuffle(x, self.scale)


class UpsampleOneStep(nn.Module):
    def __init__(self, scale: int, num_feat: int, num_out_ch: int):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return pixel_shuffle(self.conv(x), self.scale)


class SwinIR(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: list[int] | tuple[int, ...] = (6, 6, 6, 6),
        num_heads: list[int] | tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        ape: bool = False,
        patch_norm: bool = True,
        upscale: int = 2,
        img_range: float = 1.0,
        upsampler: str = "",
        resi_connection: str = "1conv",
    ):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.mean = np.array((0.4488, 0.4371, 0.4040), dtype=np.float32).reshape(1, 1, 1, 3) if in_chans == 3 else np.zeros((1, 1, 1, 1), dtype=np.float32)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, padding=1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, norm_layer=patch_norm)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        if self.ape:
            self.absolute_pos_embed = mx.zeros((1, self.patch_embed.num_patches, embed_dim))

        self.pos_drop = nn.Dropout(drop_rate)
        dpr = np.linspace(0, drop_path_rate, sum(depths)).astype(float).tolist()

        self.layers = []
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body: nn.Module | list[nn.Module] = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        elif resi_connection == "3conv":
            self.conv_after_body = [
                nn.Conv2d(embed_dim, embed_dim // 4, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, padding=1),
            ]
        else:
            raise ValueError(f"unsupported resi_connection: {resi_connection}")

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = [nn.Conv2d(embed_dim, num_feat, 3, padding=1), nn.LeakyReLU()]
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, padding=1)
        elif self.upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        elif self.upsampler == "nearest+conv":
            self.conv_before_upsample = [nn.Conv2d(embed_dim, num_feat, 3, padding=1), nn.LeakyReLU()]
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, padding=1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, padding=1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, padding=1)

    def check_image_size(self, x: mx.array) -> mx.array:
        _, height, width, _ = x.shape
        mod_pad_h = (self.window_size - height % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - width % self.window_size) % self.window_size
        if mod_pad_h == 0 and mod_pad_w == 0:
            return x
        raise ValueError("MLX SwinIR expects inputs padded to a multiple of the window size")

    def forward_features(self, x: mx.array) -> mx.array:
        x_size = (x.shape[1], x.shape[2])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def __call__(self, x: mx.array) -> mx.array:
        height, width = x.shape[1:3]
        x = self.check_image_size(x)

        mean = mx.array(self.mean, dtype=x.dtype)
        x = (x - mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            body = self.forward_features(x)
            body = self.conv_after_body(body) if isinstance(self.conv_after_body, nn.Module) else run_layers(body, self.conv_after_body)
            x = body + x
            x = run_layers(x, self.conv_before_upsample)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            body = self.forward_features(x)
            body = self.conv_after_body(body) if isinstance(self.conv_after_body, nn.Module) else run_layers(body, self.conv_after_body)
            x = self.upsample(body + x)
        elif self.upsampler == "nearest+conv":
            x = self.conv_first(x)
            body = self.forward_features(x)
            body = self.conv_after_body(body) if isinstance(self.conv_after_body, nn.Module) else run_layers(body, self.conv_after_body)
            x = body + x
            x = run_layers(x, self.conv_before_upsample)
            x = self.lrelu(self.conv_up1(nearest_upsample(x, 2)))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(nearest_upsample(x, 2)))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            body = self.forward_features(x_first)
            body = self.conv_after_body(body) if isinstance(self.conv_after_body, nn.Module) else run_layers(body, self.conv_after_body)
            x = x + self.conv_last(body + x_first)

        x = x / self.img_range + mean
        return x[:, : height * self.upscale, : width * self.upscale, :]
