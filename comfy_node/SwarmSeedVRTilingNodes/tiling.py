"""Tile geometry, image conversion, and stitching."""

from __future__ import annotations

import logging
import time
from typing import TypedDict

import numpy as np
import torch
from comfy.utils import ProgressBar
from PIL import Image


class TileInfo(TypedDict):
    tile: Image.Image
    position: tuple[int, int]
    actual_size: tuple[int, int]
    padding: tuple[int, int, int, int]
    memory_padding: tuple[int, int, int, int]
    original_tile_size: tuple[int, int]


class PreparedTile(TypedDict):
    cropped_tile: Image.Image
    paste_x: int
    paste_y: int


class TileLayout(TypedDict):
    tiles: list[TileInfo]
    original: Image.Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert one HWC or 1HWC RGB tensor to an 8-bit image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    image_np = tensor.mul(255).clamp(0, 255).byte().numpy()
    return Image.fromarray(image_np, "RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


_GPU_DIM_ALIGNMENT = 16


def _compute_axis_positions(length: int, tile_size: int) -> list[int]:
    """Compute axis-aligned tile starts while avoiding tiny remainder tiles at the boundary."""
    if length <= tile_size:
        return [0]

    positions = list(range(0, length, tile_size))
    # Shift the final tile so it ends on the image boundary.
    positions[-1] = length - tile_size
    return positions


def generate_tiles(
    image: Image.Image,
    tile_size: int,
    tile_overlap: int,
) -> list[TileInfo]:
    """Generate overlapping square tiles in Chess order."""
    width, height = image.size
    x_positions = _compute_axis_positions(width, tile_size)
    y_positions = _compute_axis_positions(height, tile_size)

    even_positions: list[tuple[int, int]] = []
    odd_positions: list[tuple[int, int]] = []
    for y_idx, y in enumerate(y_positions):
        for x_idx, x in enumerate(x_positions):
            if ((x_idx + y_idx) & 1) == 0:
                even_positions.append((x, y))
            else:
                odd_positions.append((x, y))
    positions = even_positions + odd_positions

    return [get_tile_info(image, x, y, tile_size, tile_overlap) for x, y in positions]


def calculate_efficient_tile_size(width: int, height: int) -> tuple[int, int]:
    """Round dimensions up to GPU-friendly alignment without over-padding tiny tiles."""
    alignment = _GPU_DIM_ALIGNMENT
    efficient_width = max(alignment, ((width + alignment - 1) // alignment) * alignment)
    efficient_height = max(alignment, ((height + alignment - 1) // alignment) * alignment)

    return efficient_width, efficient_height


def get_tile_info(
    image: Image.Image,
    x: int,
    y: int,
    tile_size: int,
    tile_overlap: int,
) -> TileInfo:
    """Extract tile information and crop the tile with padding.

    Uses edge extension (reflection) for memory padding instead of solid color fill
    to avoid artificial edges that the AI upscaler would process as real content.
    """
    width, height = image.size

    actual_tile_width = min(tile_size, width - x)
    actual_tile_height = min(tile_size, height - y)

    left_pad = min(tile_overlap, x)
    top_pad = min(tile_overlap, y)
    right_pad = min(tile_overlap, width - x - actual_tile_width)
    bottom_pad = min(tile_overlap, height - y - actual_tile_height)

    padded_box = (
        max(0, x - left_pad),
        max(0, y - top_pad),
        min(width, x + actual_tile_width + right_pad),
        min(height, y + actual_tile_height + bottom_pad),
    )

    tile = image.crop(padded_box)

    current_width, current_height = tile.size
    efficient_width, efficient_height = calculate_efficient_tile_size(current_width, current_height)

    memory_pad_right = efficient_width - current_width
    memory_pad_bottom = efficient_height - current_height

    if memory_pad_right > 0 or memory_pad_bottom > 0:
        tile = _pad_tile_with_reflection(
            tile=tile,
            current_width=current_width,
            current_height=current_height,
            memory_pad_right=memory_pad_right,
            memory_pad_bottom=memory_pad_bottom,
            efficient_width=efficient_width,
            efficient_height=efficient_height,
        )

    return {
        "tile": tile,
        "position": (x, y),
        "actual_size": (actual_tile_width, actual_tile_height),
        "padding": (left_pad, top_pad, right_pad, bottom_pad),
        "memory_padding": (0, 0, memory_pad_right, memory_pad_bottom),
        "original_tile_size": (current_width, current_height),
    }


def _pad_tile_with_reflection(
    tile: Image.Image,
    current_width: int,
    current_height: int,
    memory_pad_right: int,
    memory_pad_bottom: int,
    efficient_width: int,
    efficient_height: int,
) -> Image.Image:
    """Pad right/bottom edges using reflection, then edge-repeat if pad exceeds source size."""
    tile_array = np.asarray(tile)
    has_channels = tile_array.ndim == 3
    if not has_channels:
        tile_array = tile_array[:, :, np.newaxis]

    padded_array = np.empty(
        (efficient_height, efficient_width, tile_array.shape[2]),
        dtype=tile_array.dtype,
    )
    padded_array[:current_height, :current_width] = tile_array

    reflect_width = min(memory_pad_right, current_width)
    reflect_height = min(memory_pad_bottom, current_height)
    remaining_right = memory_pad_right - reflect_width
    remaining_bottom = memory_pad_bottom - reflect_height

    if reflect_width > 0:
        padded_array[:current_height, current_width : current_width + reflect_width] = tile_array[
            :, current_width - reflect_width : current_width
        ][:, ::-1]
    if remaining_right > 0:
        padded_array[:current_height, current_width + reflect_width : efficient_width] = tile_array[:, -1:, :]

    if reflect_height > 0:
        padded_array[current_height : current_height + reflect_height, :current_width] = tile_array[
            current_height - reflect_height : current_height, :
        ][::-1, :]
    if remaining_bottom > 0:
        padded_array[current_height + reflect_height : efficient_height, :current_width] = tile_array[
            -1:, :, :
        ]

    if memory_pad_right > 0 and memory_pad_bottom > 0:
        corner = tile_array[
            current_height - reflect_height : current_height,
            current_width - reflect_width : current_width,
        ]
        padded_array[
            current_height : current_height + reflect_height,
            current_width : current_width + reflect_width,
        ] = corner[::-1, ::-1]

        if remaining_right > 0:
            corner_edge_col = padded_array[
                current_height : current_height + reflect_height,
                current_width + reflect_width - 1 : current_width + reflect_width,
            ]
            padded_array[
                current_height : current_height + reflect_height,
                current_width + reflect_width : efficient_width,
            ] = np.repeat(corner_edge_col, remaining_right, axis=1)
        if remaining_bottom > 0:
            corner_edge_row = padded_array[
                current_height + reflect_height - 1 : current_height + reflect_height,
                current_width : current_width + reflect_width,
            ]
            padded_array[
                current_height + reflect_height : efficient_height,
                current_width : current_width + reflect_width,
            ] = np.repeat(corner_edge_row, remaining_bottom, axis=0)
        if remaining_right > 0 and remaining_bottom > 0:
            padded_array[
                current_height + reflect_height : efficient_height,
                current_width + reflect_width : efficient_width,
            ] = tile_array[-1, -1]

    if not has_channels:
        padded_array = padded_array[:, :, 0]

    return Image.fromarray(padded_array)


def prepare_tile(tile_info: TileInfo, ai_upscaled_tile: Image.Image, upscale_factor: float) -> PreparedTile:
    """Prepare an upscaled tile for stitching by resizing, positioning, and cropping.

    This function handles both regular overlap padding (for blending adjacent tiles)
    and memory padding (added for GPU efficiency). Memory padding must be completely
    removed as it contains reflected/extended content that shouldn't be in the output.
    """
    original_width, original_height = tile_info["original_tile_size"]
    _, _, mem_right_pad, mem_bottom_pad = tile_info["memory_padding"]
    full_tile_width = tile_info["tile"].width
    full_tile_height = tile_info["tile"].height
    target_full_width = int(full_tile_width * upscale_factor)
    target_full_height = int(full_tile_height * upscale_factor)
    if ai_upscaled_tile.size == (target_full_width, target_full_height):
        resized_tile = ai_upscaled_tile
    else:
        resized_tile = ai_upscaled_tile.resize((target_full_width, target_full_height), Image.LANCZOS)
    scaled_mem_right = int(mem_right_pad * upscale_factor)
    scaled_mem_bottom = int(mem_bottom_pad * upscale_factor)
    original_content_width = int(original_width * upscale_factor)
    original_content_height = int(original_height * upscale_factor)
    if scaled_mem_right > 0 or scaled_mem_bottom > 0:
        resized_tile = resized_tile.crop((0, 0, original_content_width, original_content_height))
    paste_x = int(tile_info["position"][0] * upscale_factor)
    paste_y = int(tile_info["position"][1] * upscale_factor)
    final_tile_width = int(tile_info["actual_size"][0] * upscale_factor)
    final_tile_height = int(tile_info["actual_size"][1] * upscale_factor)
    left_pad, top_pad, right_pad, bottom_pad = tile_info["padding"]
    scaled_left_pad = int(left_pad * upscale_factor)
    scaled_top_pad = int(top_pad * upscale_factor)
    scaled_right_pad = int(right_pad * upscale_factor)
    scaled_bottom_pad = int(bottom_pad * upscale_factor)
    keep_left = scaled_left_pad // 2
    keep_top = scaled_top_pad // 2
    keep_right = scaled_right_pad // 2
    keep_bottom = scaled_bottom_pad // 2
    crop_box = (
        scaled_left_pad - keep_left,
        scaled_top_pad - keep_top,
        min(scaled_left_pad + final_tile_width + keep_right, resized_tile.width),
        min(scaled_top_pad + final_tile_height + keep_bottom, resized_tile.height),
    )
    cropped_tile = resized_tile.crop(crop_box)
    paste_x_adjusted = max(0, paste_x - keep_left)
    paste_y_adjusted = max(0, paste_y - keep_top)
    return {
        "cropped_tile": cropped_tile,
        "paste_x": paste_x_adjusted,
        "paste_y": paste_y_adjusted,
    }


def stitch_tiles(
    tiles: list[TileInfo],
    upscaled_tiles: list[Image.Image],
    width: int,
    height: int,
    upscale_factor: float,
    progress: ProgressBar,
    original_image: Image.Image,
    anti_aliasing_strength: float = 0.0,
    blending_method: str = "multiband",
) -> Image.Image:
    from .antialias import antialias, antialias_device
    from .blending import blend_multiband, blend_weighted, multiband_device, weighted_device

    stitch_start = time.perf_counter()
    logging.info("[SeedVR Tiling] Using %s blending method", blending_method)
    args = (tiles, upscaled_tiles, original_image, width, height, upscale_factor, progress)
    if blending_method == "multiband":
        pixels = blend_multiband(*args, device=multiband_device(tiles, width, height, upscale_factor))
    elif blending_method == "simple":
        pixels = blend_weighted(*args, device=weighted_device(tiles, width, height, upscale_factor))
    else:
        raise ValueError("Unknown SeedVR blending method.")
    result = Image.fromarray(pixels)
    if anti_aliasing_strength > 0:
        result = antialias(
            result, anti_aliasing_strength, device=antialias_device(result, anti_aliasing_strength)
        )
    logging.debug(f"Stitching complete in {time.perf_counter() - stitch_start:.2f}s")
    return result
