"""Multiband pyramids and simple overlap averaging, on CPU or CUDA."""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from comfy.model_management import (
    get_free_memory,
    get_torch_device,
    throw_exception_if_processing_interrupted,
)
from comfy.utils import ProgressBar
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

from .tiling import TileInfo, prepare_tile

_KERNEL = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16


def _blur(image: NDArray[np.float32]) -> NDArray[np.float32]:
    result: NDArray[np.float32] = ndimage.convolve1d(image, _KERNEL, axis=0, mode="mirror")
    return ndimage.convolve1d(result, _KERNEL, axis=1, mode="mirror")


def _down(image: NDArray[np.float32]) -> NDArray[np.float32]:
    return _blur(image)[::2, ::2].copy()


def _up(image: NDArray[np.float32], shape: tuple[int, ...]) -> NDArray[np.float32]:
    result = np.zeros(shape, dtype=np.float32)
    result[::2, ::2] = image
    result = ndimage.convolve1d(result, _KERNEL * 2, axis=0, mode="mirror")
    return ndimage.convolve1d(result, _KERNEL * 2, axis=1, mode="mirror")


def _laplacian(image: NDArray[np.float32] | Image.Image, levels: int) -> list[NDArray[np.float32]]:
    current = np.array(image, dtype=np.float32)
    result = []
    for _ in range(levels):
        throw_exception_if_processing_interrupted()
        down = _down(current)
        current -= _up(down, current.shape)
        result.append(current)
        current = down
    result.append(current)
    return result


def _tile_patch(
    info: TileInfo,
    image: Image.Image,
    width: int,
    height: int,
    scale: float,
    alignment: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], int, int] | None:
    """Shared geometry and masks for both CPU and CUDA pyramid arithmetic."""
    halo = 4 * alignment
    prepared = prepare_tile(info, image, scale)
    tile = np.asarray(prepared["cropped_tile"], dtype=np.float32)
    x, y = prepared["paste_x"], prepared["paste_y"]
    h, w = min(tile.shape[0], height - y), min(tile.shape[1], width - x)
    if h <= 0 or w <= 0:
        return None
    tile = tile[:h, :w]
    # Clamp to the image, but keep interior patch origins aligned to all levels.
    x0 = max(0, ((x - halo) // alignment) * alignment)
    y0 = max(0, ((y - halo) // alignment) * alignment)
    x1 = min(width, ((x + w + halo + alignment - 1) // alignment) * alignment)
    y1 = min(height, ((y + h + halo + alignment - 1) // alignment) * alignment)
    left, top = x - x0, y - y0
    right, bottom = x1 - x - w, y1 - y - h
    padded = np.pad(tile, ((top, bottom), (left, right), (0, 0)), mode="symmetric")
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    feather = min(32, w // 4, h // 4)
    weights = np.ones((h, w), dtype=np.float32)
    if feather:
        # Outer image edges have no adjacent tile to blend against.
        if x > 0:
            weights *= np.minimum(1, np.arange(w, dtype=np.float32) / feather)[None, :]
        if x + w < width:
            weights *= np.minimum(1, np.arange(w - 1, -1, -1, dtype=np.float32) / feather)[None, :]
        if y > 0:
            weights *= np.minimum(1, np.arange(h, dtype=np.float32) / feather)[:, None]
        if y + h < height:
            weights *= np.minimum(1, np.arange(h - 1, -1, -1, dtype=np.float32) / feather)[:, None]
    mask[top : top + h, left : left + w] = weights
    return padded, mask, x0, y0


def _cuda_memory_budget(tiles: list[TileInfo], width: int, height: int, scale: float) -> int:
    """Conservative allocator budget, including caching/fragmentation and reserve."""
    canvas = width * height * 3 * 4
    levels = min(4, int(np.log2(min(width, height))))
    # Include reflection halos and alignment rounding around the largest tile.
    margin = 10 * 2**levels
    patch = max(
        (
            min(width, int(info["tile"].width * scale) + margin)
            * min(height, int(info["tile"].height * scale) + margin)
            * 3
            * 4
            for info in tiles
        ),
        default=0,
    )
    # The measured live tensor peak is about 3.25 canvases. Six canvases allow
    # for caching/fragmentation; the patch term covers unusually large tiles.
    return 6 * canvas + 4 * patch + 2 * 1024**3


def _select_device(width: int, height: int, required: int) -> torch.device:
    if width * height < 1024 * 1024:
        return torch.device("cpu")
    device = get_torch_device()
    if device.type != "cuda":
        return torch.device("cpu")
    free = get_free_memory(device)
    logging.debug(
        "[SeedVR Tiling] Blending VRAM: %.2f GiB available, %.2f GiB required",
        free / 1024**3,
        required / 1024**3,
    )
    return device if free >= required else torch.device("cpu")


def multiband_device(tiles: list[TileInfo], width: int, height: int, scale: float) -> torch.device:
    return _select_device(width, height, _cuda_memory_budget(tiles, width, height, scale))


def blend_multiband(
    tiles: list[TileInfo],
    restored: list[Image.Image],
    original: Image.Image,
    width: int,
    height: int,
    scale: float,
    progress: ProgressBar,
    device: torch.device | str,
) -> NDArray[np.uint8]:
    if torch.device(device).type == "cuda":
        return blend_multiband_cuda(tiles, restored, original, width, height, scale, progress, device)
    return _blend_multiband_cpu(tiles, restored, original, width, height, scale, progress)


def _blend_multiband_cpu(
    tiles: list[TileInfo],
    restored: list[Image.Image],
    original: Image.Image,
    width: int,
    height: int,
    scale: float,
    progress: ProgressBar,
) -> NDArray[np.uint8]:
    levels = min(4, int(np.log2(min(width, height))))
    alignment = 2**levels
    logging.info("[SeedVR Tiling] Preparing output pyramid")
    output = _laplacian(original, levels)
    for info, image in zip(tiles, restored):
        throw_exception_if_processing_interrupted()
        patch = _tile_patch(info, image, width, height, scale, alignment)
        if patch is None:
            progress.update(1)
            continue
        padded, mask, x0, y0 = patch
        tile_pyramid = _laplacian(padded, levels)
        for level, band in enumerate(tile_pyramid):
            throw_exception_if_processing_interrupted()
            ox, oy = x0 // (2**level), y0 // (2**level)
            region = output[level][oy : oy + band.shape[0], ox : ox + band.shape[1]]
            # In-place operations allocate only tile-sized temporaries.
            region *= 1 - mask[..., None]
            region += band * mask[..., None]
            if level < levels:
                mask = _down(mask)
        progress.update(1)
    logging.info("[SeedVR Tiling] Reconstructing output pyramid")
    result = output.pop()
    while output:
        throw_exception_if_processing_interrupted()
        band = output.pop()
        result = _up(result, band.shape)
        result += band
    np.clip(result, 0, 255, out=result)
    return result.astype(np.uint8)


def _mirrored_slice(values: torch.Tensor, axis: int, start: int, length: int) -> torch.Tensor:
    size = values.shape[axis]
    indices = torch.arange(start, start + length, device=values.device)
    if size == 1:
        indices.zero_()
    else:
        period = 2 * (size - 1)
        indices %= period
        indices = torch.where(indices < size, indices, period - indices)
    return values.index_select(axis, indices)


def _filter_axis(values: torch.Tensor, axis: int, gain: float = 1) -> torch.Tensor:
    """Match scipy.ndimage.convolve1d(..., mode='mirror') for this symmetric kernel."""
    size = values.shape[axis]
    result = torch.zeros_like(values)
    for offset, weight in enumerate(_KERNEL):
        shift = offset - 2
        start, end = max(0, -shift), min(size, size - shift)
        if start < end:
            result.narrow(axis, start, end - start).add_(
                values.narrow(axis, start + shift, end - start), alpha=weight * gain
            )
        left = min(size, start)
        if left:
            result.narrow(axis, 0, left).add_(_mirrored_slice(values, axis, shift, left), alpha=weight * gain)
        right = max(0, end)
        if right < size:
            result.narrow(axis, right, size - right).add_(
                _mirrored_slice(values, axis, right + shift, size - right), alpha=weight * gain
            )
    return result


def _down_cuda(values: torch.Tensor) -> torch.Tensor:
    vertical = _filter_axis(values, 0)
    return _filter_axis(vertical, 1)[::2, ::2].clone()


def _up_cuda(values: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    result = torch.zeros(shape, device=values.device, dtype=torch.float32)
    result[::2, ::2] = values
    result = _filter_axis(result, 0, gain=2)
    return _filter_axis(result, 1, gain=2)


def _laplacian_cuda(current: torch.Tensor, levels: int) -> list[torch.Tensor]:
    output = []
    for _ in range(levels):
        throw_exception_if_processing_interrupted()
        down = _down_cuda(current)
        current.sub_(_up_cuda(down, current.shape))
        output.append(current)
        current = down
    output.append(current)
    return output


@torch.inference_mode()
def blend_multiband_cuda(
    tiles: list[TileInfo],
    restored: list[Image.Image],
    original: Image.Image,
    width: int,
    height: int,
    scale: float,
    progress: ProgressBar,
    device: torch.device | str,
) -> NDArray[np.uint8]:
    levels = min(4, int(math.log2(min(width, height))))
    alignment = 2**levels
    logging.info("[SeedVR Tiling] Preparing output pyramid (GPU)")
    source = torch.from_numpy(np.array(original)).to(device=device, dtype=torch.float32)
    output = _laplacian_cuda(source, levels)
    # The first output band now owns the source storage.
    del source
    for info, image in zip(tiles, restored):
        throw_exception_if_processing_interrupted()
        patch = _tile_patch(info, image, width, height, scale, alignment)
        if patch is None:
            progress.update(1)
            continue
        padded, weights, x0, y0 = patch
        current = torch.from_numpy(padded).to(device)
        tile_pyramid = _laplacian_cuda(current, levels)
        mask = torch.from_numpy(weights).to(device)
        for level, band in enumerate(tile_pyramid):
            throw_exception_if_processing_interrupted()
            ox, oy = x0 // (2**level), y0 // (2**level)
            region = output[level][oy : oy + band.shape[0], ox : ox + band.shape[1]]
            region.mul_(1 - mask[..., None])
            region.add_(band * mask[..., None])
            if level < levels:
                mask = _down_cuda(mask)
        progress.update(1)
        del current, tile_pyramid, mask, band, region
    logging.info("[SeedVR Tiling] Reconstructing output pyramid (GPU)")
    result = output.pop()
    while output:
        throw_exception_if_processing_interrupted()
        band = output.pop()
        result = _up_cuda(result, band.shape)
        result.add_(band)
    result.clamp_(0, 255)
    return result.to(torch.uint8).cpu().numpy()


def _weighted_memory_budget(tiles: list[TileInfo], width: int, height: int, scale: float) -> int:
    pixels = width * height
    patch = max(
        (
            min(width, int(tile["tile"].width * scale)) * min(height, int(tile["tile"].height * scale))
            for tile in tiles
        ),
        default=0,
    )
    # RGB float output + scalar weights + final RGB uint8 conversion.
    # Scratch is bounded by a tile.
    live = pixels * 19
    scratch = patch * 48
    return 2 * (live + scratch) + 1024**3


def weighted_device(tiles: list[TileInfo], width: int, height: int, scale: float) -> torch.device:
    return _select_device(width, height, _weighted_memory_budget(tiles, width, height, scale))


def blend_weighted(
    tiles: list[TileInfo],
    restored: list[Image.Image],
    original: Image.Image,
    width: int,
    height: int,
    scale: float,
    progress: ProgressBar,
    device: torch.device | str,
) -> NDArray[np.uint8]:
    if torch.device(device).type == "cuda":
        return blend_weighted_cuda(tiles, restored, original, width, height, scale, progress, device)
    return _blend_weighted_cpu(tiles, restored, original, width, height, scale, progress)


def _blend_weighted_cpu(
    tiles: list[TileInfo],
    restored: list[Image.Image],
    original: Image.Image,
    width: int,
    height: int,
    scale: float,
    progress: ProgressBar,
) -> NDArray[np.uint8]:
    logging.info("[SeedVR Tiling] Preparing Simple output (CPU)")
    output = np.array(original, dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for info, image in zip(tiles, restored):
        throw_exception_if_processing_interrupted()
        prepared = prepare_tile(info, image, scale)
        x, y = prepared["paste_x"], prepared["paste_y"]
        tile = np.array(prepared["cropped_tile"], dtype=np.float32)
        end_x, end_y = min(x + tile.shape[1], width), min(y + tile.shape[0], height)
        h, w = end_y - y, end_x - x
        if h <= 0 or w <= 0:
            progress.update(1)
            continue
        region, weight = output[y:end_y, x:end_x], weights[y:end_y, x:end_x]
        values = tile[:h, :w]
        new_weight = weight + 1.0
        region[:] = (region * weight[..., None] + values) / new_weight[..., None]
        weight[:] = new_weight
        progress.update(1)
    return np.clip(output, 0, 255).astype(np.uint8)


@torch.inference_mode()
def blend_weighted_cuda(
    tiles: list[TileInfo],
    restored: list[Image.Image],
    original: Image.Image,
    width: int,
    height: int,
    scale: float,
    progress: ProgressBar,
    device: torch.device | str,
) -> NDArray[np.uint8]:
    logging.info("[SeedVR Tiling] Preparing Simple output (GPU)")
    output = torch.from_numpy(np.array(original)).to(device=device, dtype=torch.float32)
    weights = torch.zeros((height, width), dtype=torch.float32, device=device)
    for info, image in zip(tiles, restored):
        throw_exception_if_processing_interrupted()
        prepared = prepare_tile(info, image, scale)
        x, y = prepared["paste_x"], prepared["paste_y"]
        tile = torch.from_numpy(np.array(prepared["cropped_tile"])).to(device=device, dtype=torch.float32)
        end_x, end_y = min(x + tile.shape[1], width), min(y + tile.shape[0], height)
        h, w = end_y - y, end_x - x
        if h <= 0 or w <= 0:
            del tile
            progress.update(1)
            continue
        region, weight = output[y:end_y, x:end_x], weights[y:end_y, x:end_x]
        values = tile[:h, :w]
        region.mul_(weight[..., None]).add_(values)
        weight.add_(1.0)
        region.div_(weight[..., None])
        del tile, values, region, weight
        progress.update(1)
    throw_exception_if_processing_interrupted()
    output.clamp_(0, 255)
    return output.to(torch.uint8).cpu().numpy()
