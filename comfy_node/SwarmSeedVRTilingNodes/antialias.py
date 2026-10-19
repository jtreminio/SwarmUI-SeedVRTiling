"""Smooth jagged edges in bounded row strips."""

from __future__ import annotations

import math
from collections.abc import Sequence

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


def antialias_device(image: Image.Image, strength: float, block_rows: int = 256) -> torch.device:
    halo = max(1, math.ceil(4 * (0.5 + strength * 1.5)))
    if image.height * image.width < 1024 * 1024:
        return torch.device("cpu")
    device = get_torch_device()
    if device.type != "cuda":
        return torch.device("cpu")
    scratch = image.width * (block_rows + 2 * halo) * 3 * 4 * 12
    free = get_free_memory(device)
    return device if free >= 2 * scratch + 512 * 1024**2 else torch.device("cpu")


def antialias(
    image: Image.Image, strength: float, device: torch.device | str, block_rows: int = 256
) -> Image.Image:
    source = np.asarray(image)
    height = source.shape[0]
    sigma = 0.5 + strength * 1.5
    halo = max(1, math.ceil(4 * sigma))
    output = np.empty_like(source)
    if torch.device(device).type == "cuda":
        progress = ProgressBar(2 * math.ceil(height / block_rows))
        return _antialias_cuda(source, strength, block_rows, sigma, halo, output, progress, device)
    progress = ProgressBar(6 * math.ceil(height / block_rows))
    return _antialias_cpu(source, strength, block_rows, sigma, halo, output, progress)


@torch.inference_mode()
def _antialias_cuda(
    source: NDArray[np.uint8],
    strength: float,
    block_rows: int,
    sigma: float,
    halo: int,
    output: NDArray[np.uint8],
    progress: ProgressBar,
    device: torch.device | str,
) -> Image.Image:
    height = source.shape[0]
    max_edge = torch.zeros(3, device=device)
    radius = int(4 * sigma + 0.5)
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / sigma) ** 2)
    kernel /= kernel.sum()
    for smoothing in (False, True):
        for start in range(0, height, block_rows):
            throw_exception_if_processing_interrupted()
            end = min(height, start + block_rows)
            lo, hi = max(0, start - halo), min(height, end + halo)
            values = torch.from_numpy(source[lo:hi].copy()).to(device).float()
            edge = sobel_magnitude(values)
            if not smoothing:
                max_edge = torch.maximum(max_edge, edge[start - lo : end - lo].amax(dim=(0, 1)))
            else:
                edge /= max_edge.clamp_min(1)
                edge.mul_(strength).add_(1 - strength)
                smooth = filter_axis(filter_axis(values, kernel, 0), kernel, 1)
                smooth.mul_(1 - edge).add_(values * edge).clamp_(0, 255)
                output[start:end] = smooth[start - lo : end - lo].to(torch.uint8).cpu().numpy()
            progress.update(1)
    return Image.fromarray(output)


def _antialias_cpu(
    source: NDArray[np.uint8],
    strength: float,
    block_rows: int,
    sigma: float,
    halo: int,
    output: NDArray[np.uint8],
    progress: ProgressBar,
) -> Image.Image:
    height = source.shape[0]
    for channel in range(3):
        max_edge = 0.0
        for start in range(0, height, block_rows):
            throw_exception_if_processing_interrupted()
            end = min(height, start + block_rows)
            lo, hi = max(0, start - halo), min(height, end + halo)
            values = source[lo:hi, :, channel].astype(np.float32)
            edge: NDArray[np.float32] = ndimage.sobel(values, axis=1)
            np.hypot(edge, ndimage.sobel(values, axis=0), out=edge)
            max_edge = max(max_edge, float(np.max(edge[start - lo : end - lo])))
            progress.update(1)
        for start in range(0, height, block_rows):
            throw_exception_if_processing_interrupted()
            end = min(height, start + block_rows)
            lo, hi = max(0, start - halo), min(height, end + halo)
            values = source[lo:hi, :, channel].astype(np.float32)
            edge = ndimage.sobel(values, axis=1)
            np.hypot(edge, ndimage.sobel(values, axis=0), out=edge)
            if max_edge:
                edge /= max_edge
            edge *= strength
            edge += 1 - strength
            smoothed: NDArray[np.float32] = ndimage.gaussian_filter(values, sigma=sigma)
            smoothed *= 1 - edge
            values *= edge
            smoothed += values
            np.clip(smoothed, 0, 255, out=smoothed)
            output[start:end, :, channel] = smoothed[start - lo : end - lo].astype(np.uint8)
            progress.update(1)
    return Image.fromarray(output)


def filter_axis(
    values: torch.Tensor, weights: Sequence[float] | NDArray[np.float64], axis: int
) -> torch.Tensor:
    """Match scipy.ndimage.correlate1d's half-sample 'reflect' border mode."""
    size = values.shape[axis]
    radius = len(weights) // 2
    indices = torch.arange(-radius, size + radius, device=values.device) % (2 * size)
    indices = torch.where(indices < size, indices, 2 * size - 1 - indices)
    padded = values.index_select(axis, indices)
    result = torch.zeros_like(values)
    for offset, weight in enumerate(weights):
        if weight:
            result.add_(padded.narrow(axis, offset, size), alpha=float(weight))
    return result


def sobel_magnitude(values: torch.Tensor) -> torch.Tensor:
    horizontal = filter_axis(filter_axis(values, [-1, 0, 1], 1), [1, 2, 1], 0)
    vertical = filter_axis(filter_axis(values, [-1, 0, 1], 0), [1, 2, 1], 1)
    return torch.hypot(horizontal, vertical)
