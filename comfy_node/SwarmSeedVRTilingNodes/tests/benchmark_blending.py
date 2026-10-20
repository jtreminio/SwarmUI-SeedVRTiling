"""Opt-in blending benchmark; does not load models or write a generated image."""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=2048)
parser.add_argument("--height", type=int, default=3072)
parser.add_argument("--tile-size", type=int, default=512)
parser.add_argument("--anti-aliasing", type=float, default=0.3)
parser.add_argument("--antialias-device", choices=["auto", "cpu", "cuda"], default="auto")
parser.add_argument("--device", "--multiband-device", choices=["auto", "cpu", "cuda"], default="auto")
parser.add_argument("--method", choices=["multiband", "simple"], default="multiband")
parser.add_argument("--pattern", choices=["constant", "noise"], default="constant")
parser.add_argument("--report", type=Path, help="Write benchmark measurements as JSON")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import comfy.cli_args

if args.device == "cpu":
    comfy.cli_args.args.cpu = True

from comfy.utils import ProgressBar

from SwarmSeedVRTilingNodes import blending, tiling
from SwarmSeedVRTilingNodes.antialias import antialias, antialias_device
from SwarmSeedVRTilingNodes.blending import _cuda_memory_budget, blend_multiband

blend = blend_multiband
budget = _cuda_memory_budget
select_device = blending.multiband_device
if args.method != "multiband":
    blend = blending.blend_weighted
    budget = blending._weighted_memory_budget
    select_device = blending.weighted_device


logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
started = time.monotonic()
if args.pattern == "constant":
    original = Image.new("RGB", (args.width, args.height), (180, 110, 70))
else:
    original = Image.fromarray(
        np.random.default_rng(42).integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    )
tiles = tiling.generate_tiles(original, args.tile_size, 32)
print(f"Benchmark: {args.width}x{args.height}, {len(tiles)} tiles", flush=True)
if args.device == "cuda":
    free, total = torch.cuda.mem_get_info()
    print(f"Available VRAM before benchmark: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB", flush=True)
    estimate = budget(tiles, args.width, args.height, 1)
    if free < estimate:
        raise RuntimeError(f"Benchmark needs a conservative {estimate / 1024**3:.2f} GiB of free VRAM")
progress = ProgressBar(len(tiles))
result = Image.fromarray(
    blend(
        tiles,
        [tile["tile"] for tile in tiles],
        original,
        args.width,
        args.height,
        1.0,
        progress,
        device=select_device(tiles, args.width, args.height, 1) if args.device == "auto" else args.device,
    )
)
blending_seconds = time.monotonic() - started
print(f"{args.method} including setup: {blending_seconds:.2f}s", flush=True)
aa_seconds = 0
if args.anti_aliasing:
    aa_started = time.monotonic()
    result = antialias(
        result,
        args.anti_aliasing,
        device=antialias_device(result, args.anti_aliasing)
        if args.antialias_device == "auto"
        else args.antialias_device,
    )
    aa_seconds = time.monotonic() - aa_started
    print(f"Anti-aliasing: {aa_seconds:.2f}s", flush=True)
assert result.size == original.size
if args.pattern == "constant" or not args.anti_aliasing:
    for point in [(0, 0), (args.width // 2, args.height // 2), (args.width - 1, args.height - 1)]:
        assert max(abs(a - b) for a, b in zip(result.getpixel(point), original.getpixel(point))) <= 2
elapsed = time.monotonic() - started
rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"PASS: {elapsed:.2f}s; peak RSS {rss_gib:.2f} GiB", flush=True)
if torch.cuda.is_available():
    print(f"Peak GPU allocation: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB", flush=True)
    print(f"Peak GPU reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MiB", flush=True)
if args.report:
    args.report.write_text(
        json.dumps(
            {
                "width": args.width,
                "height": args.height,
                "tiles": len(tiles),
                "method": args.method,
                "pattern": args.pattern,
                "device": args.device,
                "antialias_device": args.antialias_device,
                "anti_aliasing": args.anti_aliasing,
                "total_seconds": elapsed,
                "blending_with_setup_seconds": blending_seconds,
                "antialias_seconds": aa_seconds,
                "peak_rss_gib": rss_gib,
                "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0,
                "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0,
                "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "scope": "Synthetic RGB image, blending and optional anti-aliasing only; excludes models, cached workflow tensors and native final color correction.",
            },
            indent=2,
        )
        + "\n"
    )
