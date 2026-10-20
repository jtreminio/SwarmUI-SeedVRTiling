from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from comfy.utils import ProgressBar
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SwarmSeedVRTilingNodes import blending as multiband
from SwarmSeedVRTilingNodes import tiling


class TensorMultibandTests(unittest.TestCase):
    def test_tensor_pyramid_matches_numpy_without_cuda(self) -> None:
        source = Image.fromarray(np.random.default_rng(7).integers(0, 256, (137, 269, 3), dtype=np.uint8))
        tiles = tiling.generate_tiles(source, 64, 16)
        restored = [tile["tile"] for tile in tiles]
        cpu = multiband.blend_multiband(tiles, restored, source, 269, 137, 1, ProgressBar(len(tiles)), "cpu")
        tensor = multiband.blend_multiband_cuda(
            tiles, restored, source, 269, 137, 1, ProgressBar(len(tiles)), "cpu"
        )
        self.assertLessEqual(np.abs(cpu.astype(float) - tensor).max(), 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class GpuMultibandTests(unittest.TestCase):
    def test_filters_match_cpu_at_odd_and_tiny_dimensions(self) -> None:
        rng = np.random.default_rng(42)
        for shape in [(1, 1), (2, 3), (19, 17), (19, 17, 3), (3, 2, 3)]:
            source = rng.random(shape, dtype=np.float32) * 255
            gpu = torch.from_numpy(source).cuda()
            down = multiband._down_cuda(gpu)
            np.testing.assert_allclose(down.cpu().numpy(), multiband._down(source), atol=4e-5, rtol=1e-6)
            up = multiband._up_cuda(down, source.shape).cpu().numpy()
            expected = multiband._up(multiband._down(source), source.shape)
            np.testing.assert_allclose(up, expected, atol=8e-5, rtol=1e-6)

    def test_different_restored_tiles_match_cpu_with_overlap_and_reversed_input(self) -> None:
        rng = np.random.default_rng(3)
        source = Image.fromarray(rng.integers(0, 256, (269, 397, 3), dtype=np.uint8))
        for reverse in [False, True]:
            for overlap in [0, 32, 63]:
                tiles = tiling.generate_tiles(source, 128, overlap)
                if reverse:
                    tiles.reverse()
                # Give every tile different restoration detail to exercise masks,
                # Laplacian blending, overlap order, and all image boundaries.
                restored = [
                    Image.fromarray(
                        rng.integers(0, 256, (tile["tile"].height, tile["tile"].width, 3), dtype=np.uint8)
                    )
                    for tile in tiles
                ]
                cpu_progress, gpu_progress = ProgressBar(len(tiles)), ProgressBar(len(tiles))
                args = (tiles, restored, source, source.width, source.height, 1)
                cpu = multiband.blend_multiband(*args, cpu_progress, device="cpu")
                gpu = multiband.blend_multiband_cuda(*args, gpu_progress, "cuda")
                self.assertEqual(gpu_progress.current, len(tiles))
                self.assertLessEqual(np.abs(cpu.astype(float) - gpu).max(), 1)

    def test_constant_color_and_single_tile_have_no_dark_edges(self) -> None:
        for size in [(2, 2), (37, 19), (794, 526)]:
            source = Image.new("RGB", size, (180, 110, 70))
            tiles = tiling.generate_tiles(source, 128, 32)
            actual = multiband.blend_multiband_cuda(
                tiles, [t["tile"] for t in tiles], source, *size, 1, ProgressBar(len(tiles)), "cuda"
            )
            self.assertLessEqual(np.abs(actual.astype(float) - np.asarray(source)).max(), 1)

    def test_cancellation_interrupts_pyramid_work(self) -> None:
        with (
            patch.object(
                multiband, "throw_exception_if_processing_interrupted", side_effect=RuntimeError("cancelled")
            ),
            self.assertRaisesRegex(RuntimeError, "cancelled"),
        ):
            multiband._laplacian_cuda(torch.zeros((64, 64, 3), device="cuda"), 4)
