from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from comfy.utils import ProgressBar
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SwarmSeedVRTilingNodes import blending as weighted
from SwarmSeedVRTilingNodes import tiling
from SwarmSeedVRTilingNodes.tiling import prepare_tile


class SimpleBlendingTests(unittest.TestCase):
    def test_simple_overlaps_match_independent_sum_and_count(self) -> None:
        source = Image.new("RGB", (269, 137), (0, 0, 0))
        tiles = tiling.generate_tiles(source, 128, 32)
        restored = [
            Image.new("RGB", info["tile"].size, (20 + i * 30, 80, 140)) for i, info in enumerate(tiles)
        ]
        sums = np.zeros((137, 269, 3), dtype=np.float64)
        counts = np.zeros((137, 269), dtype=np.float64)
        for info, image in zip(tiles, restored):
            prepared = prepare_tile(info, image, 1)
            tile = np.asarray(prepared["cropped_tile"])
            x, y = prepared["paste_x"], prepared["paste_y"]
            sums[y : y + tile.shape[0], x : x + tile.shape[1]] += tile
            counts[y : y + tile.shape[0], x : x + tile.shape[1]] += 1
        self.assertTrue(np.all(counts > 0))
        expected = (sums / counts[..., None]).astype(np.uint8)
        actual = weighted.blend_weighted(
            tiles, restored, source, 269, 137, 1, ProgressBar(len(tiles)), device="cpu"
        )
        self.assertLessEqual(np.abs(actual.astype(float) - expected).max(), 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class SimpleGpuTests(unittest.TestCase):
    def test_restored_tiles_match_cpu_for_irregular_edges_and_overlap(self) -> None:
        rng = np.random.default_rng(2)
        source = Image.fromarray(rng.integers(0, 256, (269, 397, 3), dtype=np.uint8))
        for overlap in (0, 32, 63):
            tiles = tiling.generate_tiles(source, 128, overlap)
            restored = [
                Image.fromarray(
                    rng.integers(0, 256, (tile["tile"].height, tile["tile"].width, 3), dtype=np.uint8)
                )
                for tile in tiles
            ]
            args = (tiles, restored, source, source.width, source.height, 1)
            cpu = weighted.blend_weighted(*args, ProgressBar(len(tiles)), device="cpu")
            progress = ProgressBar(len(tiles))
            gpu = weighted.blend_weighted(*args, progress, device="cuda")
            self.assertEqual(progress.current, len(tiles))
            self.assertLessEqual(np.abs(cpu.astype(float) - gpu).max(), 1)

    def test_constant_colors_single_tiles_and_narrow_images_have_no_seams(self) -> None:
        for size in [(2, 2), (37, 19), (2, 135), (794, 526)]:
            source = Image.new("RGB", size, (180, 110, 70))
            tiles = tiling.generate_tiles(source, 128, 32)
            result = weighted.blend_weighted(
                tiles, [t["tile"] for t in tiles], source, *size, 1, ProgressBar(len(tiles)), device="cuda"
            )
            self.assertLessEqual(np.abs(result.astype(float) - np.asarray(source)).max(), 1)


if __name__ == "__main__":
    unittest.main()
