from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from comfy.utils import ProgressBar
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SwarmSeedVRTilingNodes import blending as multiband
from SwarmSeedVRTilingNodes import tiling
from SwarmSeedVRTilingNodes.antialias import antialias


class BlendingScaleTests(unittest.TestCase):
    def test_each_tile_pyramid_is_local_and_constant_colors_have_no_seams(self) -> None:
        original = Image.new("RGB", (794, 526), (180, 110, 70))
        tiles = tiling.generate_tiles(original, 128, 32)
        shapes = []
        real = multiband._laplacian

        def track(image: NDArray[np.float32] | Image.Image, levels: int) -> list[NDArray[np.float32]]:
            shapes.append(np.asarray(image).shape)
            return real(image, levels)

        progress = ProgressBar(len(tiles))
        with patch.object(multiband, "_laplacian", side_effect=track):
            result = multiband.blend_multiband(
                tiles,
                [tile["tile"] for tile in tiles],
                original,
                original.width,
                original.height,
                1,
                progress,
                device="cpu",
            )
        self.assertEqual(shapes[0][:2], (526, 794))
        self.assertTrue(all(h <= 320 and w <= 320 for h, w, _ in shapes[1:]))
        self.assertEqual(progress.current, len(tiles))
        self.assertLessEqual(np.abs(result.astype(float) - np.asarray(original)).max(), 1)

    def test_interrupt_is_checked_during_pyramid_preparation(self) -> None:
        with (
            patch.object(
                multiband, "throw_exception_if_processing_interrupted", side_effect=RuntimeError("cancelled")
            ),
            self.assertRaisesRegex(RuntimeError, "cancelled"),
        ):
            multiband._laplacian(np.zeros((64, 64, 3)), 4)

    def test_strip_antialias_matches_original_full_frame_algorithm(self) -> None:
        rng = np.random.default_rng(4)
        source = rng.integers(0, 256, (139, 117, 3), dtype=np.uint8)
        for strength in [0.1, 0.3, 1.0]:
            expected = np.zeros(source.shape, dtype=np.float32)
            for channel in range(3):
                values = source[:, :, channel].astype(np.float32)
                edge = np.sqrt(ndimage.sobel(values, axis=1) ** 2 + ndimage.sobel(values, axis=0) ** 2)
                if edge.max() > 0:
                    edge /= edge.max()
                mask = 1 - ((1 - edge) * strength)
                smooth = ndimage.gaussian_filter(values, sigma=0.5 + strength * 1.5)
                expected[:, :, channel] = values * mask + smooth * (1 - mask)
            expected = np.clip(expected, 0, 255).astype(np.uint8)
            actual = np.asarray(antialias(Image.fromarray(source), strength, block_rows=17, device="cpu"))
            self.assertLessEqual(np.abs(actual.astype(float) - expected).max(), 1)
