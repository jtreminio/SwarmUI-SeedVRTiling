from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SwarmSeedVRTilingNodes.antialias import antialias


class GpuAntialiasTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_matches_cpu_at_strip_boundaries_and_image_edges(self) -> None:
        rng = np.random.default_rng(10)
        for shape in [(139, 117, 3), (2, 2, 3), (21, 1, 3)]:
            for strength in [0.1, 0.3, 1.0]:
                source = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
                cpu = np.asarray(antialias(source, strength, block_rows=17, device="cpu"))
                gpu = np.asarray(antialias(source, strength, block_rows=17, device="cuda"))
                self.assertLessEqual(np.abs(cpu.astype(float) - gpu).max(), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_preserves_constant_channels(self) -> None:
        source = Image.new("RGB", (121, 137), (180, 0, 255))
        actual = np.asarray(antialias(source, 0.3, block_rows=17, device="cuda"))
        self.assertLessEqual(np.abs(actual.astype(float) - np.asarray(source)).max(), 1)
