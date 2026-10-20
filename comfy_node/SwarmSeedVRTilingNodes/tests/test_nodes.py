from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comfy_api.internal import make_locked_method_func
from comfy_extras.nodes_seedvr import SeedVR2PostProcessing, SeedVR2Preprocess

from SwarmSeedVRTilingNodes import comfy_entrypoint
from SwarmSeedVRTilingNodes.nodes import SeedVRTileMerge, SeedVRTileSplit


class NativeTilingTests(unittest.TestCase):
    def test_registered_nodes_preserve_pixels_across_irregular_tile_boundaries(self) -> None:
        extension = asyncio.run(comfy_entrypoint())
        nodes = {node.define_schema().node_id: node for node in asyncio.run(extension.get_node_list())}
        split_node = nodes["SwarmSeedVRTileSplit"]
        merge_node = nodes["SwarmSeedVRTileMerge"]
        split = make_locked_method_func(split_node, "execute", split_node.PREPARE_CLASS_CLONE(None))
        merge = make_locked_method_func(merge_node, "execute", merge_node.PREPARE_CLASS_CLONE(None))
        for height, width in [(64, 64), (70, 66), (129, 198), (2, 135)]:
            image = torch.randint(0, 256, (1, height, width, 3)).float() / 255
            tiles, layout = split(image=image, tile_size=64, tile_overlap=16)
            with self.subTest(size=(height, width)):
                result = merge(images=tiles, layout=[layout], anti_aliasing=[0.0], blending=["simple"])[0]
                self.assertEqual(result.shape, image.shape)
                self.assertLessEqual((result - image).abs().max().item(), 1 / 255 + 1e-6)

    def test_native_padding_and_global_alpha_restoration(self) -> None:
        image = torch.rand(1, 70, 98, 4)
        tiles, layout = SeedVRTileSplit.execute(image, 64, 16)
        decoded = [SeedVR2Preprocess.execute(tile)[0] for tile in tiles]
        merged = SeedVRTileMerge.execute(decoded, [layout], [0.0], ["simple"])[0]
        result = SeedVR2PostProcessing.execute(merged, image, "none")[0]
        self.assertEqual(result.shape, image.shape)
        torch.testing.assert_close(result[..., 3], image[..., 3])
        self.assertLessEqual((result[..., :3] - image[..., :3]).abs().max().item(), 2 / 255)


if __name__ == "__main__":
    unittest.main()
