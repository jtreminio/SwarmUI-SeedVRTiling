"""Split large images into smaller tiles for SeedVR, then blend the results back together."""

from __future__ import annotations

import logging

import torch
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.utils import ProgressBar
from comfy_api.latest import io

from .tiling import TileLayout, generate_tiles, pil_to_tensor, stitch_tiles, tensor_to_pil

BLENDING_METHODS = ["multiband", "simple"]


class SeedVRTileSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SwarmSeedVRTileSplit",
            display_name="Split SeedVR Image Tiles",
            category="image/seedvr",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("tile_size", default=512, min=64, step=8),
                io.Int.Input("tile_overlap", default=32, min=0, step=8),
            ],
            outputs=[
                io.Image.Output("tiles", is_output_list=True),
                io.Custom("SEEDVR_TILE_LAYOUT").Output("layout"),
            ],
        )

    @classmethod
    def validate_inputs(cls, tile_size: int, tile_overlap: int) -> bool | str:
        if tile_size < 64:
            return "SeedVR tile size must be at least 64 pixels."
        return (
            True
            if 0 <= tile_overlap < tile_size
            else "SeedVR tile overlap must be nonnegative and smaller than tile size."
        )

    @classmethod
    def execute(cls, image: torch.Tensor, tile_size: int, tile_overlap: int) -> io.NodeOutput:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(
                "SeedVR image tiling requires one image. Disable tiling for videos or image batches."
            )
        if min(image.shape[1:3]) < 2 or image.shape[-1] not in (3, 4):
            raise ValueError("SeedVR image tiling requires an RGB/RGBA image at least 2 pixels on each axis.")
        original = tensor_to_pil(image[..., :3].detach().cpu())
        tiles = generate_tiles(original, tile_size, tile_overlap)
        logging.info(
            "[SeedVR Tiling] %dx%d image, tile size %d, overlap %d: %d tiles",
            original.width,
            original.height,
            tile_size,
            tile_overlap,
            len(tiles),
        )
        tensors = []
        for info in tiles:
            throw_exception_if_processing_interrupted()
            tile = info["tile"]
            tensors.append(pil_to_tensor(tile))
        return io.NodeOutput(tensors, TileLayout(tiles=tiles, original=original))


class SeedVRTileMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SwarmSeedVRTileMerge",
            display_name="Merge SeedVR Image Tiles",
            category="image/seedvr",
            is_input_list=True,
            inputs=[
                io.Image.Input("images"),
                io.Custom("SEEDVR_TILE_LAYOUT").Input("layout"),
                io.Float.Input("anti_aliasing", default=0.0, min=0.0, max=1.0),
                io.Combo.Input("blending", options=BLENDING_METHODS),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        images: list[torch.Tensor],
        layout: list[TileLayout],
        anti_aliasing: list[float],
        blending: list[str],
    ) -> io.NodeOutput:
        if len(layout) != 1:
            raise ValueError("Connect the layout output from one SeedVR tile split node.")
        tile_layout = layout[0]
        if len(images) != len(tile_layout["tiles"]):
            raise ValueError("SeedVR decoded tile count does not match the layout.")
        restored = []
        for image, info in zip(images, tile_layout["tiles"]):
            width, height = info["tile"].size
            throw_exception_if_processing_interrupted()
            if image.ndim == 5 and image.shape[:2] == (1, 1):
                image = image[0]
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] < height or image.shape[2] < width:
                raise ValueError("SeedVR decoded tile has an unexpected shape.")
            restored.append(tensor_to_pil(image[:, :height, :width, :3].detach().cpu()))
        original = tile_layout["original"]
        progress = ProgressBar(len(restored))
        result = stitch_tiles(
            tiles=tile_layout["tiles"],
            upscaled_tiles=restored,
            width=original.width,
            height=original.height,
            upscale_factor=1.0,
            progress=progress,
            original_image=original,
            anti_aliasing_strength=anti_aliasing[0],
            blending_method=blending[0],
        )
        logging.info("[SeedVR Tiling] Stitching finished")
        return io.NodeOutput(pil_to_tensor(result))
