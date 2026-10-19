"""SeedVR image tiling nodes for ComfyUI's V3 API."""

from comfy_api.latest import ComfyExtension, io

from .nodes import SeedVRTileMerge, SeedVRTileSplit


class SeedVRTilingExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SeedVRTileSplit, SeedVRTileMerge]


async def comfy_entrypoint() -> SeedVRTilingExtension:
    return SeedVRTilingExtension()
