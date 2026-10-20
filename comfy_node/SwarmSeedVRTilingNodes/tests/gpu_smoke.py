"""Opt-in smoke test with real native models. Run with ComfyUI on PYTHONPATH.

python comfy_node/SwarmSeedVRTilingNodes/tests/gpu_smoke.py --models /path/to/Models --output /tmp/seedvr-smoke.png
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--swarm-sampler", type=Path, help="Path to SwarmKSampler.py to test Swarm's native sampler"
)
parser.add_argument("--width", type=int, default=384)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--method", choices=["multiband", "simple", "all"], default="multiband")
options = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folder_paths
import nodes
import torch
from comfy_extras.nodes_seedvr import SeedVR2Conditioning, SeedVR2PostProcessing, SeedVR2Preprocess

from SwarmSeedVRTilingNodes.nodes import SeedVRTileMerge, SeedVRTileSplit
from SwarmSeedVRTilingNodes.tiling import tensor_to_pil

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
swarm_sampler = None
if options.swarm_sampler:
    spec = importlib.util.spec_from_file_location("seedvr_smoke_swarm_sampler", options.swarm_sampler)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    swarm_sampler = module.SwarmKSampler()

folder_paths.add_model_folder_path("diffusion_models", str(options.models / "diffusion_models"))
folder_paths.add_model_folder_path("vae", str(options.models / "VAE"))
model = nodes.UNETLoader().load_unet("seedvr2/seedvr2_3b_int8_convrot.safetensors", "default")[0]
vae = nodes.VAELoader().load_vae("SeedVR2/seedvr2_ema_vae_fp16.safetensors")[0]
y, x = torch.meshgrid(
    torch.linspace(0, 1, options.height), torch.linspace(0, 1, options.width), indexing="ij"
)
source = torch.stack((x, y, (torch.sin(x * 25) + 1) / 2), dim=-1).unsqueeze(0)
with torch.inference_mode():
    tiles, layout = SeedVRTileSplit.execute(source, 256, 32)
    decoded = []
    for index, tile in enumerate(tiles):
        padded = SeedVR2Preprocess.execute(tile)[0]
        latent = nodes.VAEEncode().encode(vae, padded)[0]
        conditioning = SeedVR2Conditioning.execute(model, latent)
        if swarm_sampler is None:
            sampled = nodes.common_ksampler(
                model, 42, 1, 1.0, "euler", "simple", conditioning[0], conditioning[1], latent
            )[0]
        else:
            sampled = swarm_sampler.run_sampling(
                model=model,
                noise_seed=42,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                positive=conditioning[0],
                negative=conditioning[1],
                latent_image=latent,
                start_at_step=0,
                end_at_step=10000,
                var_seed=0,
                var_seed_strength=0.0,
                sigma_max=-1.0,
                sigma_min=-1.0,
                rho=7.0,
                add_noise="enable",
                return_with_leftover_noise="disable",
                previews="none",
                tile_sample=False,
                tile_size=1024,
            )[0]
        restored = nodes.VAEDecode().decode(vae, sampled)[0]
        decoded.append(restored)
        logging.info("Restored tile %d/%d: %s", index + 1, len(tiles), tuple(restored.shape))
    methods = ["multiband", "simple"] if options.method == "all" else [options.method]
    for method in methods:
        stitched = SeedVRTileMerge().execute(decoded, [layout], [0.3], [method])[0]
        result = SeedVR2PostProcessing.execute(stitched, source, "lab")[0]
        assert result.shape == source.shape, (result.shape, source.shape)
        assert torch.isfinite(result).all()
        output = (
            options.output.with_stem(f"{options.output.stem}-{method}")
            if len(methods) > 1
            else options.output
        )
        tensor_to_pil(result.cpu()).save(output)
        print(f"PASS: {method}: {len(tiles)} native SeedVR tiles; output {tuple(result.shape)}; {output}")

# Release model patchers while ComfyUI's modules still exist (before interpreter teardown).
import comfy.model_management

comfy.model_management.unload_all_models()
del model, vae
