# SwarmUI SeedVR Tiling

Restore large images with SeedVR using smaller, overlapping tiles. This reduces the GPU memory needed to process each part of the image, then blends the restored tiles back together. Controls appear inside SwarmUI’s existing **SeedVR** group.

## Will this help me?

- **SeedVR runs out of VRAM on large images:** tiling lets you reduce the size processed at once.
- **You want to restore larger images:** adjust tile size to fit your GPU while keeping the full output resolution.
- **Your images already fit comfortably:** processing the whole image gives SeedVR more context and avoids possible tile seams. Tiling can take longer and change fine detail.

This extension works with **single still images** in ordinary Swarm-generated workflows. Videos and image batches use SwarmUI’s normal processing. Custom Comfy workflows and SwarmUI's standalone **Run SeedVR** shortcut bypass the tiling hooks. Joining tiles, final color correction, and saving still need memory for the full image; tiling does not remove every memory limit.

## Installation

Install **SeedVRTiling** through **Server → Extensions**, restart SwarmUI, then refresh your browser. You need working built-in SeedVR support with a model and VAE. The extension uses Python packages already included with ComfyUI; no additional package installation is needed.

For manual installation, run this from your SwarmUI directory, then restart:

```sh
git clone https://github.com/jtreminio/SwarmUI-SeedVRTiling src/Extensions/SwarmUI-SeedVRTiling
```

Managed ComfyUI backends load the bundled nodes automatically. For an external backend, copy `comfy_node/SwarmSeedVRTilingNodes` into its `custom_nodes` directory and restart ComfyUI. Keep that copy updated with the extension.

## Getting started

Enable **SeedVR** and choose your model. In the same group, check **SeedVR Image Tiling** to show its settings. Start with **512-pixel tiles**, **32-pixel overlap**, and **anti-aliasing off**, then generate normally. Unchecking it hides the tiling settings and leaves ordinary SeedVR processing enabled.

| Setting | Default | How to choose |
| --- | --- | --- |
| Tile Size | 512 | Measured after upscaling. Lower it if restoration runs out of VRAM; increase it for more image context when memory allows. |
| Tile Overlap | 32 | More overlap can improve tile edges, but adds work and memory use. Keep it smaller than Tile Size. |
| Blending Method | Multiband | **Multiband** smooths seams and color transitions. **Simple** averages overlaps with less memory and processing, but seams may be more visible. |
| Anti-Aliasing Strength | 0 | Leave off to preserve detail. Try a low value for jagged edges; higher values soften texture. |
| Fallback on mega-dimensions | PNG | Choose PNG for lossless output and transparency, or JPG for smaller files when an image exceeds WebP’s size limit. |

Enable SwarmUI’s advanced controls to see Blending Method and Anti-Aliasing Strength. Blending and anti-aliasing use the GPU when enough memory is available, otherwise the CPU. SeedVR Split Latent has no effect while image tiling is enabled.

## Benchmarks

These saved development measurements show **the time to join tiles**, including setup and the optional smoothing shown below. They **exclude SeedVR restoration, model loading, final color correction, and saving**. They are not full generation times or measurements of VRAM savings versus untiled SeedVR.

Measured October 19, 2026, on an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, using synthetic images in development versions 0.4.0–0.5.0:

| Image size | Tiles | Blending | Anti-aliasing | Time | Peak VRAM reserved |
| --- | ---: | --- | ---: | ---: | ---: |
| 4,096 × 6,144 | 24 | Simple | Off | 0.29 s | 0.46 GiB |
| 10,816 × 15,808 | 176 | Simple | Off | 1.80 s | 3.04 GiB |
| 13,312 × 19,456 | 988 | Multiband | 0.3 | 6.66 s | 12.30 GiB |

The Simple runs used noise images; the Multiband run used a constant-color image. These are examples at different workloads, not a direct comparison between blending methods. VRAM figures include GPU cache reserved by the benchmark and exclude SeedVR models. Your GPU, image, tile settings, and available memory will affect results.

## Saving very large images

WebP allows at most **16,383 pixels on either side**. Larger outputs automatically use your selected PNG or JPG fallback. Images exceeding JPG’s **65,535-pixel side limit** use PNG. Images within WebP’s size limit keep your selected WebP format.

## Maintenance

Use **Server → Extensions** to update, disable, or uninstall the extension, then restart SwarmUI and refresh your browser. For a manual Git installation, update with `git -C src/Extensions/SwarmUI-SeedVRTiling pull` from the SwarmUI directory before restarting. External ComfyUI installations also need their copied node package updated or removed, followed by a ComfyUI restart.

To stop tiling for a generation, uncheck **SeedVR Image Tiling**. Ordinary SeedVR restoration remains available. Local C# edits require recompilation and a server restart.

## Development checks

Run `./run-tests` from this extension's directory with .NET 8 and a Python environment containing ComfyUI's dependencies:

```sh
COMFYUI_ROOT=/path/to/ComfyUI \
PYTHON_BIN=/path/to/comfy-python \
SEEDVR_TEST_CPU=1 \
./run-tests
```

The runner checks C# workflows and image encoding in `Tests/`, and Python nodes and blending in `comfy_node/SwarmSeedVRTilingNodes/tests/`. `SEEDVR_TEST_CPU=1` allows CPU-only checks; CUDA-specific tests skip when CUDA is unavailable. Optional `SWARM_ASSEMBLY` selects a compiled SwarmUI assembly for the C# tests.

ComfyUI scans immediate subdirectories of `comfy_node/` as node packages. Keep tests inside `SwarmSeedVRTilingNodes/` and tool caches outside `comfy_node/` to avoid failed imports.

These checks do not run SeedVR model inference. After recompiling and restarting SwarmUI, verify that tiling controls appear inside **SeedVR**, leaving the tiling checkbox off preserves ordinary SeedVR behavior, and checking it restores a still image.

## Connections and support

The tiling code makes no external network requests and requires no credentials or paid service. Installing or updating the extension contacts GitHub. SwarmUI and ComfyUI retain their own backend communication, update checks, and any model downloads you request.

Created by Juan Treminio. Report problems through [GitHub Issues](https://github.com/jtreminio/SwarmUI-SeedVRTiling/issues), including SwarmUI/ComfyUI versions, backend type, tile settings, and relevant error messages.

## License

MIT. See [LICENSE](LICENSE).
