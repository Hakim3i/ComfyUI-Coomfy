# ComfyUI-Coomfy

Asset download support for **Coomfy** Photo Lab and Video Lab. The Coomfy webapp injects ensure/download nodes into API workflows before `POST /prompt`; ComfyUI downloads missing files into the matching `models/` folders (checkpoints, LoRAs, ControlNets, upscalers, detailer detectors + SAM, diffusion models, text encoders, VAE).

- **Editor:** https://github.com/Hakim3i/Coomfy
- **This pack:** https://github.com/Hakim3i/ComfyUI-Coomfy

## Nodes

| Node | Role |
|------|------|
| **Coomfy Asset Downloader** (`ComfySpritesDownloader`) | Download every missing asset (checkpoints, LoRAs, ControlNets, upscalers, detailers, diffusion models, text encoders, VAE); output inference `ckpt_name` |
| **Coomfy Asset Download Output** (`ComfySpritesDownloadOutput`) | Terminal `OUTPUT_NODE` for the preflight download workflow |
| **Coomfy Ensure SDXL LoRAs** | Download SDXL LoRAs from `loras_json`, pass `MODEL` + `CLIP` through |
| **Coomfy Ensure LTX LoRAs** | Download LTX LoRAs from `loras_json`, pass `MODEL` through |
| **Coomfy Export Image** | Strip metadata and compress stills (WebP/JPEG/PNG) before Coomfy download |
| **Coomfy Export Audio** | Resample / downmix audio before video mux |
| **Coomfy Export Video** | Metadata-free H.264 MP4 mux for Video Lab (replaces VHS combine in Coomfy workflows) |

### Inputs (injected by Coomfy webapp)

| Input | Description |
|-------|-------------|
| `loras_json` | JSON array of LoRA rows (`filename`, `download_url`, `version_id`, …) |
| `civitai_token` | From Coomfy **Settings** (workspace `.env`) |
| `hf_token` | From Coomfy **Settings** |

Ensure nodes **do not** read `CIVITAI_TOKEN` / `HF_TOKEN` from the ComfyUI process environment.

## Install

Copy or symlink this folder into ComfyUI:

```
ComfyUI/custom_nodes/ComfyUI-Coomfy
```

Restart ComfyUI. Photo / Video Lab queueing requires these node types on the ComfyUI host.

**Video export** needs `ffmpeg` on the ComfyUI host `PATH` (or `imageio-ffmpeg` in the ComfyUI Python env).

## Scope

- **Multi-asset preflight** — `ComfySpritesDownloader` fetches checkpoints, LoRAs, ControlNets, upscalers, detailer detectors + SAM, diffusion models, text encoders, and VAE before generation.
- The legacy ensure-LoRA nodes remain for in-workflow LoRA loading.

## Docs

- [INTEGRATION.md](INTEGRATION.md) — how Coomfy webapp, ComfyUI, and this pack fit together (Photo / Video Lab flow, tokens, webapp modules)

## Files

| Path | Role |
|------|------|
| `coomfy_assets/download.py` | Civitai / HF download into every `models/` folder (multi-asset) |
| `coomfy_assets/download_utils.py` | Civitai/HF URL resolution |
| `coomfy_assets/paths.py` | Resolve ComfyUI model folders (loras, checkpoints, upscale_models, ultralytics, sams, …) |
| `nodes.py` | ComfyUI node registrations |
| `coomfy_memory.py` | `CoomfyFreeVram` node |
| `coomfy_export/` | Strip metadata + compress image/audio/video helpers |
| `__init__.py` | Pack entry (merges node registries) |
| `tests/` | Pytest for download URL + export helpers |
