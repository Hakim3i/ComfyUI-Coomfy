# ComfyUI-Coomfy

Asset download support for **Coomfy** Photo Lab and Video Lab. The Coomfy webapp injects ensure/download nodes into API workflows before `POST /prompt`; ComfyUI downloads missing files into the matching `models/` folders (checkpoints, LoRAs, ControlNets, upscalers, detailer detectors + SAM, diffusion models, text encoders, VAE).

- **Editor:** https://github.com/Hakim3i/Coomfy
- **This pack:** https://github.com/Hakim3i/ComfyUI-Coomfy

## Nodes

| Node | Role |
|------|------|
| **Coomfy Asset Downloader** (`CoomfyAssetDownloader`) | Download every missing asset (checkpoints, LoRAs, ControlNets, upscalers, detailers, diffusion models, text encoders, VAE); output inference `ckpt_name` |
| **Coomfy Asset Download Output** (`CoomfyAssetDownloadOutput`) | Terminal `OUTPUT_NODE` for the preflight download workflow |
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

Copy or clone this folder into ComfyUI:

```
ComfyUI/custom_nodes/ComfyUI-Coomfy
```

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Hakim3i/ComfyUI-Coomfy.git
cd ComfyUI-Coomfy
bash install_comfyui_custom_nodes.sh   # clones/updates all other required custom nodes
```

ComfyUI root is inferred as `../..`. Override with an argument or `COMFYUI_PATH` if needed. Re-run the script to update packs.

**Use ComfyUI's Python for pip installs** — the script installs each pack's `requirements.txt`. Point it at the venv or embedded interpreter ComfyUI actually runs:

```bash
# Linux / RunPod venv
COMFYUI_PYTHON=/workspace/ComfyUI/venv/bin/python bash install_comfyui_custom_nodes.sh

# Windows portable (from custom_nodes/ComfyUI-Coomfy)
COMFYUI_PYTHON=../../python_embeded/python.exe bash install_comfyui_custom_nodes.sh
```

Or activate the ComfyUI venv first; the script then uses that environment's `python3`.

The installer also pulls **ComfyUI-Crystools** (CPU/GPU monitor for Coomfy lab status badges) and patches **ComfyUI-LTXVideo** for `kornia>=0.8` (`pad` import).

Restart ComfyUI. Photo / Video Lab queueing requires these node types on the ComfyUI host.

**Video export:** on first load ComfyUI-Coomfy installs `imageio-ffmpeg` (if needed) and copies ffmpeg into `bin/`. `Coomfy Export Video` uses that bundled binary — not WinGet/PATH symlinks.

## Scope

- **Multi-asset preflight** — `CoomfyAssetDownloader` fetches checkpoints, LoRAs, ControlNets, upscalers, detailer detectors + SAM, diffusion models, text encoders, and VAE before generation.
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
| `coomfy_export/ffmpeg_install.py` | Bundle ffmpeg into `bin/` when ComfyUI loads this pack |
| `requirements.txt` | `imageio-ffmpeg` for bundled ffmpeg |
| `install_comfyui_custom_nodes.sh` | Clone/update all ComfyUI packs Photo + Video Lab need |
| `__init__.py` | Pack entry (merges node registries) |
| `tests/` | Pytest for download URL + export helpers |
