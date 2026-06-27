# Coomfy ↔ ComfyUI integration architecture

How the **Coomfy** webapp relates to a **local ComfyUI** instance and this **ComfyUI-Coomfy** custom node pack. Use this when designing or debugging Photo / Video Lab generation.

---

## Three processes

| Component | Default URL | Role |
|-----------|-------------|------|
| **Coomfy** (parent repo) | `http://127.0.0.1:8765` | Dataset editor, `POST /api/build`, `GET /api/dropdowns`, Photo / Video Lab UI |
| **ComfyUI** | configurable (`COMFYUI_BASE_URL`) | Executes node graphs; `/prompt`, `/history`, `/view`, `/ws` |
| **ComfyUI-Coomfy** (this pack) | `ComfyUI/custom_nodes/` | Asset download + LoRA ensure + export nodes; downloads on GPU host; tokens injected by Coomfy webapp |

They are **separate processes**. Coomfy does not need to live inside `custom_nodes/`.

---

## Photo Lab backend data flow

```
Photo Lab (Alpine.js)
    │
    ├─► POST /api/build          (Coomfy) — compose prompts + inference
    │
    └─► POST /api/photo-lab/generate   (Coomfy)
            │
            ├─ Load workflow template (Photo Studio.json)
            ├─ Patch nodes from build (checkpoint, LoRAs, latent, prompts, KSampler, batch_size)
            ├─ Patch node 128 (Coomfy Ensure SDXL LoRAs): loras_json + Settings tokens
            ├─ Compose detailers from Detailers.json when enabled
            ├─ WS /ws?clientId=…         (ComfyUI) — before queue
            ├─ POST /prompt              (ComfyUI)
            └─ On complete: download Save node 76 → outputs/photos/
```

Coomfy is the **prompt and metadata authority**; ComfyUI is the **compute engine**.

### Configuration

| Variable | Consumer | Purpose |
|----------|----------|---------|
| `COMFYUI_BASE_URL` | Coomfy server | ComfyUI base URL (Settings / workspace `.env`) |
| `CIVITAI_TOKEN` / `HF_TOKEN` | Coomfy webapp → workflow injection | Settings; sent on ensure node inputs (not ComfyUI host env) |

---

## LoRA ensure nodes (this pack)

| Node | Photo / Video wiring |
|------|----------------------|
| **Coomfy Ensure SDXL LoRAs** | Photo Studio node **`128`** (passthrough to Power Lora `101`) |
| **Coomfy Ensure LTX LoRAs** | Video Studio — between diffusion `257` and Power Lora `314` (or baked-in node id in your export) |
| **Coomfy Export Image** | Photo Studio node **`132`** — strip metadata + compress before PreviewImage `131` |
| **Coomfy Export Audio** | Video Studio node **`319`** — audio prep before mux |
| **Coomfy Export Video** | Video Studio node **`59`** — metadata-free H.264 mux (Coomfy download target) |
| **ComfySpritesDownloader** / **ComfySpritesDownloadOutput** | Photo Lab asset preflight via [`download_workflow.py`](../webapp/comfyui/download_workflow.py) |

Before `POST /prompt`, Coomfy ([`../webapp/comfyui/inject_assets.py`](../webapp/comfyui/inject_assets.py)) patches:

- `loras_json` — from `composer.build()`
- `civitai_token` / `hf_token` — from [`load_api_keys()`](../webapp/env_settings.py)
- `enabled` on export nodes — from `request.export_compress` (defaults **on**; UI toggle planned)

**Multi-asset preflight:** `ComfySpritesDownloader` downloads missing checkpoints, LoRAs, ControlNets, upscalers, detailer detectors + SAM, diffusion models, text encoders, and VAE before generation. Legacy ensure-LoRA nodes remain for in-workflow LoRA loading.

### Install

1. Copy or symlink this folder to `ComfyUI/custom_nodes/ComfyUI-Coomfy`.
2. Restart ComfyUI.
3. Set Civitai / HF tokens in Coomfy **Settings**.

---

## Webapp modules

| Module | Role |
|--------|------|
| [`../webapp/comfyui/workflow.py`](../webapp/comfyui/workflow.py) | Photo Studio load / patch / node id map |
| [`../webapp/comfyui/asset_manifest.py`](../webapp/comfyui/asset_manifest.py) | Asset manifest rows for ensure/download nodes |
| [`../webapp/comfyui/inject_assets.py`](../webapp/comfyui/inject_assets.py) | Patch ensure node inputs + tokens |
| [`../webapp/comfyui/generate.py`](../webapp/comfyui/generate.py) | Queue Photo / Video Lab jobs |
| [`../webapp/comfyui/workflows/`](../webapp/comfyui/workflows/) | API-format JSON templates + patch docs |

Workflow patch tables: [`Photo Studio.md`](../webapp/comfyui/workflows/Photo%20Studio.md), [`Video Studio.md`](../webapp/comfyui/workflows/Video%20Studio.md).

---

## Related docs

- [README.md](README.md) — install and node inputs (this pack)
- [../README.md](../README.md) — run Coomfy webapp
- [../.cursor/comfyui-engine.mdc](../.cursor/comfyui-engine.mdc) — WS progress and live previews
