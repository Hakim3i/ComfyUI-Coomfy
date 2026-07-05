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
| **Coomfy Ensure LTX LoRAs** | Video Studio — between diffusion model and **Multi LoRA Loader** (main `lora` node; LTX mode with per-layer Vid/Aud/V2A/A2V) |
| **Coomfy Export Image** | Photo Studio node **`132`** — strip metadata + compress before PreviewImage `131` |
| **Coomfy Export Audio** | Video Studio node **`319`** — audio prep before mux |
| **Coomfy Export Video** | Video Studio node **`59`** — metadata-free H.264 mux (Coomfy download target) |
| **CoomfyAssetDownloader** / **CoomfyAssetDownloadOutput** | Photo Lab asset preflight via [`download_workflow.py`](../webapp/comfyui/download_workflow.py) |

Before `POST /prompt`, Coomfy ([`../webapp/comfyui/inject_assets.py`](../webapp/comfyui/inject_assets.py)) patches:

- `loras_json` — from `composer.build()`
- `civitai_token` / `hf_token` — from [`load_api_keys()`](../webapp/env_settings.py)
- `enabled` on export nodes — from `request.export_compress` (defaults **on**; UI toggle planned)

**Multi-asset preflight:** `CoomfyAssetDownloader` downloads missing checkpoints, LoRAs, ControlNets, upscalers, detailer detectors + SAM, diffusion models, text encoders, and VAE before generation. Ensure-LoRA nodes remain for in-workflow LoRA loading.

### Install

1. Copy or symlink this folder to `ComfyUI/custom_nodes/ComfyUI-Coomfy`.
2. Restart ComfyUI.
3. Set Civitai / HF tokens in Coomfy **Settings**.

### Prompt Relay (Video Lab timed phases)

Video Lab toggle **Prompt Relay (timed phases)** (`ltx_include_time_brackets`) switches temporal prompting:

| Toggle | Behavior |
|--------|----------|
| **ON** | Kijai **`PromptRelayEncode`** — `global_prompt` + `local_prompts` (`beat1 \| beat2 \| …`) + `segment_lengths` (pixel frames per phase). Positive conditioning bypasses the single `CLIPTextEncode` caption. |
| **OFF** | Legacy path — one `CLIPTextEncode` caption; optional `[0-5 sec]` text brackets in the caption string. |

**Install on the ComfyUI host** (not bundled with Coomfy):

```bash
cd ComfyUI/custom_nodes/ComfyUI-Coomfy
bash install_comfyui_custom_nodes.sh              # full stack (Vast.ai / RunPod)
```

Restart ComfyUI after install. Requires an up-to-date **ComfyUI-LTXVideo** stack and **[ComfyUI-MultiLoRALoader](https://github.com/phazei/ComfyUI-MultiLoRALoader)** for Video Studio generation (replaces rgthree Power LoRA on the main LTX `lora` node; distilled passes still use Power LoRA). Photo Lab detailers need **ComfyUI-Impact-Pack** plus **ComfyUI-Impact-Subpack** (`UltralyticsDetectorProvider`); both are installed by `install_comfyui_custom_nodes.sh`. If a node is missing, queued jobs fail at execution time with an unknown node type error. For the full node list on Vast.ai / RunPod, run `install_comfyui_custom_nodes.sh` in this folder (see parent [README.md](../README.md#remote-comfyui-vastai--runpod)).

Syntax matches the [PromptRelay README](https://github.com/kijai/ComfyUI-PromptRelay): persistent `global_prompt`, pipe-separated local beats, comma-separated frame counts aligned with Coomfy phase timing (`phase_frame_ranges` / timed LoRA schedules).

### Arc toggles (Video Lab)

- **Framing** uses the act view's **natural language** once in the opener (`framing` segment). No per-phase camera cues.
- Video Lab checkboxes omit parts of the arc at preview/generate time without changing stored act data:
  - `ltx_include_audio_tiers` — `Audio:` tier hints
  - `ltx_include_voice_lines` — dialogue beats
- **Personality** dropdown (`personality`, default `character`) overrides which voice-line pool is resolved; otherwise the rolled character's personality is used.

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
