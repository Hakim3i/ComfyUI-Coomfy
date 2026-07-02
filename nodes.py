"""ComfyUI-Coomfy — LoRA ensure + export nodes for Photo / Video Lab workflows."""

from __future__ import annotations

from .coomfy_assets.download import ensure_all_assets, ensure_loras_from_json
from .coomfy_export import export_audio, export_images, mux_video

_LOG = "[Coomfy ensure]"


class CoomfyExportImage:
    """Strip metadata and compress images before Coomfy downloads them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When off, pass frames through unchanged.",
                    },
                ),
                "format": (
                    ["webp", "jpeg", "png"],
                    {
                        "default": "webp",
                        "tooltip": "Output encoding when enabled (metadata is always stripped).",
                    },
                ),
                "quality": (
                    "INT",
                    {
                        "default": 85,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Lossy quality for webp/jpeg.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "export"
    CATEGORY = "Coomfy/Export"

    def export(self, images, enabled: bool, format: str, quality: int):
        return (export_images(images, enabled=enabled, fmt=format, quality=quality),)


class CoomfyExportAudio:
    """Prepare generated audio for lean, metadata-free video muxing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When off, pass audio through unchanged.",
                    },
                ),
                "target_sample_rate": (
                    "INT",
                    {
                        "default": 44100,
                        "min": 8000,
                        "max": 48000,
                        "step": 1000,
                        "tooltip": "Resample rate when enabled.",
                    },
                ),
                "mono": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Downmix to mono when enabled.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "export"
    CATEGORY = "Coomfy/Export"

    def export(
        self,
        audio,
        enabled: bool,
        target_sample_rate: int,
        mono: bool,
    ):
        return (
            export_audio(
                audio,
                enabled=enabled,
                target_sample_rate=target_sample_rate,
                mono=mono,
            ),
        )


class CoomfyExportVideo:
    """Mux frames + audio into a metadata-free H.264 MP4 for Coomfy download."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1},
                ),
                "filename_prefix": ("STRING", {"default": "Coomfy/Video"}),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When off, keep higher quality (lower CRF / higher audio bitrate).",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "H.264 CRF when compression is enabled.",
                    },
                ),
                "audio_bitrate_kbps": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 320,
                        "step": 8,
                        "tooltip": "AAC bitrate when compression is enabled.",
                    },
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "Coomfy/Export"

    def export(
        self,
        images,
        frame_rate: float,
        filename_prefix: str,
        enabled: bool,
        crf: int,
        audio_bitrate_kbps: int,
        audio=None,
    ):
        import folder_paths

        output_dir = folder_paths.get_temp_directory()
        entry = mux_video(
            images,
            frame_rate=frame_rate,
            audio=audio,
            filename_prefix=filename_prefix,
            output_dir=output_dir,
            enabled=enabled,
            crf=crf,
            audio_bitrate_kbps=audio_bitrate_kbps,
        )
        return {
            "ui": {
                "gifs": [
                    {
                        "filename": entry["filename"],
                        "subfolder": entry["subfolder"],
                        "type": entry["type"],
                        "format": entry["format"],
                        "frame_rate": entry["frame_rate"],
                    }
                ]
            }
        }


class CoomfyEnsureLoras:
    """Download LoRAs from ``loras_json`` (SDXL, LTX, etc.), passthrough model/clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "loras_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "tooltip": "JSON array of LoRA entries from Coomfy /api/build.",
                    },
                ),
                "civitai_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by Coomfy webapp from Settings (not read from env).",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by Coomfy webapp from Settings (not read from env).",
                    },
                ),
            },
            "optional": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": "Optional CLIP passthrough (Photo Studio / LTX dual-CLIP).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "ensure"
    CATEGORY = "Coomfy"

    def ensure(
        self,
        model,
        loras_json: str,
        civitai_token: str,
        hf_token: str,
        clip=None,
    ):
        applied = ensure_loras_from_json(
            loras_json,
            civitai_token=civitai_token or "",
            hf_token=hf_token or "",
        )
        if applied:
            print(f"{_LOG} LoRAs ready: {', '.join(applied)}")
        return (model, clip)


class CoomfyPreflightLoras:
    """Download LoRAs only — terminal output node for Coomfy preflight workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loras_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "tooltip": "JSON array of LoRA entries from Coomfy /api/build.",
                    },
                ),
                "civitai_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by Coomfy webapp from Settings (not read from env).",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by Coomfy webapp from Settings (not read from env).",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preflight"
    OUTPUT_NODE = True
    CATEGORY = "Coomfy"

    def preflight(self, loras_json: str, civitai_token: str, hf_token: str):
        applied = ensure_loras_from_json(
            loras_json,
            civitai_token=civitai_token or "",
            hf_token=hf_token or "",
        )
        if applied:
            print(f"{_LOG} preflight LoRAs ready: {', '.join(applied)}")
        return {"ui": {"text": [f"LoRAs ready: {', '.join(applied) or 'none'}"]}}


class CoomfyAssetDownloader:
    """Download every missing asset (checkpoints, LoRAs, ControlNets, upscalers,
    detailers, diffusion models, text encoders, VAE); output inference ``ckpt_name``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Inference checkpoint filename for the SDXL Loader.",
                    },
                ),
                "checkpoints_json": ("STRING", {"multiline": True, "default": "[]"}),
                "loras_json": ("STRING", {"multiline": True, "default": "[]"}),
                "controlnets_json": ("STRING", {"multiline": True, "default": "[]"}),
                "upscalers_json": ("STRING", {"multiline": True, "default": "[]"}),
                "detailers_json": ("STRING", {"multiline": True, "default": "[]"}),
                "diffusion_models_json": ("STRING", {"multiline": True, "default": "[]"}),
                "text_encoders_json": ("STRING", {"multiline": True, "default": "[]"}),
                "vae_json": ("STRING", {"multiline": True, "default": "[]"}),
                "civitai_token": ("STRING", {"default": ""}),
                "hf_token": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ckpt_name",)
    FUNCTION = "download"
    CATEGORY = "Coomfy"

    def download(
        self,
        ckpt_name: str,
        checkpoints_json: str,
        loras_json: str,
        controlnets_json: str,
        upscalers_json: str,
        detailers_json: str,
        diffusion_models_json: str,
        text_encoders_json: str,
        vae_json: str,
        civitai_token: str,
        hf_token: str,
    ):
        try:
            from comfy.utils import ProgressBar
        except ImportError:
            ProgressBar = None  # type: ignore[misc, assignment]

        from .coomfy_assets.download import count_pending_assets

        pending = count_pending_assets(
            checkpoints_json=checkpoints_json,
            loras_json=loras_json,
            controlnets_json=controlnets_json,
            upscalers_json=upscalers_json,
            detailers_json=detailers_json,
            diffusion_models_json=diffusion_models_json,
            text_encoders_json=text_encoders_json,
            vae_json=vae_json,
        )
        pbar = ProgressBar(pending) if ProgressBar is not None and pending > 0 else None

        def _on_asset_progress(frac: float) -> None:
            if pbar is not None:
                pbar.update_absolute(int(round(frac * pending)), pending)

        applied = ensure_all_assets(
            checkpoints_json=checkpoints_json,
            loras_json=loras_json,
            controlnets_json=controlnets_json,
            upscalers_json=upscalers_json,
            detailers_json=detailers_json,
            diffusion_models_json=diffusion_models_json,
            text_encoders_json=text_encoders_json,
            vae_json=vae_json,
            civitai_token=civitai_token or "",
            hf_token=hf_token or "",
            on_progress=_on_asset_progress if pbar is not None else None,
        )
        if pbar is not None:
            pbar.update_absolute(pending, pending)
        name = (ckpt_name or "").strip()
        if not name:
            for key in (
                "diffusion_models",
                "text_encoders",
                "vae",
                "loras",
                "checkpoints",
            ):
                rows = applied.get(key) or []
                if rows:
                    name = str(rows[0]).strip()
                    break
            if not name:
                name = "assets-ready"
        parts: list[str] = []
        for key in (
            "checkpoints",
            "loras",
            "controlnets",
            "upscalers",
            "detailers",
            "diffusion_models",
            "text_encoders",
            "vae",
        ):
            rows = applied.get(key) or []
            if rows:
                parts.append(f"{key}={', '.join(rows)}")
        if parts:
            print(f"{_LOG} downloaded: {'; '.join(parts)}")
        print(f"{_LOG} assets ready; inference checkpoint: {name}")
        return (name,)


class CoomfyAssetDownloadOutput:
    """Terminal output node for the asset-download workflow (satisfies ``OUTPUT_NODE``)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": (
                    "STRING",
                    {"default": "", "tooltip": "Wire from Coomfy Asset Downloader output."},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "output"
    OUTPUT_NODE = True
    CATEGORY = "Coomfy"

    def output(self, message: str):
        text = (message or "").strip() or "ok"
        print(f"{_LOG} download workflow complete: {text}")
        return {"ui": {"text": [text]}}


NODE_CLASS_MAPPINGS = {
    "CoomfyAssetDownloader": CoomfyAssetDownloader,
    "CoomfyAssetDownloadOutput": CoomfyAssetDownloadOutput,
    "CoomfyEnsureLoras": CoomfyEnsureLoras,
    "CoomfyEnsureSDXLLoras": CoomfyEnsureLoras,
    "CoomfyEnsureLTXLoras": CoomfyEnsureLoras,
    "CoomfyPreflightLoras": CoomfyPreflightLoras,
    "CoomfyExportImage": CoomfyExportImage,
    "CoomfyExportAudio": CoomfyExportAudio,
    "CoomfyExportVideo": CoomfyExportVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoomfyAssetDownloader": "Coomfy Asset Downloader",
    "CoomfyAssetDownloadOutput": "Coomfy Asset Download Output",
    "CoomfyEnsureLoras": "Coomfy Ensure LoRAs",
    "CoomfyEnsureSDXLLoras": "Coomfy Ensure LoRAs",
    "CoomfyEnsureLTXLoras": "Coomfy Ensure LoRAs",
    "CoomfyPreflightLoras": "Coomfy Preflight LoRAs",
    "CoomfyExportImage": "Coomfy Export Image",
    "CoomfyExportAudio": "Coomfy Export Audio",
    "CoomfyExportVideo": "Coomfy Export Video",
}
