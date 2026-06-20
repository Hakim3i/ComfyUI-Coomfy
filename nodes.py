"""ComfyUI-Coomfy — LoRA ensure + export nodes for Photo / Video Lab workflows."""

from __future__ import annotations

from .coomfy_assets.download import ensure_loras_from_json
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
    """Download LoRAs from ``loras_json`` (SDXL, LTX, ZIT, etc.), passthrough model/clip."""

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


NODE_CLASS_MAPPINGS = {
    "CoomfyEnsureLoras": CoomfyEnsureLoras,
    "CoomfyEnsureSDXLLoras": CoomfyEnsureLoras,
    "CoomfyEnsureLTXLoras": CoomfyEnsureLoras,
    "CoomfyPreflightLoras": CoomfyPreflightLoras,
    "CoomfyExportImage": CoomfyExportImage,
    "CoomfyExportAudio": CoomfyExportAudio,
    "CoomfyExportVideo": CoomfyExportVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoomfyEnsureLoras": "Coomfy Ensure LoRAs",
    "CoomfyEnsureSDXLLoras": "Coomfy Ensure LoRAs",
    "CoomfyEnsureLTXLoras": "Coomfy Ensure LoRAs",
    "CoomfyPreflightLoras": "Coomfy Preflight LoRAs",
    "CoomfyExportImage": "Coomfy Export Image",
    "CoomfyExportAudio": "Coomfy Export Audio",
    "CoomfyExportVideo": "Coomfy Export Video",
}
