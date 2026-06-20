"""VRAM cleanup passthrough for multi-pass Video Lab workflows."""

from __future__ import annotations

_LOG = "[Coomfy VRAM]"


class CoomfyFreeVram:
    """Unload loaded models and clear GPU cache; passthrough latent unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When off, pass latent through without unloading models.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "free"
    CATEGORY = "Coomfy"

    def free(self, latent, enabled: bool):
        if enabled:
            import comfy.model_management as mm

            mm.unload_all_models()
            mm.soft_empty_cache()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"{_LOG} models unloaded between passes")
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "CoomfyFreeVram": CoomfyFreeVram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoomfyFreeVram": "Coomfy Free VRAM",
}
