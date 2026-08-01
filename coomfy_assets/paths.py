"""ComfyUI model folder paths for downloadable assets."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _resolve_dir(folder_key: str, fallback: str) -> Path:
    """Resolve a ComfyUI ``models/<folder>`` dir (``folder_paths`` or side-by-side)."""
    try:
        import folder_paths  # type: ignore[import-not-found]

        paths = folder_paths.get_folder_paths(folder_key)
        if paths:
            return Path(paths[0])
    except Exception:
        pass
    return PACKAGE_DIR.parent.parent / "models" / fallback


def loras_dir() -> Path:
    """Resolve ComfyUI ``models/loras``."""
    return _resolve_dir("loras", "loras")


def controlnet_dir() -> Path:
    """Resolve ComfyUI ``models/controlnet``."""
    return _resolve_dir("controlnet", "controlnet")


def checkpoints_dir() -> Path:
    """Resolve ComfyUI ``models/checkpoints``."""
    return _resolve_dir("checkpoints", "checkpoints")


def upscale_models_dir() -> Path:
    """Resolve ComfyUI ``models/upscale_models``."""
    return _resolve_dir("upscale_models", "upscale_models")


def latent_upscale_models_dir() -> Path:
    """Resolve ComfyUI ``models/latent_upscale_models`` (LTX spatial upscaler)."""
    return _resolve_dir("latent_upscale_models", "latent_upscale_models")


def ultralytics_dir() -> Path:
    """Resolve ComfyUI ``models/ultralytics`` (Impact Pack detailer detectors)."""
    return _resolve_dir("ultralytics", "ultralytics")


def sams_dir() -> Path:
    """Resolve ComfyUI ``models/sams`` (Impact Pack SAM weights)."""
    return _resolve_dir("sams", "sams")


def diffusion_models_dir() -> Path:
    """Resolve ComfyUI ``models/diffusion_models``."""
    return _resolve_dir("diffusion_models", "diffusion_models")


def text_encoders_dir() -> Path:
    """Resolve ComfyUI ``models/text_encoders``."""
    return _resolve_dir("text_encoders", "text_encoders")


def vae_dir() -> Path:
    """Resolve ComfyUI ``models/vae``."""
    return _resolve_dir("vae", "vae")


def vae_approx_dir() -> Path:
    """Resolve ComfyUI ``models/vae_approx`` (TAESD preview autoencoders)."""
    return _resolve_dir("vae_approx", "vae_approx")
