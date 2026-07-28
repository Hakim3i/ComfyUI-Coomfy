"""ComfyUI-Coomfy — LoRA download ensure nodes for Coomfy Photo / Video Lab."""

import os

from .coomfy_memory import (
    NODE_CLASS_MAPPINGS as _MEMORY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _MEMORY_NODE_DISPLAY_NAME_MAPPINGS,
)
from .coomfy_monitor import start_monitor as _start_monitor
from .coomfy_preview import (
    NODE_CLASS_MAPPINGS as _PREVIEW_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
)
from .coomfy_utils import (
    NODE_CLASS_MAPPINGS as _UTILS_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _UTILS_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    **_MEMORY_NODE_CLASS_MAPPINGS,
    **_UTILS_NODE_CLASS_MAPPINGS,
    **_PREVIEW_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    **_MEMORY_NODE_DISPLAY_NAME_MAPPINGS,
    **_UTILS_NODE_DISPLAY_NAME_MAPPINGS,
    **_PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
}

_LOG = "[ComfyUI-Coomfy]"


def _setup_bundled_ffmpeg() -> None:
    try:
        from .coomfy_export.ffmpeg_install import ensure_bundled_ffmpeg, ffmpeg_supports_encoder

        path = ensure_bundled_ffmpeg()
        encoder = "h264_nvenc" if ffmpeg_supports_encoder(path, "h264_nvenc") else "libx264"
        print(f"{_LOG} ffmpeg ready: {path} ({encoder})")
    except Exception as exc:
        print(f"{_LOG} ffmpeg setup failed: {exc}")


_setup_bundled_ffmpeg()
_start_monitor()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
