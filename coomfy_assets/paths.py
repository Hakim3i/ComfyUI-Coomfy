"""ComfyUI model folder paths for LoRA files."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent


def loras_dir() -> Path:
    """Resolve ComfyUI ``models/loras`` (``folder_paths`` or side-by-side install)."""
    try:
        import folder_paths  # type: ignore[import-not-found]

        paths = folder_paths.get_folder_paths("loras")
        if paths:
            return Path(paths[0])
    except Exception:
        pass
    return PACKAGE_DIR.parent.parent / "models" / "loras"
