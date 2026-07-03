"""Inspect LTX LoRA safetensors keys for pure audio / video / cross-modal groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def classify_ltx_lora_key(key: str) -> str | None:
    """Classify a LoRA state-dict key into Multi LoRA LTX groups."""
    if key.startswith("connectors."):
        return "other"
    if "video_to_audio_attn" in key:
        return "v2a"
    if "audio_to_video_attn" in key:
        return "a2v"
    if "audio_attn" in key or "audio_ff.net" in key:
        return "aud"
    if "attn" in key or "ff.net" in key:
        return "vid"
    return "other"


def inspect_ltx_lora_keys(keys: list[str]) -> dict[str, bool]:
    groups = {classify_ltx_lora_key(k) for k in keys}
    groups.discard(None)
    return {
        "has_pure_audio": "aud" in groups,
        "has_video": "vid" in groups,
        "has_v2a": "v2a" in groups,
        "has_a2v": "a2v" in groups,
    }


def _read_safetensors_keys(path: Path) -> list[str]:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            return list(handle.keys())
    except Exception:
        pass
    try:
        import comfy.utils

        data = comfy.utils.load_torch_file(str(path), safe_load=True)
        if isinstance(data, dict):
            return [str(k) for k in data.keys()]
    except Exception:
        pass
    return []


def inspect_ltx_lora_file(path: Path) -> dict[str, Any]:
    """Return inspect metadata for one on-disk LoRA file."""
    filename = path.name
    if not path.is_file():
        return {
            "filename": filename,
            "has_pure_audio": False,
            "has_video": False,
            "has_v2a": False,
            "has_a2v": False,
            "key_count": 0,
            "error": "not_found",
        }
    try:
        keys = _read_safetensors_keys(path)
    except Exception as exc:
        return {
            "filename": filename,
            "has_pure_audio": False,
            "has_video": False,
            "has_v2a": False,
            "has_a2v": False,
            "key_count": 0,
            "error": str(exc),
        }
    if not keys:
        return {
            "filename": filename,
            "has_pure_audio": False,
            "has_video": False,
            "has_v2a": False,
            "has_a2v": False,
            "key_count": 0,
            "error": "no_keys",
        }
    flags = inspect_ltx_lora_keys(keys)
    return {
        "filename": filename,
        **flags,
        "key_count": len(keys),
        "error": None,
    }


def inspect_ltx_lora_filenames(
    filenames: list[str],
    *,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Inspect LoRA files under ``models/loras`` (or ``base_dir``)."""
    if base_dir is None:
        from .paths import loras_dir

        base_dir = loras_dir()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in filenames:
        name = Path((raw or "").strip()).name
        if not name or name in seen:
            continue
        seen.add(name)
        rows.append(inspect_ltx_lora_file(base_dir / name))
    return rows
