"""Shared Civitai / HuggingFace download URL resolution."""

from __future__ import annotations

from typing import Any


def download_candidates(info: dict[str, Any]) -> list[str]:
    """Primary ``download_url`` first, then optional Civitai fallbacks."""
    urls: list[str] = []
    for key in ("download_url", "hf_download_url", "api_download_url", "download_fallback_url"):
        raw = (info.get(key) or "").strip() if isinstance(info.get(key), str) else ""
        if raw and raw not in urls:
            urls.append(raw)
    version_id = info.get("version_id")
    if version_id:
        for base in (
            "https://civitai.red/api/download/models/",
            "https://civitai.com/api/download/models/",
        ):
            civ = f"{base}{version_id}"
            if civ not in urls:
                urls.append(civ)
    return urls
