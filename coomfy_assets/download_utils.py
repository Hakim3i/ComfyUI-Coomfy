"""Shared Civitai / HuggingFace download URL resolution."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def is_civitai_url(url: str) -> bool:
    low = (url or "").lower()
    return "civitai.com" in low or "civitai.red" in low


def is_huggingface_url(url: str) -> bool:
    low = (url or "").lower()
    return "huggingface.co" in low or "hf.co" in low


def civitai_authenticated_url(url: str, token: str) -> str:
    """Append Civitai API token as ``?token=`` / ``&token=`` (required for downloads)."""
    raw_url = (url or "").strip()
    token = (token or "").strip()
    if not raw_url or not token or not is_civitai_url(raw_url):
        return raw_url
    parsed = urlparse(raw_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query.get("token"):
        return raw_url
    query["token"] = token
    return urlunparse(parsed._replace(query=urlencode(query)))


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
