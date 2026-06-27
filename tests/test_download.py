"""Tests for LoRA download URL resolution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coomfy_assets.download_utils import civitai_authenticated_url, download_candidates


def test_download_candidates_primary_and_version_id():
    info = {
        "download_url": "https://example.com/a.safetensors",
        "version_id": 99,
    }
    urls = download_candidates(info)
    assert urls[0] == "https://example.com/a.safetensors"
    assert any("civitai" in u and "99" in u for u in urls)


def test_civitai_authenticated_url_appends_token():
    url = "https://civitai.red/api/download/models/123"
    out = civitai_authenticated_url(url, "secret")
    assert "token=secret" in out
    assert out.startswith(url.split("?")[0])


def test_civitai_authenticated_url_skips_when_token_present():
    url = "https://civitai.red/api/download/models/123?token=existing"
    assert civitai_authenticated_url(url, "secret") == url


def test_civitai_authenticated_url_noop_without_token():
    url = "https://civitai.red/api/download/models/123"
    assert civitai_authenticated_url(url, "") == url
