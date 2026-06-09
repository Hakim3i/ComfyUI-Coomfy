"""Tests for LoRA download URL resolution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coomfy_assets.download_utils import download_candidates


def test_download_candidates_primary_and_version_id():
    info = {
        "download_url": "https://example.com/a.safetensors",
        "version_id": 99,
    }
    urls = download_candidates(info)
    assert urls[0] == "https://example.com/a.safetensors"
    assert any("civitai" in u and "99" in u for u in urls)
