"""Tests for Coomfy asset download progress messages."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coomfy_assets.progress import format_asset_download_message


def test_format_asset_download_message_includes_kind_name_and_pct():
    msg = format_asset_download_message(
        {
            "asset_kind": "lora",
            "filename": "MatureFemaleSliderAnima.safetensors",
            "current": 3,
            "total": 16,
            "file_pct": 42.5,
        }
    )
    assert "lora 3/16" in msg
    assert "MatureFemaleSliderAnima.safetensors" in msg
    assert "43%" in msg or "42%" in msg
