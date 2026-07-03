"""LTX LoRA inspect helpers (ComfyUI-Coomfy)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coomfy_assets.ltx_lora_inspect import classify_ltx_lora_key, inspect_ltx_lora_keys


def test_comfy_classify_matches_webapp_rules():
    assert classify_ltx_lora_key("transformer_blocks.0.audio_attn1.to_k") == "aud"
    assert classify_ltx_lora_key("transformer_blocks.0.attn1.to_k") == "vid"


def test_comfy_inspect_keys_mixed():
    flags = inspect_ltx_lora_keys(
        [
            "transformer_blocks.0.audio_attn1.to_k",
            "transformer_blocks.0.audio_to_video_attn.to_v",
        ]
    )
    assert flags["has_pure_audio"] is True
    assert flags["has_a2v"] is True
    assert flags["has_video"] is False
