"""Tests for Coomfy export strip/compress helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coomfy_export.audio import export_audio
from coomfy_export.image import export_images
from coomfy_export.tensors import audio_dict, encode_image_bytes, image_tensor_to_pil


def _rgb_batch() -> torch.Tensor:
    red = torch.zeros((1, 8, 8, 3))
    red[..., 0] = 1.0
    return red


def test_export_images_passthrough_when_disabled():
    images = _rgb_batch()
    out = export_images(images, enabled=False, fmt="webp", quality=85)
    assert torch.equal(out, images)


def test_export_images_reencodes_without_png_text_chunks():
    images = _rgb_batch()
    out = export_images(images, enabled=True, fmt="webp", quality=80)
    pil = image_tensor_to_pil(out[0])
    data = encode_image_bytes(pil, fmt="webp", quality=80)
    assert b"comfy" not in data.lower()
    assert out.shape == images.shape


def test_export_audio_downmix_and_resample():
    waveform = torch.zeros((1, 2, 48000))
    waveform[0, 0, :] = 1.0
    audio = audio_dict(waveform, 48000)
    out = export_audio(audio, enabled=True, target_sample_rate=22050, mono=True)
    assert out["sample_rate"] == 22050
    assert out["waveform"].shape[1] == 1
    assert out["waveform"].shape[-1] == 22050


def test_export_audio_passthrough_when_disabled():
    waveform = torch.randn(1, 2, 1000)
    audio = audio_dict(waveform, 44100)
    out = export_audio(audio, enabled=False, target_sample_rate=22050, mono=True)
    assert torch.equal(out["waveform"], waveform)
    assert out["sample_rate"] == 44100
