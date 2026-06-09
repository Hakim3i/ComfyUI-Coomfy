"""Prepare ComfyUI ``AUDIO`` tensors for lean muxed output."""

from __future__ import annotations

import torch

from .tensors import audio_dict


def _resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr or waveform.shape[-1] == 0:
        return waveform
    duration = waveform.shape[-1] / float(orig_sr)
    out_len = max(1, int(round(duration * target_sr)))
    # Linear resample per channel (batch × channels × samples).
    flat = waveform.reshape(-1, waveform.shape[-1])
    rows = []
    for ch in flat:
        rows.append(torch.nn.functional.interpolate(
            ch[None, None, :], size=out_len, mode="linear", align_corners=True
        )[0, 0])
    out = torch.stack(rows, dim=0).reshape(waveform.shape[:-1] + (out_len,))
    return out


def export_audio(
    audio: dict,
    *,
    enabled: bool,
    target_sample_rate: int,
    mono: bool,
):
    """Return a cleaned ``AUDIO`` dict; downsample when ``enabled``."""
    if audio is None:
        return None
    if not enabled:
        return audio

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    target_sr = max(8000, int(target_sample_rate))

    if mono and waveform.shape[1] > 1:
        waveform = waveform.mean(dim=1, keepdim=True)

    waveform = _resample_waveform(waveform, sample_rate, target_sr)
    return audio_dict(waveform, target_sr)
