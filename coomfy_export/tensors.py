"""Tensor ↔ media conversions for Coomfy export nodes."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image


def image_tensor_to_pil(image) -> Image.Image:
    """Convert one ComfyUI ``IMAGE`` batch row (H×W×C float) to PIL RGB/RGBA."""
    array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    mode = "RGBA" if array.shape[-1] == 4 else "RGB"
    if mode == "RGB" and array.shape[-1] == 3:
        return Image.fromarray(array, mode="RGB")
    if mode == "RGBA":
        return Image.fromarray(array, mode="RGBA")
    return Image.fromarray(array[..., :3], mode="RGB")


def pil_to_image_tensor(pil: Image.Image):
    """Convert PIL image to one ComfyUI ``IMAGE`` row (1×H×W×C float)."""
    import torch

    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    array = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def audio_dict(waveform, sample_rate: int) -> dict[str, Any]:
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def encode_image_bytes(pil: Image.Image, *, fmt: str, quality: int) -> bytes:
    """Encode a PIL image without metadata."""
    buf = io.BytesIO()
    fmt_upper = fmt.upper()
    if fmt_upper in {"JPG", "JPEG"}:
        if pil.mode == "RGBA":
            pil = pil.convert("RGB")
        pil.save(buf, format="JPEG", quality=quality, optimize=True)
    elif fmt_upper == "WEBP":
        pil.save(buf, format="WEBP", quality=quality, method=6)
    else:
        pil.save(buf, format="PNG", optimize=True, compress_level=6)
    return buf.getvalue()


def decode_image_bytes(data: bytes):
    """Decode image bytes to one ComfyUI ``IMAGE`` row."""
    with Image.open(io.BytesIO(data)) as pil:
        pil.load()
        return pil_to_image_tensor(pil)
