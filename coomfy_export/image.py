"""Strip metadata and compress ComfyUI image batches."""

from __future__ import annotations

import torch

from .tensors import decode_image_bytes, encode_image_bytes, image_tensor_to_pil


def export_images(images, *, enabled: bool, fmt: str, quality: int):
    """Return an ``IMAGE`` batch; re-encode when ``enabled``."""
    if not enabled:
        return images

    fmt_key = (fmt or "webp").strip().lower()
    q = max(1, min(100, int(quality)))
    rows = []
    for row in images:
        pil = image_tensor_to_pil(row)
        data = encode_image_bytes(pil, fmt=fmt_key, quality=q)
        rows.append(decode_image_bytes(data)[0])
    return torch.stack(rows, dim=0)
