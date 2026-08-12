"""Copy missing model files from peer SMB shares into local ComfyUI models/."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .paths import (
    checkpoints_dir,
    controlnet_dir,
    diffusion_models_dir,
    latent_upscale_models_dir,
    loras_dir,
    sams_dir,
    text_encoders_dir,
    ultralytics_dir,
    upscale_models_dir,
    vae_dir,
)

_BUCKET_DIRS = {
    "loras": loras_dir,
    "checkpoints": checkpoints_dir,
    "controlnet": controlnet_dir,
    "upscale_models": upscale_models_dir,
    "latent_upscale_models": latent_upscale_models_dir,
    "ultralytics": ultralytics_dir,
    "sams": sams_dir,
    "diffusion_models": diffusion_models_dir,
    "text_encoders": text_encoders_dir,
    "vae": vae_dir,
}


def _parse_peers(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, str]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        unc = str(row.get("unc") or row.get("models_share") or "").strip()
        if not unc:
            continue
        out.append(
            {
                "unc": unc.rstrip("\\/"),
                "user": str(row.get("user") or row.get("smb_user") or "").strip(),
                "password": str(row.get("password") or row.get("smb_password") or ""),
            }
        )
    return out


def _parse_files(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, str]] = []
    for row in data:
        if isinstance(row, str):
            name = row.strip()
            if name:
                out.append({"bucket": "loras", "filename": Path(name).name})
            continue
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename") or "").strip()
        if not filename:
            continue
        bucket = str(row.get("bucket") or row.get("folder") or "loras").strip() or "loras"
        out.append({"bucket": bucket, "filename": Path(filename).name})
    return out


def _ensure_smb(unc: str, user: str, password: str) -> None:
    if os.name != "nt" or not user:
        return
    # Map the share root for the current process (ComfyUI service account).
    root = unc
    # \\host\share\rest → \\host\share
    parts = unc.replace("/", "\\").strip("\\").split("\\")
    if len(parts) >= 2:
        root = f"\\\\{parts[0]}\\{parts[1]}"
    cmd = ["net", "use", root, f"/user:{user}", password]
    subprocess.run(cmd, check=False, capture_output=True, text=True)


def copy_peer_models(
    *,
    files_json: str,
    peers_json: str,
    prompt_id: str | None = None,
) -> dict[str, Any]:
    """Copy missing files from peer UNC models trees into local folder_paths."""
    files = _parse_files(files_json)
    peers = _parse_peers(peers_json)
    copied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    if not files:
        return {"copied": copied, "skipped": skipped, "missing": missing}

    for peer in peers:
        _ensure_smb(peer["unc"], peer["user"], peer["password"])

    for item in files:
        bucket = item["bucket"]
        filename = item["filename"]
        resolver = _BUCKET_DIRS.get(bucket)
        if resolver is None:
            missing.append(f"{bucket}/{filename}")
            continue
        dest_dir = resolver()
        dest = dest_dir / filename
        if dest.is_file():
            skipped.append(filename)
            continue
        found = False
        for peer in peers:
            # Prefer models/<bucket>/<file> under the share; also try bare <file>.
            candidates = [
                Path(peer["unc"]) / bucket / filename,
                Path(peer["unc"]) / filename,
            ]
            for src in candidates:
                try:
                    if not src.is_file():
                        continue
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + f".{uuid.uuid4().hex}.part")
                    shutil.copy2(src, tmp)
                    tmp.replace(dest)
                    copied.append(filename)
                    found = True
                    break
                except OSError:
                    continue
            if found:
                break
        if not found:
            missing.append(f"{bucket}/{filename}")

    # Optional progress event for Coomfy UI (best-effort).
    try:
        from .progress import send_asset_download_progress

        send_asset_download_progress(
            {
                "phase": "peer_copy",
                "copied": len(copied),
                "skipped": len(skipped),
                "missing": len(missing),
            },
            prompt_id=prompt_id,
        )
    except Exception:
        pass

    return {"copied": copied, "skipped": skipped, "missing": missing}
