"""Copy missing model files from peer SMB shares into local ComfyUI models/."""

from __future__ import annotations

import json
import os
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

_LOG = "[Coomfy peer]"
_CHUNK = 1 << 20  # 1 MiB — same cadence as HTTP download progress
_PROGRESS_PRINT_EVERY = 8  # print at most ~every 8 MiB (avoid spam)

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


def _format_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


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


def _emit_progress(
    status: dict[str, Any],
    *,
    prompt_id: str | None,
) -> None:
    try:
        from .progress import send_asset_download_progress

        send_asset_download_progress(status, prompt_id=prompt_id)
    except Exception:
        pass


def _copy_file_with_progress(
    src: Path,
    dest: Path,
    *,
    bucket: str,
    index: int,
    total_files: int,
    prompt_id: str | None,
) -> None:
    """Chunked copy with console + WS progress (mirrors HTTP download logging)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".{uuid.uuid4().hex}.part")
    total = 0
    try:
        total = int(src.stat().st_size)
    except OSError:
        total = 0

    print(f"{_LOG} COPY START")
    print(f"{_LOG}   Asset: {bucket}/{src.name}")
    print(f"{_LOG}   From:  {src}")
    print(f"{_LOG}   Save:  {dest}")
    if total:
        print(f"{_LOG}   Size: {_format_mb(total)}")

    _emit_progress(
        {
            "asset_kind": "peer_copy",
            "filename": src.name,
            "display_name": src.name,
            "current": index,
            "total": total_files,
            "file_pct": 0.0,
            "overall_frac": (index - 1) / max(1, total_files),
            "bytes_done": 0,
            "bytes_total": total or None,
            "message": f"Copying {src.name} from peer…",
        },
        prompt_id=prompt_id,
    )

    downloaded = 0
    last_printed_pct = -1
    with open(src, "rb") as reader, open(tmp, "wb") as writer:
        while True:
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = min(100, downloaded * 100 // total)
                if pct != last_printed_pct:
                    print(
                        f"{_LOG}   Progress: {_format_mb(downloaded)} / "
                        f"{_format_mb(total)} ({pct}%)",
                        end="\r",
                        flush=True,
                    )
                    last_printed_pct = pct
                    if pct == 100 or pct % 5 == 0:
                        _emit_progress(
                            {
                                "asset_kind": "peer_copy",
                                "filename": src.name,
                                "display_name": src.name,
                                "current": index,
                                "total": total_files,
                                "file_pct": float(pct),
                                "overall_frac": (index - 1 + downloaded / total)
                                / max(1, total_files),
                                "bytes_done": downloaded,
                                "bytes_total": total,
                                "message": f"Copying {src.name} from peer… {pct}%",
                            },
                            prompt_id=prompt_id,
                        )
            elif downloaded % (_CHUNK * _PROGRESS_PRINT_EVERY) < _CHUNK:
                print(
                    f"{_LOG}   Progress: {_format_mb(downloaded)}",
                    end="\r",
                    flush=True,
                )
    print()
    tmp.replace(dest)
    print(f"{_LOG} COPY OK -> {dest.name}")
    _emit_progress(
        {
            "asset_kind": "peer_copy",
            "filename": src.name,
            "display_name": src.name,
            "current": index,
            "total": total_files,
            "file_pct": 100.0,
            "overall_frac": index / max(1, total_files),
            "bytes_done": downloaded,
            "bytes_total": total or downloaded,
            "message": f"Copied {src.name} from peer",
        },
        prompt_id=prompt_id,
    )


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

    print(
        f"{_LOG} peer copy plan: {len(files)} file(s), {len(peers)} peer(s)"
    )
    for peer in peers:
        _ensure_smb(peer["unc"], peer["user"], peer["password"])

    total_files = len(files)
    for index, item in enumerate(files, start=1):
        bucket = item["bucket"]
        filename = item["filename"]
        resolver = _BUCKET_DIRS.get(bucket)
        if resolver is None:
            missing.append(f"{bucket}/{filename}")
            print(f"{_LOG} SKIP unknown bucket {bucket!r} for {filename}")
            continue
        dest_dir = resolver()
        dest = dest_dir / filename
        if dest.is_file():
            skipped.append(filename)
            print(f"{_LOG} {filename}: already on disk ({_format_mb(dest.stat().st_size)})")
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
                    _copy_file_with_progress(
                        src,
                        dest,
                        bucket=bucket,
                        index=index,
                        total_files=total_files,
                        prompt_id=prompt_id,
                    )
                    copied.append(filename)
                    found = True
                    break
                except OSError as exc:
                    print(f"{_LOG} COPY FAILED from {src}: {exc}")
                    continue
            if found:
                break
        if not found:
            missing.append(f"{bucket}/{filename}")
            print(f"{_LOG} MISSING on all peers: {bucket}/{filename}")

    print(
        f"{_LOG} peer copy done: copied={len(copied)} "
        f"skipped={len(skipped)} missing={len(missing)}"
    )
    _emit_progress(
        {
            "phase": "peer_copy",
            "asset_kind": "peer_copy",
            "copied": len(copied),
            "skipped": len(skipped),
            "missing": len(missing),
            "overall_frac": 1.0,
            "message": (
                f"Peer copy done · copied={len(copied)} "
                f"skipped={len(skipped)} missing={len(missing)}"
            ),
        },
        prompt_id=prompt_id,
    )

    return {"copied": copied, "skipped": skipped, "missing": missing}
