"""Download model assets into the matching ComfyUI ``models/`` folders.

Covers checkpoints, LoRAs, ControlNets, upscalers, detailer detectors + SAM,
diffusion models, text encoders, and VAE. Token strings are passed in by the
Coomfy webapp (never read from ``os.environ``).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .download_utils import (
    civitai_authenticated_url,
    download_candidates,
    is_civitai_url,
    is_huggingface_url,
)
from .paths import (
    checkpoints_dir,
    controlnet_dir,
    diffusion_models_dir,
    loras_dir,
    sams_dir,
    text_encoders_dir,
    ultralytics_dir,
    upscale_models_dir,
    vae_dir,
)

_LOG = "[Coomfy LoRA]"
_CN_LOG = "[Coomfy ControlNet]"
_CKPT_LOG = "[Coomfy Checkpoint]"
_UP_LOG = "[Coomfy Upscale]"
_DT_LOG = "[Coomfy Detailer]"
_DM_LOG = "[Coomfy DiffusionModel]"
_TE_LOG = "[Coomfy TextEncoder]"
_VAE_LOG = "[Coomfy VAE]"


@dataclass
class _DownloadProgress:
    """Maps per-file byte progress into one 0–1 bar across *total* pending assets."""

    total: int
    done: int = 0
    on_progress: Callable[[float], None] | None = None

    def file_bytes(self, byte_frac: float) -> None:
        if self.on_progress is None or self.total <= 0:
            return
        inner = max(0.0, min(1.0, byte_frac))
        self.on_progress(min(1.0, (self.done + inner) / self.total))

    def file_finished(self) -> None:
        self.done += 1
        if self.on_progress is None or self.total <= 0:
            return
        self.on_progress(min(1.0, self.done / self.total))


def _parse_json_array(raw: str, *, log: str) -> list[dict[str, Any]]:
    import json

    text = (raw or "").strip()
    if not text:
        return []
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{log} invalid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise RuntimeError(f"{log} JSON must be an array")
    return [e for e in entries if isinstance(e, dict)]


def _detailer_target_path(entry: dict[str, Any]) -> Path | None:
    folder = str(entry.get("folder") or "").strip().lower()
    rel = str(entry.get("relative_path") or entry.get("filename") or "").strip()
    if not rel or folder not in {"ultralytics", "sams"}:
        return None
    base = ultralytics_dir() if folder == "ultralytics" else sams_dir()
    return base / rel


def _model_target_path(base_dir: Path, filename: str) -> Path:
    return base_dir / filename


def count_pending_assets(
    *,
    checkpoints_json: str = "",
    loras_json: str = "",
    controlnets_json: str = "",
    upscalers_json: str = "",
    detailers_json: str = "",
    diffusion_models_json: str = "",
    text_encoders_json: str = "",
    vae_json: str = "",
) -> int:
    """Count manifest rows that still need downloading (not already on disk)."""
    pending = 0
    for entry in _parse_json_array(checkpoints_json, log=_CKPT_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and not (checkpoints_dir() / filename).is_file():
            pending += 1
    for entry in _parse_json_array(loras_json, log=_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and not (loras_dir() / filename).is_file():
            pending += 1
    for entry in _parse_json_array(controlnets_json, log=_CN_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and not (controlnet_dir() / filename).is_file():
            pending += 1
    for entry in _parse_json_array(upscalers_json, log=_UP_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and not (upscale_models_dir() / filename).is_file():
            pending += 1
    for entry in _parse_json_array(detailers_json, log=_DT_LOG):
        target = _detailer_target_path(entry)
        if target is not None and not target.is_file():
            pending += 1
    for entry in _parse_json_array(diffusion_models_json, log=_DM_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and not _model_target_path(diffusion_models_dir(), filename).is_file():
            pending += 1
    for entry in _parse_json_array(text_encoders_json, log=_TE_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and not _model_target_path(text_encoders_dir(), filename).is_file():
            pending += 1
    for entry in _parse_json_array(vae_json, log=_VAE_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and not _model_target_path(vae_dir(), filename).is_file():
            pending += 1
    return pending


def _format_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def _format_download_error(exc: Exception | None) -> str:
    if exc is None:
        return "unknown error"
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:240]
        except Exception:
            pass
        detail = f"HTTP {exc.code} {exc.reason}"
        if body:
            detail = f"{detail}: {body}"
        return detail
    return str(exc)


def _download_file(
    url: str,
    target: Path,
    *,
    civitai_token: str,
    hf_token: str,
    label: str,
    log: str = _LOG,
    on_file_progress: Callable[[float], None] | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "comfyui-coomfy/3.0"}
    request_url = url
    if is_civitai_url(url):
        token = (civitai_token or "").strip()
        if token:
            # Query token only — do not send Authorization on Civitai URLs. urllib
            # forwards that header to the S3/R2 redirect target and S3 returns 400.
            request_url = civitai_authenticated_url(url, token)
    elif is_huggingface_url(url):
        token = (hf_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(request_url, headers=headers)
    print(f"{log} DOWNLOAD START")
    print(f"{log}   Asset: {label}")
    print(f"{log}   URL:  {url}")
    print(f"{log}   Save: {target}")
    if on_file_progress is not None:
        on_file_progress(0.0)
    with urllib.request.urlopen(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        if total:
            print(f"{log}   Size: {_format_mb(total)}")
        downloaded = 0
        with open(tmp, "wb") as handle:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(100, downloaded * 100 // total)
                    print(
                        f"{log}   Progress: {_format_mb(downloaded)} / "
                        f"{_format_mb(total)} ({pct}%)",
                        end="\r",
                        flush=True,
                    )
                    if on_file_progress is not None:
                        on_file_progress(downloaded / total)
        print()
    if on_file_progress is not None:
        on_file_progress(1.0)
    tmp.replace(target)
    print(f"{log} DOWNLOAD OK -> {target.name}")


def _ensure_named_file(
    info: dict[str, Any],
    *,
    base_dir: Path,
    log: str,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> Path:
    """Ensure ``info['filename']`` exists under *base_dir*; download if missing."""
    filename = info.get("filename")
    name = info.get("name") or filename or "?"
    if not filename or not str(filename).strip():
        raise RuntimeError(f"{log} catalog entry missing filename ({name!r})")
    filename = str(filename).strip()
    target = _model_target_path(base_dir, filename)
    if target.is_file():
        print(f"{log} {filename}: on disk ({_format_mb(target.stat().st_size)})")
        return target

    urls = download_candidates(info)
    if not urls:
        raise RuntimeError(
            f"{log} {filename!r}: missing download_url and version_id; "
            "add a download URL on the row in Coomfy."
        )

    civitai_token = (civitai_token or "").strip()
    hf_token = (hf_token or "").strip()
    last_error: Exception | None = None
    file_progress = progress.file_bytes if progress is not None else None
    for idx, url in enumerate(urls):
        src = "huggingface" if is_huggingface_url(url) else "civitai"
        try:
            if idx:
                print(f"{log} retrying with {src} mirror ({idx + 1}/{len(urls)})")
            _download_file(
                url,
                target,
                civitai_token=civitai_token,
                hf_token=hf_token,
                label=str(name),
                log=log,
                on_file_progress=file_progress,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"{log} DOWNLOAD FAILED: {_format_download_error(exc)} ({url})")

    if not target.is_file():
        raise RuntimeError(
            f"{log} could not download {filename!r}: {_format_download_error(last_error)}"
        )
    if progress is not None:
        progress.file_finished()
    return target


def ensure_lora_file(
    info: dict[str, Any],
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> Path:
    """Ensure ``info['filename']`` exists under ``models/loras/``; download if missing."""
    return _ensure_named_file(
        info,
        base_dir=loras_dir(),
        log=_LOG,
        civitai_token=civitai_token,
        hf_token=hf_token,
        progress=progress,
    )


def ensure_loras_from_json(
    loras_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> list[str]:
    """Parse a JSON list of LoRA dicts and ensure each file exists. Returns filenames."""
    applied: list[str] = []
    for entry in _parse_json_array(loras_json, log=_LOG):
        path = ensure_lora_file(
            entry,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        )
        applied.append(path.name)
    return applied


def ensure_controlnet_file(
    info: dict[str, Any],
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> Path:
    return _ensure_named_file(
        info,
        base_dir=controlnet_dir(),
        log=_CN_LOG,
        civitai_token=civitai_token,
        hf_token=hf_token,
        progress=progress,
    )


def ensure_controlnets_from_json(
    controlnets_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> list[str]:
    applied: list[str] = []
    for entry in _parse_json_array(controlnets_json, log=_CN_LOG):
        path = ensure_controlnet_file(
            entry,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        )
        applied.append(path.name)
    return applied


def ensure_checkpoint_file(
    info: dict[str, Any],
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> Path:
    """Ensure ``info['filename']`` exists under ``models/checkpoints/``."""
    return _ensure_named_file(
        info,
        base_dir=checkpoints_dir(),
        log=_CKPT_LOG,
        civitai_token=civitai_token,
        hf_token=hf_token,
        progress=progress,
    )


def ensure_checkpoints_from_json(
    checkpoints_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> list[str]:
    """Parse a JSON list of checkpoint dicts and ensure each file exists."""
    applied: list[str] = []
    for entry in _parse_json_array(checkpoints_json, log=_CKPT_LOG):
        path = ensure_checkpoint_file(
            entry,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        )
        applied.append(path.name)
    return applied


def ensure_upscale_file(
    info: dict[str, Any],
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> Path:
    return _ensure_named_file(
        info,
        base_dir=upscale_models_dir(),
        log=_UP_LOG,
        civitai_token=civitai_token,
        hf_token=hf_token,
        progress=progress,
    )


def ensure_upscalers_from_json(
    upscalers_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> list[str]:
    applied: list[str] = []
    for entry in _parse_json_array(upscalers_json, log=_UP_LOG):
        path = ensure_upscale_file(
            entry,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        )
        applied.append(path.name)
    return applied


def ensure_detailer_file(
    info: dict[str, Any],
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> Path:
    rel = str(info.get("relative_path") or info.get("filename") or "").strip()
    name = info.get("name") or rel or "?"
    target = _detailer_target_path(info)
    if target is None:
        raise RuntimeError(f"{_DT_LOG} catalog entry missing folder/path ({name!r})")
    if target.is_file():
        print(f"{_DT_LOG} {rel}: on disk ({_format_mb(target.stat().st_size)})")
        return target

    urls = download_candidates(info)
    if not urls:
        raise RuntimeError(f"{_DT_LOG} {rel!r}: missing download_url")

    civitai_token = (civitai_token or "").strip()
    hf_token = (hf_token or "").strip()
    last_error: Exception | None = None
    file_progress = progress.file_bytes if progress is not None else None
    for idx, url in enumerate(urls):
        src = "huggingface" if is_huggingface_url(url) else "direct"
        try:
            if idx:
                print(f"{_DT_LOG} retrying with {src} mirror ({idx + 1}/{len(urls)})")
            _download_file(
                url,
                target,
                civitai_token=civitai_token,
                hf_token=hf_token,
                label=str(name),
                log=_DT_LOG,
                on_file_progress=file_progress,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"{_DT_LOG} DOWNLOAD FAILED: {_format_download_error(exc)} ({url})")

    if not target.is_file():
        raise RuntimeError(
            f"{_DT_LOG} could not download {rel!r}: {_format_download_error(last_error)}"
        )
    if progress is not None:
        progress.file_finished()
    return target


def ensure_detailers_from_json(
    detailers_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> list[str]:
    applied: list[str] = []
    for entry in _parse_json_array(detailers_json, log=_DT_LOG):
        path = ensure_detailer_file(
            entry,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        )
        rel = str(entry.get("relative_path") or entry.get("filename") or path.name).strip()
        applied.append(rel)
    return applied


def ensure_diffusion_models_from_json(
    diffusion_models_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> list[str]:
    applied: list[str] = []
    for entry in _parse_json_array(diffusion_models_json, log=_DM_LOG):
        path = _ensure_named_file(
            entry,
            base_dir=diffusion_models_dir(),
            log=_DM_LOG,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        )
        rel = str(entry.get("filename") or path.name).strip()
        applied.append(rel)
    return applied


def ensure_text_encoders_from_json(
    text_encoders_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> list[str]:
    applied: list[str] = []
    for entry in _parse_json_array(text_encoders_json, log=_TE_LOG):
        path = _ensure_named_file(
            entry,
            base_dir=text_encoders_dir(),
            log=_TE_LOG,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        )
        rel = str(entry.get("filename") or path.name).strip()
        applied.append(rel)
    return applied


def ensure_vae_from_json(
    vae_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
    progress: _DownloadProgress | None = None,
) -> list[str]:
    applied: list[str] = []
    for entry in _parse_json_array(vae_json, log=_VAE_LOG):
        path = _ensure_named_file(
            entry,
            base_dir=vae_dir(),
            log=_VAE_LOG,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        )
        rel = str(entry.get("filename") or path.name).strip()
        applied.append(rel)
    return applied


def ensure_all_assets(
    *,
    checkpoints_json: str = "",
    loras_json: str = "",
    controlnets_json: str = "",
    upscalers_json: str = "",
    detailers_json: str = "",
    diffusion_models_json: str = "",
    text_encoders_json: str = "",
    vae_json: str = "",
    civitai_token: str = "",
    hf_token: str = "",
    on_progress: Callable[[float], None] | None = None,
) -> dict[str, list[str]]:
    """Download every asset listed in the JSON manifests."""
    civitai_token = (civitai_token or "").strip()
    hf_token = (hf_token or "").strip()
    pending = count_pending_assets(
        checkpoints_json=checkpoints_json,
        loras_json=loras_json,
        controlnets_json=controlnets_json,
        upscalers_json=upscalers_json,
        detailers_json=detailers_json,
        diffusion_models_json=diffusion_models_json,
        text_encoders_json=text_encoders_json,
        vae_json=vae_json,
    )
    progress = (
        _DownloadProgress(total=pending, on_progress=on_progress)
        if on_progress is not None and pending > 0
        else None
    )
    result = {
        "checkpoints": ensure_checkpoints_from_json(
            checkpoints_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        ),
        "loras": ensure_loras_from_json(
            loras_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        ),
        "controlnets": ensure_controlnets_from_json(
            controlnets_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        ),
        "upscalers": ensure_upscalers_from_json(
            upscalers_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        ),
        "detailers": ensure_detailers_from_json(
            detailers_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        ),
        "diffusion_models": ensure_diffusion_models_from_json(
            diffusion_models_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        ),
        "text_encoders": ensure_text_encoders_from_json(
            text_encoders_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        ),
        "vae": ensure_vae_from_json(
            vae_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
            progress=progress,
        ),
    }
    from .ltx_lora_inspect import inspect_ltx_lora_filenames

    result["lora_inspect"] = inspect_ltx_lora_filenames(result.get("loras") or [])
    if on_progress is not None and pending > 0:
        on_progress(1.0)
    return result
