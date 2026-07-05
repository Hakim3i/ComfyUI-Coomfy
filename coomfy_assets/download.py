"""Download model assets into the matching ComfyUI ``models/`` folders.

Covers checkpoints, LoRAs, ControlNets, upscalers, detailer detectors + SAM,
diffusion models, text encoders, and VAE. Token strings are passed in by the
Coomfy webapp (never read from ``os.environ``).
"""

from __future__ import annotations

import json
import struct
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
    current_kind: str = ""
    current_name: str = ""
    current_display: str = ""
    bytes_done: int = 0
    bytes_total: int = 0
    on_progress: Callable[[float], None] | None = None
    on_status: Callable[[dict[str, Any]], None] | None = None

    def begin_asset(self, kind: str, filename: str, display_name: str = "") -> None:
        self.current_kind = str(kind or "asset").strip()
        self.current_name = str(filename or "").strip()
        self.current_display = str(display_name or filename or "").strip()
        self.bytes_done = 0
        self.bytes_total = 0
        self._notify(file_frac=0.0)

    def file_bytes(
        self,
        byte_frac: float,
        *,
        bytes_done: int = 0,
        bytes_total: int = 0,
    ) -> None:
        if self.on_progress is None and self.on_status is None:
            return
        inner = max(0.0, min(1.0, byte_frac))
        if bytes_done > 0:
            self.bytes_done = bytes_done
        if bytes_total > 0:
            self.bytes_total = bytes_total
        if self.on_progress is not None and self.total > 0:
            self.on_progress(min(1.0, (self.done + inner) / self.total))
        self._notify(file_frac=inner)

    def file_finished(self) -> None:
        self.done += 1
        if self.on_progress is not None and self.total > 0:
            self.on_progress(min(1.0, self.done / self.total))
        self._notify(file_frac=1.0)

    def _notify(self, *, file_frac: float | None = None) -> None:
        if self.on_status is None:
            return
        overall = 0.0
        if self.total > 0:
            inner = 0.0 if file_frac is None else max(0.0, min(1.0, file_frac))
            overall = min(1.0, (self.done + inner) / self.total)
        payload: dict[str, Any] = {
            "overall_frac": overall,
            "asset_kind": self.current_kind,
            "filename": self.current_name,
            "display_name": self.current_display,
            "current": min(self.total, self.done + 1) if self.total else self.done + 1,
            "total": self.total,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
        }
        if file_frac is not None:
            payload["file_frac"] = round(file_frac * 100.0, 1)
        self.on_status(payload)


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


def _min_bytes_for_kind(asset_kind: str) -> int:
    kind = str(asset_kind or "").strip().lower()
    if kind in {"checkpoint", "diffusion model", "text encoder", "vae"}:
        return 10_000_000
    if kind == "lora":
        return 50_000
    return 64_000


def _safetensors_tensor_names(path: Path) -> tuple[str, ...]:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            return tuple(handle.keys())
    except Exception:
        return _safetensors_tensor_names_raw(path)


def _safetensors_tensor_names_raw(path: Path) -> tuple[str, ...]:
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) < 8:
                return ()
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len < 2 or header_len > min(size - 8, 50_000_000):
                return ()
            header_bytes = handle.read(int(header_len))
        header = json.loads(header_bytes.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return ()
    if not isinstance(header, dict):
        return ()
    return tuple(
        str(key)
        for key in header
        if str(key) and str(key) != "__metadata__"
    )


def _is_plausible_safetensors(path: Path, *, min_bytes: int) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < min_bytes:
        return False
    try:
        with open(path, "rb") as handle:
            head = handle.read(256)
    except OSError:
        return False
    if not head:
        return False
    lowered = head.lstrip().lower()
    if lowered.startswith((b"<!doctype", b"<html", b"<head", b"{", b"[")):
        return False
    if len(head) < 8:
        return False
    header_len = struct.unpack("<Q", head[:8])[0]
    if header_len < 2 or header_len > min(size - 8, 50_000_000):
        return False
    with open(path, "rb") as handle:
        handle.seek(8)
        header_json = handle.read(min(int(header_len), 4096))
    return bool(header_json.lstrip().startswith(b"{"))
    

def _is_plausible_checkpoint_file(path: Path, *, min_bytes: int) -> bool:
    """Reject LoRAs, HTML stubs, and diffusion-only files in checkpoints/."""
    if not _is_plausible_safetensors(path, min_bytes=min_bytes):
        return False
    names = _safetensors_tensor_names(path)
    if not names:
        return False
    count = len(names)
    if count < 80:
        return False
    blob = " ".join(names).lower()
    lora_named = sum(1 for name in names if "lora" in name.lower())
    if lora_named and lora_named >= max(3, count // 2):
        return False
    full_ckpt_markers = (
        "model.diffusion_model",
        "cond_stage_model",
        "first_stage_model",
        "conditioner.embedders",
        "conditioner.",
    )
    if any(marker in blob for marker in full_ckpt_markers):
        return True
    # Misplaced diffusion-only UNET files fail CheckpointLoaderSimple.
    if count < 400:
        return False
    return "diffusion_model" in blob


def _is_plausible_lora_file(path: Path, *, min_bytes: int) -> bool:
    if not _is_plausible_safetensors(path, min_bytes=min_bytes):
        return False
    names = _safetensors_tensor_names(path)
    if not names:
        return False
    if len(names) > 2000:
        return False
    blob = " ".join(names).lower()
    if any(
        marker in blob
        for marker in (
            "model.diffusion_model",
            "cond_stage_model",
            "first_stage_model",
        )
    ):
        return False
    return "lora" in blob or len(names) < 400


def _file_passes_validation(path: Path, *, asset_kind: str, min_bytes: int) -> bool:
    kind = str(asset_kind or "").strip().lower()
    if path.suffix.lower() != ".safetensors":
        try:
            return path.stat().st_size >= min_bytes
        except OSError:
            return False
    if kind == "checkpoint":
        return _is_plausible_checkpoint_file(path, min_bytes=min_bytes)
    if kind == "lora":
        return _is_plausible_lora_file(path, min_bytes=min_bytes)
    return _is_plausible_safetensors(path, min_bytes=min_bytes)


def _needs_download(
    path: Path,
    *,
    asset_kind: str = "asset",
    validate_safetensors: bool = True,
) -> bool:
    if not path.is_file():
        return True
    min_bytes = _min_bytes_for_kind(asset_kind)
    if validate_safetensors and not _file_passes_validation(
        path, asset_kind=asset_kind, min_bytes=min_bytes
    ):
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        print(
            f"[Coomfy ensure] removing invalid {asset_kind} "
            f"{path.name} ({size} bytes)"
        )
        try:
            path.unlink()
        except OSError:
            pass
        return True
    try:
        if path.stat().st_size < min_bytes:
            try:
                path.unlink()
            except OSError:
                pass
            return True
    except OSError:
        return True
    return False


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
        if filename and _needs_download(
            checkpoints_dir() / filename, asset_kind="checkpoint"
        ):
            pending += 1
    for entry in _parse_json_array(loras_json, log=_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and _needs_download(loras_dir() / filename, asset_kind="lora"):
            pending += 1
    for entry in _parse_json_array(controlnets_json, log=_CN_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and _needs_download(
            controlnet_dir() / filename, asset_kind="controlnet"
        ):
            pending += 1
    for entry in _parse_json_array(upscalers_json, log=_UP_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and _needs_download(
            upscale_models_dir() / filename, asset_kind="upscaler"
        ):
            pending += 1
    for entry in _parse_json_array(detailers_json, log=_DT_LOG):
        target = _detailer_target_path(entry)
        if target is not None and _needs_download(
            target, asset_kind="detailer", validate_safetensors=False
        ):
            pending += 1
    for entry in _parse_json_array(diffusion_models_json, log=_DM_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and _needs_download(
            _model_target_path(diffusion_models_dir(), filename),
            asset_kind="diffusion model",
        ):
            pending += 1
    for entry in _parse_json_array(text_encoders_json, log=_TE_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and _needs_download(
            _model_target_path(text_encoders_dir(), filename),
            asset_kind="text encoder",
        ):
            pending += 1
    for entry in _parse_json_array(vae_json, log=_VAE_LOG):
        filename = str(entry.get("filename") or "").strip()
        if filename and _needs_download(
            _model_target_path(vae_dir(), filename), asset_kind="vae"
        ):
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
    on_file_progress: Callable[..., None] | None = None,
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
                        on_file_progress(
                            downloaded / total,
                            bytes_done=downloaded,
                            bytes_total=total,
                        )
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
    asset_kind: str = "asset",
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
    if not _needs_download(target, asset_kind=asset_kind):
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
    if progress is not None:
        progress.begin_asset(asset_kind, filename, str(name))

    def _file_progress(
        byte_frac: float,
        *,
        bytes_done: int = 0,
        bytes_total: int = 0,
    ) -> None:
        if progress is not None:
            progress.file_bytes(
                byte_frac,
                bytes_done=bytes_done,
                bytes_total=bytes_total,
            )

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
                on_file_progress=_file_progress if progress is not None else None,
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
        asset_kind="lora",
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
        asset_kind="controlnet",
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
        asset_kind="checkpoint",
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
        asset_kind="upscaler",
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
    if not _needs_download(target, asset_kind="detailer", validate_safetensors=False):
        print(f"{_DT_LOG} {rel}: on disk ({_format_mb(target.stat().st_size)})")
        return target

    urls = download_candidates(info)
    if not urls:
        raise RuntimeError(f"{_DT_LOG} {rel!r}: missing download_url")

    civitai_token = (civitai_token or "").strip()
    hf_token = (hf_token or "").strip()
    last_error: Exception | None = None
    if progress is not None:
        progress.begin_asset("detailer", rel, str(name))

    def _file_progress(
        byte_frac: float,
        *,
        bytes_done: int = 0,
        bytes_total: int = 0,
    ) -> None:
        if progress is not None:
            progress.file_bytes(
                byte_frac,
                bytes_done=bytes_done,
                bytes_total=bytes_total,
            )

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
                on_file_progress=_file_progress if progress is not None else None,
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
            asset_kind="diffusion model",
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
            asset_kind="text encoder",
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
            asset_kind="vae",
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
    on_status: Callable[[dict[str, Any]], None] | None = None,
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
        _DownloadProgress(
            total=pending,
            on_progress=on_progress,
            on_status=on_status,
        )
        if pending > 0 and (on_progress is not None or on_status is not None)
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
