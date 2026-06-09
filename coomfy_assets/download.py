"""Download LoRA files into ComfyUI ``models/loras/``."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .download_utils import download_candidates
from .paths import loras_dir

_LOG = "[Coomfy LoRA]"


def _format_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def _is_huggingface_url(url: str) -> bool:
    low = (url or "").lower()
    return "huggingface.co" in low or "hf.co" in low


def _download_file(
    url: str,
    target: Path,
    *,
    civitai_token: str,
    hf_token: str,
    label: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "comfyui-coomfy/3.0"}
    if _is_huggingface_url(url):
        token = (hf_token or "").strip()
    else:
        token = (civitai_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    print(f"{_LOG} DOWNLOAD START")
    print(f"{_LOG}   LoRA: {label}")
    print(f"{_LOG}   URL:  {url}")
    print(f"{_LOG}   Save: {target}")
    with urllib.request.urlopen(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        if total:
            print(f"{_LOG}   Size: {_format_mb(total)}")
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
                        f"{_LOG}   Progress: {_format_mb(downloaded)} / "
                        f"{_format_mb(total)} ({pct}%)",
                        end="\r",
                        flush=True,
                    )
        print()
    tmp.replace(target)
    print(f"{_LOG} DOWNLOAD OK -> {target.name}")


def ensure_lora_file(
    info: dict[str, Any],
    *,
    civitai_token: str = "",
    hf_token: str = "",
) -> Path:
    """Ensure ``info['filename']`` exists under ``models/loras/``; download if missing.

    Uses only the token strings passed in (no ``os.environ``). Raises on failure.
    """
    filename = info.get("filename")
    name = info.get("name") or filename or "?"
    if not filename or not str(filename).strip():
        raise RuntimeError(f"{_LOG} catalog entry missing filename ({name!r})")
    filename = str(filename).strip()
    target = loras_dir() / filename
    if target.is_file():
        print(f"{_LOG} {filename}: on disk ({_format_mb(target.stat().st_size)})")
        return target

    urls = download_candidates(info)
    if not urls:
        raise RuntimeError(
            f"{_LOG} {filename!r}: missing download_url and version_id; "
            "add a download URL on the LoRA row in Coomfy."
        )

    civitai_token = (civitai_token or "").strip()
    hf_token = (hf_token or "").strip()
    last_error: Exception | None = None
    for idx, url in enumerate(urls):
        src = "huggingface" if _is_huggingface_url(url) else "civitai"
        if src == "civitai" and not civitai_token:
            raise RuntimeError(
                f"{_LOG} {filename!r}: Civitai download requires a token. "
                "Set Civitai API key in Coomfy Settings."
            )
        try:
            if idx:
                print(f"{_LOG} retrying with {src} mirror ({idx + 1}/{len(urls)})")
            _download_file(
                url,
                target,
                civitai_token=civitai_token,
                hf_token=hf_token,
                label=str(name),
            )
            last_error = None
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            print(f"{_LOG} DOWNLOAD FAILED: HTTP {exc.code} {exc.reason} ({url})")
        except Exception as exc:
            last_error = exc
            print(f"{_LOG} DOWNLOAD FAILED: {exc} ({url})")

    if not target.is_file():
        detail = str(last_error) if last_error else "unknown error"
        raise RuntimeError(f"{_LOG} could not download {filename!r}: {detail}")
    return target


def ensure_loras_from_json(
    loras_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
) -> list[str]:
    """Parse a JSON list of LoRA dicts and ensure each file exists. Returns filenames."""
    import json

    raw = (loras_json or "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_LOG} invalid loras_json: {exc}") from exc
    if not isinstance(entries, list):
        raise RuntimeError(f"{_LOG} loras_json must be a JSON array")
    applied: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = ensure_lora_file(
            entry,
            civitai_token=civitai_token,
            hf_token=hf_token,
        )
        applied.append(path.name)
    return applied
