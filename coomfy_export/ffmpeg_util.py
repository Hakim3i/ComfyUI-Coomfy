"""FFmpeg helpers for Coomfy export nodes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .ffmpeg_install import bundled_ffmpeg, ensure_bundled_ffmpeg, ffmpeg_supports_encoder


def _looks_like_ffmpeg(path: str) -> bool:
  return Path(path).stem.lower() == "ffmpeg"


def find_ffmpeg() -> str:
  bundled = bundled_ffmpeg()
  if bundled:
    return bundled

  try:
    return ensure_bundled_ffmpeg()
  except Exception:
    pass

  try:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
  except Exception:
    pass

  for name in ("ffmpeg", "ffmpeg.exe"):
    path = shutil.which(name)
    if path:
      return path

  raise FileNotFoundError(
    "ffmpeg not found. Restart ComfyUI so ComfyUI-Coomfy can bundle ffmpeg "
    "into bin/, or install imageio-ffmpeg."
  )


def run_ffmpeg(args: list[str], *, cwd: str | None = None) -> None:
  if not args:
    raise ValueError("run_ffmpeg requires at least one argument")
  ffmpeg = find_ffmpeg()
  rest = args[1:] if _looks_like_ffmpeg(args[0]) else args
  proc = subprocess.run(
    [ffmpeg, *rest],
    cwd=cwd,
    capture_output=True,
    text=True,
    check=False,
  )
  if proc.returncode != 0:
    err = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(err or f"ffmpeg exited with code {proc.returncode}")
