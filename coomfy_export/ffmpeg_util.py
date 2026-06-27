"""FFmpeg helpers for Coomfy export nodes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_ffmpeg() -> str:
  for name in ("ffmpeg", "ffmpeg.exe"):
    path = shutil.which(name)
    if path:
      return path
  try:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
  except Exception:
    pass
  raise FileNotFoundError(
    "ffmpeg not found on PATH (install ffmpeg or imageio-ffmpeg for video export)"
  )


def run_ffmpeg(args: list[str], *, cwd: str | None = None) -> None:
  proc = subprocess.run(
    args,
    cwd=cwd,
    capture_output=True,
    text=True,
    check=False,
  )
  if proc.returncode != 0:
    err = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(err or f"ffmpeg exited with code {proc.returncode}")
