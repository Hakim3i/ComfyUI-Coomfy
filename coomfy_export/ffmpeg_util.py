"""FFmpeg helpers for Coomfy export nodes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .ffmpeg_install import bundled_ffmpeg, ensure_bundled_ffmpeg


def find_ffmpeg() -> str:
  bundled = bundled_ffmpeg()
  if bundled:
    return bundled

  for name in ("ffmpeg", "ffmpeg.exe"):
    path = shutil.which(name)
    if path:
      return path

  try:
    return ensure_bundled_ffmpeg()
  except Exception:
    pass

  try:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
  except Exception:
    pass

  raise FileNotFoundError(
    "ffmpeg not found. Run install_comfyui_custom_nodes.sh from ComfyUI-Coomfy, "
    "or install ffmpeg / imageio-ffmpeg for video export."
  )


def run_ffmpeg(args: list[str], *, cwd: str | None = None) -> None:
  if not args:
    raise ValueError("run_ffmpeg requires at least one argument")
  ffmpeg = find_ffmpeg()
  if Path(args[0]).name not in {"ffmpeg", "ffmpeg.exe"}:
    cmd = [ffmpeg, *args]
  else:
    cmd = [ffmpeg, *args[1:]]
  proc = subprocess.run(
    cmd,
    cwd=cwd,
    capture_output=True,
    text=True,
    check=False,
  )
  if proc.returncode != 0:
    err = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(err or f"ffmpeg exited with code {proc.returncode}")
