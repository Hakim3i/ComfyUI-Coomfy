"""Bundle ffmpeg into ComfyUI-Coomfy/bin for Coomfy Export Video."""

from __future__ import annotations

import platform
import shutil
import stat
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = PACK_ROOT / "bin"


def bundled_ffmpeg_path() -> Path:
  if platform.system() == "Windows":
    return BIN_DIR / "ffmpeg.exe"
  return BIN_DIR / "ffmpeg"


def bundled_ffmpeg() -> str | None:
  path = bundled_ffmpeg_path()
  return str(path) if path.is_file() else None


def _make_executable(path: Path) -> None:
  if platform.system() == "Windows":
    return
  mode = path.stat().st_mode
  path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_ffmpeg(src: Path, dest: Path) -> str:
  BIN_DIR.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dest)
  _make_executable(dest)
  return str(dest)


def ensure_bundled_ffmpeg(*, force: bool = False) -> str:
  """Install ffmpeg under ``ComfyUI-Coomfy/bin`` (idempotent)."""
  dest = bundled_ffmpeg_path()
  if not force and dest.is_file():
    return str(dest)

  try:
    import imageio_ffmpeg

    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
  except Exception as exc:
    raise FileNotFoundError(
      "Could not bundle ffmpeg: install imageio-ffmpeg first "
      "(pip install imageio-ffmpeg), or place ffmpeg on PATH."
    ) from exc

  if not src.is_file():
    raise FileNotFoundError(f"imageio-ffmpeg returned missing binary: {src}")

  return _copy_ffmpeg(src, dest)


def main() -> None:
  path = ensure_bundled_ffmpeg()
  print(path)


if __name__ == "__main__":
  main()
