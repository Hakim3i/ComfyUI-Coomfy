"""Bundle ffmpeg into ComfyUI-Coomfy/bin for Coomfy Export Video."""

from __future__ import annotations

import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = PACK_ROOT / "bin"
_IMAGEIO_FFMPEG_SPEC = "imageio-ffmpeg>=0.5.1"
_BTBN_BASE = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"


def bundled_ffmpeg_path() -> Path:
  if platform.system() == "Windows":
    return BIN_DIR / "ffmpeg.exe"
  return BIN_DIR / "ffmpeg"


def bundled_ffmpeg() -> str | None:
  path = bundled_ffmpeg_path()
  return str(path) if path.is_file() else None


def ffmpeg_supports_encoder(ffmpeg: str | Path, encoder: str) -> bool:
  try:
    proc = subprocess.run(
      [str(ffmpeg), "-hide_banner", "-encoders"],
      capture_output=True,
      text=True,
      check=False,
    )
  except Exception:
    return False
  haystack = f"{proc.stdout or ''}\n{proc.stderr or ''}"
  return encoder in haystack


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


def _btbn_archive_url() -> str | None:
  system = platform.system()
  machine = platform.machine().lower()
  if system == "Windows" and machine in {"amd64", "x86_64"}:
    return f"{_BTBN_BASE}/ffmpeg-master-latest-win64-gpl.zip"
  if system == "Linux" and machine in {"amd64", "x86_64"}:
    return f"{_BTBN_BASE}/ffmpeg-master-latest-linux64-gpl.tar.xz"
  return None


def _download_file(url: str, dest: Path) -> None:
  dest.parent.mkdir(parents=True, exist_ok=True)
  req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Coomfy/1.0"})
  with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:
    shutil.copyfileobj(resp, out)


def _extract_ffmpeg_binary(archive: Path, dest: Path) -> None:
  name = archive.name.lower()
  dest.parent.mkdir(parents=True, exist_ok=True)
  if name.endswith(".zip"):
    with zipfile.ZipFile(archive) as zf:
      for info in zf.infolist():
        normalized = info.filename.replace("\\", "/")
        if normalized.endswith("/bin/ffmpeg.exe"):
          with zf.open(info) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)
          return
    raise FileNotFoundError(f"ffmpeg.exe not found inside {archive.name}")
  if name.endswith(".tar.xz"):
    with tarfile.open(archive, "r:xz") as tf:
      for member in tf.getmembers():
        if member.isfile() and member.name.replace("\\", "/").endswith("/bin/ffmpeg"):
          extracted = tf.extractfile(member)
          if extracted is None:
            continue
          with extracted, dest.open("wb") as out:
            shutil.copyfileobj(extracted, out)
          return
    raise FileNotFoundError(f"ffmpeg not found inside {archive.name}")
  raise ValueError(f"unsupported ffmpeg archive: {archive.name}")


def _install_btbn_ffmpeg(dest: Path) -> str:
  url = _btbn_archive_url()
  if not url:
    raise RuntimeError("no BtbN ffmpeg build for this platform")
  with tempfile.TemporaryDirectory(prefix="coomfy_ffmpeg_") as tmp:
    archive = Path(tmp) / url.rsplit("/", 1)[-1]
    print(f"[ComfyUI-Coomfy] downloading NVENC ffmpeg: {url}")
    _download_file(url, archive)
    _extract_ffmpeg_binary(archive, dest)
  if not dest.is_file():
    raise FileNotFoundError("ffmpeg binary missing after BtbN extract")
  _make_executable(dest)
  if not ffmpeg_supports_encoder(dest, "h264_nvenc"):
    raise RuntimeError("downloaded ffmpeg does not expose h264_nvenc")
  return str(dest)


def _import_imageio_ffmpeg():
  try:
    import imageio_ffmpeg

    return imageio_ffmpeg
  except ImportError:
    subprocess.check_call(
      [sys.executable, "-m", "pip", "install", _IMAGEIO_FFMPEG_SPEC],
    )
    import imageio_ffmpeg

    return imageio_ffmpeg


def _install_imageio_ffmpeg(dest: Path) -> str:
  imageio_ffmpeg = _import_imageio_ffmpeg()
  src = Path(imageio_ffmpeg.get_ffmpeg_exe())
  if not src.is_file():
    raise FileNotFoundError(f"imageio-ffmpeg returned missing binary: {src}")
  print("[ComfyUI-Coomfy] falling back to imageio-ffmpeg (CPU libx264 only)")
  return _copy_ffmpeg(src, dest)


def ensure_bundled_ffmpeg(*, force: bool = False) -> str:
  """Install ffmpeg under ``ComfyUI-Coomfy/bin`` with NVENC when available."""
  dest = bundled_ffmpeg_path()
  if (
    not force
    and dest.is_file()
    and ffmpeg_supports_encoder(dest, "h264_nvenc")
  ):
    return str(dest)

  if dest.is_file() and not ffmpeg_supports_encoder(dest, "h264_nvenc"):
    print("[ComfyUI-Coomfy] bundled ffmpeg lacks NVENC; upgrading build")

  try:
    return _install_btbn_ffmpeg(dest)
  except Exception as exc:
    print(f"[ComfyUI-Coomfy] NVENC ffmpeg download failed: {exc}")

  if dest.is_file():
    return str(dest)

  return _install_imageio_ffmpeg(dest)


def main() -> None:
  path = ensure_bundled_ffmpeg(force=True)
  encoder = "h264_nvenc" if ffmpeg_supports_encoder(path, "h264_nvenc") else "libx264"
  print(f"{path} ({encoder})")


if __name__ == "__main__":
  main()
