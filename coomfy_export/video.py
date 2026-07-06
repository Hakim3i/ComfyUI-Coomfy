"""Mux ComfyUI frames + audio into a metadata-free MP4."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .ffmpeg_util import find_ffmpeg, ffmpeg_supports_encoder, run_ffmpeg
from .tensors import image_tensor_to_pil


def _next_counter(output_dir: Path, prefix: str) -> int:
  matcher = re.compile(rf"{re.escape(prefix)}_(\d+)\D*\..+", re.IGNORECASE)
  max_counter = 0
  if output_dir.is_dir():
    for name in os.listdir(output_dir):
      match = matcher.fullmatch(name)
      if match:
        max_counter = max(max_counter, int(match.group(1)))
  return max_counter + 1


def _write_wav(path: Path, waveform, sample_rate: int) -> None:
  data = waveform.detach().cpu().numpy()
  if data.ndim == 3:
    data = data[0]
  if data.ndim == 1:
    data = data[None, :]
  channels, samples = data.shape
  pcm = np.clip(data, -1.0, 1.0)
  pcm = (pcm.T.reshape(-1) * 32767.0).astype(np.int16)
  with wave.open(str(path), "wb") as wf:
    wf.setnchannels(int(channels))
    wf.setsampwidth(2)
    wf.setframerate(int(sample_rate))
    wf.writeframes(pcm.tobytes())


def _frame_to_rgb24(frame) -> np.ndarray:
  """One ComfyUI IMAGE row -> contiguous H×W×3 uint8."""
  array = np.clip(255.0 * frame.detach().cpu().numpy(), 0, 255).astype(np.uint8)
  if array.ndim == 4:
    array = array[0]
  if array.shape[-1] > 3:
    array = array[..., :3]
  return np.ascontiguousarray(array)


def _mux_via_pipe(
    ffmpeg: str,
    frames: list,
    *,
    frame_rate: float,
    file_path: Path,
    audio_path: Path | None,
    crf_value: int,
    audio_kbps: int,
) -> None:
  """Stream RGB frames on stdin — avoids per-frame PNG disk I/O."""
  if not frames:
    raise ValueError("mux_video requires at least one frame")
  first = _frame_to_rgb24(frames[0])
  height, width = int(first.shape[0]), int(first.shape[1])
  use_nvenc = ffmpeg_supports_encoder(ffmpeg, "h264_nvenc")

  args = [
    ffmpeg,
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "-s",
    f"{width}x{height}",
    "-r",
    str(frame_rate),
    "-i",
    "pipe:0",
  ]
  if audio_path is not None:
    args.extend(["-i", str(audio_path)])
  if use_nvenc:
    args.extend(
      [
        "-c:v",
        "h264_nvenc",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(max(0, min(51, crf_value))),
      ]
    )
  else:
    args.extend(
      [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "faster",
        "-crf",
        str(crf_value),
      ]
    )
  args.extend(["-movflags", "+faststart", "-map_metadata", "-1"])
  if audio_path is not None:
    args.extend(["-c:a", "aac", "-b:a", f"{audio_kbps}k", "-shortest"])
  else:
    args.append("-an")
  args.append(str(file_path))

  proc = subprocess.Popen(
    args,
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
  )
  assert proc.stdin is not None
  try:
    proc.stdin.write(first.tobytes())
    for frame in frames[1:]:
      proc.stdin.write(_frame_to_rgb24(frame).tobytes())
  finally:
    proc.stdin.close()
  stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="replace")
  if proc.wait() != 0:
    raise RuntimeError(stderr.strip() or "ffmpeg pipe mux failed")


def mux_video(
    images,
    *,
    frame_rate: float,
    audio: dict | None,
    filename_prefix: str,
    output_dir: str,
    enabled: bool,
    crf: int,
    audio_bitrate_kbps: int,
) -> dict[str, Any]:
  """Write an MP4 under ``output_dir`` and return a ComfyUI UI payload entry."""
  import folder_paths

  frame_rate = max(1.0, float(frame_rate))
  crf_value = max(0, min(51, int(crf)))
  if not enabled:
    crf_value = min(crf_value, 14)

  audio_kbps = max(32, int(audio_bitrate_kbps))
  if not enabled:
    audio_kbps = max(audio_kbps, 192)

  out_root = Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)
  subfolder = ""
  if "/" in filename_prefix or "\\" in filename_prefix:
    prefix_path = Path(filename_prefix.replace("\\", "/"))
    subfolder = prefix_path.parent.as_posix()
    prefix = prefix_path.name
    target_dir = out_root / subfolder
  else:
    prefix = filename_prefix
    target_dir = out_root
  target_dir.mkdir(parents=True, exist_ok=True)

  counter = _next_counter(target_dir, prefix)
  filename = f"{prefix}_{counter:05}.mp4"
  file_path = target_dir / filename

  frames = list(images)
  ffmpeg = find_ffmpeg()
  with tempfile.TemporaryDirectory(prefix="coomfy_export_") as tmp:
    tmp_path = Path(tmp)
    audio_path: Path | None = None
    if audio is not None and audio.get("waveform") is not None:
      audio_path = tmp_path / "audio.wav"
      _write_wav(audio_path, audio["waveform"], int(audio["sample_rate"]))

    try:
      _mux_via_pipe(
        ffmpeg,
        frames,
        frame_rate=frame_rate,
        file_path=file_path,
        audio_path=audio_path,
        crf_value=crf_value,
        audio_kbps=audio_kbps,
      )
    except Exception:
      # Fallback: PNG sequence (older ffmpeg / pipe issues).
      frame_paths: list[Path] = []
      for index, frame in enumerate(frames):
        frame_path = tmp_path / f"frame_{index:06d}.png"
        image_tensor_to_pil(frame).save(frame_path, format="PNG", compress_level=3)
        frame_paths.append(frame_path)
      args = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(frame_rate),
        "-i",
        str(tmp_path / "frame_%06d.png"),
      ]
      if audio_path is not None:
        args.extend(["-i", str(audio_path)])
      args.extend(
        [
          "-c:v",
          "libx264",
          "-pix_fmt",
          "yuv420p",
          "-preset",
          "faster",
          "-crf",
          str(crf_value),
          "-movflags",
          "+faststart",
          "-map_metadata",
          "-1",
        ]
      )
      if audio_path is not None:
        args.extend(["-c:a", "aac", "-b:a", f"{audio_kbps}k", "-shortest"])
      else:
        args.append("-an")
      args.append(str(file_path))
      run_ffmpeg(args)

  rel_type = "output" if str(output_dir) == folder_paths.get_output_directory() else "temp"
  return {
    "filename": filename,
    "subfolder": subfolder,
    "type": rel_type,
    "format": "video/mp4",
    "frame_rate": frame_rate,
    "workflow": None,
    "fullpath": str(file_path),
  }
