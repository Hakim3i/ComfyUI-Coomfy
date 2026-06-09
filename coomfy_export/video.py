"""Mux ComfyUI frames + audio into a metadata-free MP4."""

from __future__ import annotations

import os
import re
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .ffmpeg_util import find_ffmpeg, run_ffmpeg
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

  ffmpeg = find_ffmpeg()
  with tempfile.TemporaryDirectory(prefix="coomfy_export_") as tmp:
    tmp_path = Path(tmp)
    frame_paths: list[Path] = []
    for index, frame in enumerate(images):
      frame_path = tmp_path / f"frame_{index:06d}.png"
      image_tensor_to_pil(frame).save(frame_path, format="PNG", compress_level=3)
      frame_paths.append(frame_path)

    audio_path: Path | None = None
    if audio is not None and audio.get("waveform") is not None:
      audio_path = tmp_path / "audio.wav"
      _write_wav(audio_path, audio["waveform"], int(audio["sample_rate"]))

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
        "-crf",
        str(crf_value),
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
      ]
    )
    if audio_path is not None:
      args.extend(
        [
          "-c:a",
          "aac",
          "-b:a",
          f"{audio_kbps}k",
          "-shortest",
        ]
      )
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
