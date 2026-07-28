"""Coomfy system monitor — broadcasts CPU/RAM/GPU/VRAM stats over the ComfyUI websocket.

Replaces the Crystools dependency: a single daemon thread emits one small
``coomfy.monitor`` JSON message per second to all connected clients (same
broadcast pattern as crystools.monitor), so Coomfy and any ComfyUI browser tab
can render live host stats without third-party packs.

Payload shape (ints, percent 0-100)::

    {
        "type": "coomfy.monitor",
        "cpu_pct": 12,
        "ram_pct": 42,
        "gpus": [
            {"index": 0, "gpu_pct": 72, "vram_pct": 81,
             "vram_used_gb": 26.1, "vram_total_gb": 32.0}
        ],
    }

Missing dependencies degrade gracefully: without psutil the cpu/ram fields are
omitted; without pynvml the gpus list is empty (Coomfy falls back to the
/system_stats HTTP endpoint for VRAM).
"""

from __future__ import annotations

import threading
import time

import server

_LOG = "[ComfyUI-Coomfy]"
_INTERVAL_SECONDS = 1.0

try:
    import psutil

    psutil.cpu_percent(interval=None)  # prime the non-blocking sampler
except ImportError:
    psutil = None
    print(f"{_LOG} monitor: psutil not installed — cpu/ram stats disabled")

_NVML = None
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML = pynvml
except Exception as exc:  # ImportError or NVML driver error
    print(f"{_LOG} monitor: pynvml unavailable ({exc}) — gpu stats disabled")

_thread: threading.Thread | None = None
_reported_error: str | None = None


def _gpu_rows() -> list[dict]:
    if _NVML is None:
        return []
    rows = []
    for index in range(_NVML.nvmlDeviceGetCount()):
        handle = _NVML.nvmlDeviceGetHandleByIndex(index)
        util = _NVML.nvmlDeviceGetUtilizationRates(handle)
        mem = _NVML.nvmlDeviceGetMemoryInfo(handle)
        rows.append(
            {
                "index": index,
                "gpu_pct": int(util.gpu),
                "vram_pct": int(round(mem.used / mem.total * 100)) if mem.total else 0,
                "vram_used_gb": round(mem.used / 1024**3, 1),
                "vram_total_gb": round(mem.total / 1024**3, 1),
            }
        )
    return rows


def _sample() -> dict:
    payload: dict = {"type": "coomfy.monitor"}
    if psutil is not None:
        payload["cpu_pct"] = int(psutil.cpu_percent(interval=None))
        payload["ram_pct"] = int(psutil.virtual_memory().percent)
    payload["gpus"] = _gpu_rows()
    return payload


def _loop() -> None:
    global _reported_error
    serv = None
    while serv is None:
        time.sleep(0.5)
        serv = server.PromptServer.instance
    print(f"{_LOG} monitor: broadcasting coomfy.monitor every {_INTERVAL_SECONDS:.0f}s")
    while True:
        try:
            serv.send_sync("coomfy.monitor", _sample())
        except Exception as exc:  # never let the monitor die silently
            key = f"{type(exc).__name__}: {exc}"
            if key != _reported_error:
                _reported_error = key
                print(f"{_LOG} monitor error: {exc}")
        time.sleep(_INTERVAL_SECONDS)


def start_monitor() -> None:
    global _thread
    if _thread is not None:
        return
    _thread = threading.Thread(target=_loop, daemon=True, name="coomfy-monitor")
    _thread.start()
