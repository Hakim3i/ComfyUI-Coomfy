"""WebSocket progress events for Coomfy asset downloads (read by Coomfy webapp)."""

from __future__ import annotations

from typing import Any


def format_asset_download_message(data: dict[str, Any]) -> str:
    """Human-readable status line for the Coomfy ComfyUI status bar."""
    kind = str(data.get("asset_kind") or "asset").replace("_", " ")
    name = str(data.get("display_name") or data.get("filename") or "").strip()
    current = data.get("current")
    total = data.get("total")
    file_pct = data.get("file_pct")
    overall_pct = data.get("overall_pct")

    parts: list[str] = []
    if current is not None and total is not None:
        try:
            parts.append(f"Downloading {kind} {int(current)}/{int(total)}")
        except (TypeError, ValueError):
            parts.append(f"Downloading {kind}")
    else:
        parts.append(f"Downloading {kind}")

    if name:
        parts.append(name)

    if file_pct is not None:
        try:
            parts.append(f"{int(round(float(file_pct)))}%")
        except (TypeError, ValueError):
            pass
    elif overall_pct is not None:
        try:
            parts.append(f"{int(round(float(overall_pct)))}% overall")
        except (TypeError, ValueError):
            pass

    return " · ".join(parts)


def send_asset_download_progress(
    status: dict[str, Any],
    *,
    prompt_id: str | None = None,
) -> None:
    """Broadcast ``coomfy.asset_download`` on ComfyUI ``/ws`` for the active client."""
    try:
        from server import PromptServer
    except ImportError:
        return

    server = getattr(PromptServer, "instance", None)
    if server is None:
        return

    pid = (prompt_id or "").strip()
    data: dict[str, Any] = {
        "asset_kind": str(status.get("asset_kind") or ""),
        "filename": str(status.get("filename") or ""),
        "display_name": str(
            status.get("display_name") or status.get("name") or status.get("filename") or ""
        ),
        "current": status.get("current"),
        "total": status.get("total"),
        "file_pct": status.get("file_pct", status.get("file_frac")),
        "overall_pct": round(float(status.get("overall_frac", 0.0)) * 100.0, 1),
        "bytes_done": status.get("bytes_done"),
        "bytes_total": status.get("bytes_total"),
    }
    if pid:
        data["prompt_id"] = pid
    data["message"] = str(status.get("message") or format_asset_download_message(data))

    # Broadcast to every /ws client (Coomfy webapp opens its own clientId).
    server.send_sync("coomfy.asset_download", data, None)
