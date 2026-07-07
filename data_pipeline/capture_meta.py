"""Parser for the session manifest (``capture_meta.json``).

Only the fields the pipeline consumes are surfaced; the rest is preserved in
``raw``. Parsing tolerates missing keys — a sparse manifest yields ``None``
fields rather than raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CaptureMeta:
    """Manifest fields the pipeline consumes."""

    video_name: Optional[str] = None
    video_t0_boottime_ns: Optional[int] = None
    timestamp_source: Optional[str] = None
    dropped_frames: Optional[int] = None
    mono_to_boot_offset_ns: Optional[int] = None
    audio_timebase: Optional[str] = None
    anchor_format: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_boottime(self) -> bool:
        src = (self.timestamp_source or "").lower()
        if src and src != "boottime":
            return False
        return True


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def parse_capture_meta(path: Path) -> CaptureMeta:
    """Parse ``capture_meta.json`` into a :class:`CaptureMeta`.

    Raises ``FileNotFoundError`` if the file is missing; tolerates any missing
    or malformed individual keys.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {}
    video = data.get("video") or {}
    audio = data.get("audio") or {}
    clock = data.get("clock") or {}
    return CaptureMeta(
        video_name=(video.get("mp4") or None),
        video_t0_boottime_ns=_to_int(video.get("video_t0_boottime_ns")),
        timestamp_source=(video.get("timestamp_source") or None),
        dropped_frames=_to_int(video.get("dropped_frames")),
        mono_to_boot_offset_ns=_to_int(clock.get("mono_to_boot_offset_ns")),
        audio_timebase=(audio.get("timebase") or None),
        anchor_format=(data.get("anchor_format") or None),
        raw=data,
    )
