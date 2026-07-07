#!/usr/bin/env python
"""Thin CLI wrapper: export samples on the stream-zero timeline.

Equivalent to ``python -m data_pipeline.audio_frame_export``; kept as a
script so it is discoverable next to the other pipeline entry points.

Usage:
    python scripts/export_frames_from_audio.py --session <dir> --out <dir>
        [--pos solution.pos] [--fps 6] [--format png]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a source checkout without installing.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_pipeline.audio_frame_export import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
