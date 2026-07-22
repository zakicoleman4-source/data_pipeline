"""Per-frame time-notation table: every clock for every extracted frame.

Today the per-frame timing story is scattered across several CSVs (source PTS
in ``extracted_frame_times.csv``, UTC inside Georef.csv, audio-relative times
in ``frames_for_external.csv``) and the satellite-time (GPST) notation for a
frame exists nowhere. This module emits ONE table, one row per frame, with
every notation side by side:

    Image, video_pts_s, boot_ns, utc_s, utc_iso, gpst_s, t_audio_s

Column semantics (all derived from the shared boot->UTC anchor -- the same
resolution chain the rest of the pipeline uses, see
:mod:`data_pipeline.frame_time` and
:func:`data_pipeline.audio_frame_export.resolve_session_anchors`):

* ``video_pts_s`` -- the frame's true source PTS in seconds (verbatim
  ``t_video_s`` from ``extracted_frame_times.csv``).
* ``boot_ns``     -- ``video_t0_boot_ns + video_pts_s * 1e9`` (CLOCK_BOOTTIME).
  Blank for legacy sessions with no boottime t0.
* ``utc_s``       -- absolute UTC (POSIX seconds) via the canonical
  :func:`data_pipeline.frame_time.make_frame_to_utc` lift.
* ``utc_iso``     -- the same instant as ISO-8601 ``YYYY-MM-DDThh:mm:ss.sssZ``
  (millisecond, half-up -- mirrors ``stages.user_export._iso_utc``).
* ``gpst_s``      -- ``utc_s + leap_seconds(utc_s)`` (18 s in 2026).
* ``t_audio_s``   -- seconds from audio sample 0
  (``(boot_ns - audio_start_boot_ns) / 1e9``). Blank when the session has no
  audio anchor, when ``boot_ns`` is unavailable, and for pre-audio frames
  (``t_audio_s < 0``) -- matching the pipeline's convention of treating
  pre-audio frames as outside the audio timeline (they are still listed).
"""

from __future__ import annotations

import csv
import datetime as _dt
import html as _html
import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "FRAME_TIME_TABLE_HEADER",
    "build_frame_time_table",
]

#: Exact output CSV header (order matters; downstream tooling keys off it).
FRAME_TIME_TABLE_HEADER = (
    "Image",
    "video_pts_s",
    "boot_ns",
    "utc_s",
    "utc_iso",
    "gpst_s",
    "t_audio_s",
)

LogFn = Callable[[str], None]


def _log_through(log: Optional[LogFn]) -> LogFn:
    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    return _log


def _iso_utc(utc_s: Optional[float]) -> str:
    """ISO-8601 UTC ``YYYY-MM-DDThh:mm:ss.sssZ`` (millisecond, half-up).

    Mirrors ``data_pipeline.stages.user_export._iso_utc``; empty string when
    the input is missing or not finite.
    """
    if utc_s is None or not math.isfinite(utc_s):
        return ""
    ms_total = math.floor(utc_s * 1000.0 + 0.5)  # half-up at ms precision
    secs, ms = divmod(ms_total, 1000)
    t = _dt.datetime.fromtimestamp(secs, tz=_dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"


# ---------------------------------------------------------------------------
# Anchor resolution
# ---------------------------------------------------------------------------


def _resolve_anchors(session_dir: Path, log: LogFn):
    """Resolve (boot_anchor, video_t0_boot_ns, audio_start_boot_ns).

    Primary path reuses :func:`audio_frame_export.resolve_session_anchors`
    (identical chop-anchor precedence). Sessions without an audio anchor --
    which that resolver refuses -- fall back to resolving just the boot->UTC
    anchor and video t0 with the same canonical helpers (``audio_start`` then
    stays ``None`` and ``t_audio_s`` is blank).
    """
    from .pipeline import RawInputs

    inputs = RawInputs.from_folder(Path(session_dir))

    if inputs.audio_anchor_txt is not None:
        from .audio_frame_export import resolve_session_anchors

        a = resolve_session_anchors(
            Path(session_dir), inputs=inputs, need_utc=True, log=log
        )
        return a.boot_anchor, a.video_t0_boot_ns, a.audio_start_boot_ns

    log("[frame-times] no audio_anchor_*.txt: t_audio_s column will be blank")
    from .frame_time import resolve_video_t0_boottime_ns
    from .time_sync import fit_time_anchor_with_fallback

    boot_anchor, source = fit_time_anchor_with_fallback(
        inputs.recording_txt, inputs.measurements_txt
    )
    log(f"[frame-times] boot->UTC anchor from {source}")
    video_t0 = resolve_video_t0_boottime_ns(
        capture_meta=inputs.capture_meta_json,
        video_anchor=inputs.video_anchor_txt,
        chop_video_anchor=inputs.chop_video_anchor if inputs.is_chop else None,
        log=log,
    )
    return boot_anchor, video_t0, None


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _build_rows(
    frame_times: Sequence[Tuple[str, float]],
    boot_anchor,
    video_t0_boot_ns: Optional[float],
    audio_start_boot_ns: Optional[float],
) -> List[Tuple[str, str, str, str, str, str, str]]:
    """Compute the formatted CSV cells for every (Image, t_video_s) row."""
    from .frame_time import make_frame_to_utc
    from .time_sync import get_leap_seconds_for_epoch

    to_utc = make_frame_to_utc(boot_anchor, video_t0_boot_ns)

    rows: List[Tuple[str, str, str, str, str, str, str]] = []
    for image, pts in frame_times:
        pts = float(pts)
        utc_s = float(to_utc(pts))
        gpst_s = utc_s + float(get_leap_seconds_for_epoch(utc_s))

        if video_t0_boot_ns is not None:
            boot_ns = float(video_t0_boot_ns) + pts * 1e9
            boot_cell = f"{boot_ns:.0f}"
        else:
            boot_ns = None
            boot_cell = ""

        audio_cell = ""
        if boot_ns is not None and audio_start_boot_ns is not None:
            t_audio_s = (boot_ns - float(audio_start_boot_ns)) / 1e9
            # Pre-audio frames stay listed but the cell is left blank (the
            # pipeline's "pre-audio" convention: not on the audio timeline).
            if t_audio_s >= 0.0:
                audio_cell = f"{t_audio_s:.6f}"

        rows.append(
            (
                str(image),
                f"{pts:.6f}",
                boot_cell,
                f"{utc_s:.3f}",
                _iso_utc(utc_s),
                f"{gpst_s:.3f}",
                audio_cell,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# HTML emission (self-contained, no external deps)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font: 13px/1.45 system-ui, sans-serif; margin: 1.2rem; color: #1a1a2e; }}
  h1 {{ font-size: 1.1rem; margin: 0 0 .6rem; }}
  .meta {{ color: #667; margin-bottom: .8rem; }}
  input#q {{ padding: .35rem .55rem; width: 22rem; max-width: 90%;
            border: 1px solid #bbc; border-radius: 4px; margin-bottom: .7rem; }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; white-space: nowrap; }}
  th, td {{ padding: .22rem .6rem; border-bottom: 1px solid #e2e2ea;
           text-align: right; font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ cursor: pointer; position: sticky; top: 0; background: #f4f4f8;
       user-select: none; }}
  th .dir {{ color: #99a; font-size: .8em; }}
  tr:hover td {{ background: #f7f7fc; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">{n_rows} frames &middot; click a header to sort &middot; type to filter</div>
<input id="q" type="search" placeholder="filter rows (substring match)&hellip;">
<div class="wrap">
<table id="t">
<thead><tr>{head_cells}</tr></thead>
<tbody>
{body_rows}
</tbody>
</table>
</div>
<script>
(function () {{
  var table = document.getElementById('t');
  var tbody = table.tBodies[0];
  var rows = Array.prototype.slice.call(tbody.rows);
  document.getElementById('q').addEventListener('input', function () {{
    var q = this.value.toLowerCase();
    rows.forEach(function (r) {{
      r.style.display = r.textContent.toLowerCase().indexOf(q) >= 0 ? '' : 'none';
    }});
  }});
  var ths = table.tHead.rows[0].cells;
  var sortCol = -1, asc = true;
  function cellVal(r, i) {{ return r.cells[i].textContent.trim(); }}
  Array.prototype.forEach.call(ths, function (th, i) {{
    th.addEventListener('click', function () {{
      asc = (sortCol === i) ? !asc : true;
      sortCol = i;
      var numeric = rows.every(function (r) {{
        var v = cellVal(r, i);
        return v === '' || !isNaN(parseFloat(v));
      }});
      rows.sort(function (a, b) {{
        var va = cellVal(a, i), vb = cellVal(b, i), c;
        if (numeric) {{
          var na = va === '' ? -Infinity : parseFloat(va);
          var nb = vb === '' ? -Infinity : parseFloat(vb);
          c = na - nb;
        }} else {{
          c = va < vb ? -1 : va > vb ? 1 : 0;
        }}
        return asc ? c : -c;
      }});
      rows.forEach(function (r) {{ tbody.appendChild(r); }});
      Array.prototype.forEach.call(ths, function (h) {{
        var d = h.querySelector('.dir'); if (d) d.textContent = '';
      }});
      var dir = th.querySelector('.dir');
      if (dir) dir.textContent = asc ? ' \\u25B2' : ' \\u25BC';
    }});
  }});
}})();
</script>
</body>
</html>
"""


def _write_html(
    html_path: Path,
    rows: Sequence[Tuple[str, str, str, str, str, str, str]],
    title: str,
) -> None:
    head_cells = "".join(
        f"<th>{_html.escape(h)}<span class=\"dir\"></span></th>"
        for h in FRAME_TIME_TABLE_HEADER
    )
    body = "\n".join(
        "<tr>" + "".join(f"<td>{_html.escape(c)}</td>" for c in row) + "</tr>"
        for row in rows
    )
    html_path.write_text(
        _HTML_TEMPLATE.format(
            title=_html.escape(title),
            n_rows=len(rows),
            head_cells=head_cells,
            body_rows=body,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_frame_time_table(
    extracted_frame_times_csv: Path,
    *,
    session_dir: Optional[Path] = None,
    anchors=None,
    out_csv: Path,
    write_html: bool = True,
    log: Optional[LogFn] = None,
) -> Path:
    """Write the per-frame every-time-notation table.

    ``anchors`` may be any object exposing ``boot_anchor`` (a fitted
    :class:`~data_pipeline.time_sync.TimeAnchor`), ``video_t0_boot_ns``
    (``None`` for legacy sessions) and ``audio_start_boot_ns`` (``None`` when
    no audio) -- e.g. an :class:`~data_pipeline.audio_frame_export.SessionAnchors`.
    When omitted, anchors are resolved from ``session_dir`` (reusing
    :func:`audio_frame_export.resolve_session_anchors`; sessions without an
    audio anchor fall back to boot-anchor + video-t0 resolution only).

    Writes ``out_csv`` with header exactly
    ``Image,video_pts_s,boot_ns,utc_s,utc_iso,gpst_s,t_audio_s`` and, when
    ``write_html``, a sibling ``<stem>.html`` self-contained
    searchable/sortable table. Returns the CSV path.
    """
    _log = _log_through(log)

    from .parsers import read_frame_times_csv

    frame_times = read_frame_times_csv(Path(extracted_frame_times_csv))
    if not frame_times:
        raise ValueError(
            f"No frame rows in {extracted_frame_times_csv} "
            "(expected 'Image,t_video_s' CSV)"
        )

    if anchors is not None:
        boot_anchor = getattr(anchors, "boot_anchor", None)
        video_t0_boot_ns = getattr(anchors, "video_t0_boot_ns", None)
        audio_start_boot_ns = getattr(anchors, "audio_start_boot_ns", None)
    elif session_dir is not None:
        boot_anchor, video_t0_boot_ns, audio_start_boot_ns = _resolve_anchors(
            Path(session_dir), _log
        )
    else:
        raise ValueError("build_frame_time_table needs session_dir or anchors")

    if boot_anchor is None:
        raise ValueError(
            "No boot->UTC time anchor available: cannot map frames to "
            "UTC/GPST (need a fitted TimeAnchor)"
        )
    if video_t0_boot_ns is None:
        _log(
            "[frame-times] legacy session (no boottime t0): using the "
            "anchor's direct pts->utc mapping; boot_ns / t_audio_s blank"
        )

    rows = _build_rows(
        frame_times, boot_anchor, video_t0_boot_ns, audio_start_boot_ns
    )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FRAME_TIME_TABLE_HEADER)
        w.writerows(rows)
    _log(f"[frame-times] wrote {out_csv} ({len(rows)} frames)")

    if write_html:
        html_path = out_csv.with_suffix(".html")
        _write_html(html_path, rows, title=f"Frame time table — {out_csv.stem}")
        _log(f"[frame-times] wrote {html_path}")

    return out_csv
