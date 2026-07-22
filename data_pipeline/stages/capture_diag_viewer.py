"""Stage: build the offline capture-diagnostics HTML viewer.

Renders one self-contained ``capture_diag.html`` for a session showing:

* a STATS card grid (resolution, fps, MB/min, duration, focal length, file size);
* a SYNC panel with stream->Signal and media->Signal offset (ms) + drift (ppm)
  shown as horizontal indicator bars;
* a Cut timeline bar (head cut | kept | tail cut) with the kept percentage.

The HTML loads the vendored ``plotly.min.js`` next to itself (via the existing
``viewers._copy_plotly_next_to``) so the output folder is air-gapped.

CLI::

    python -m data_pipeline.stages.capture_diag_viewer <session_dir> <out.html>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..capture_diag import CaptureDiag, compute_capture_diag
from ..pipeline import LogFn, make_logger
from .viewers import _copy_plotly_next_to


@dataclass(frozen=True)
class CaptureDiagViewerResult:
    html_path: Path
    js_path: Path
    diag: CaptureDiag


def _fmt(value, suffix: str = "", nd: int = 2, *, unavailable: str = "unavailable") -> str:
    if value is None:
        return unavailable
    if isinstance(value, float):
        return f"{value:.{nd}f}{suffix}"
    return f"{value}{suffix}"


def _bar(value: Optional[float], scale: float, *, pos="#34d399", neg="#f87171") -> str:
    """Return inline CSS for a centred signed indicator bar (% width + colour)."""
    if value is None:
        return "width:0%;background:#475569"
    frac = max(-1.0, min(1.0, value / scale)) if scale else 0.0
    pct = abs(frac) * 50.0  # half-width = full scale
    color = pos if value >= 0 else neg
    side = "left:50%" if value >= 0 else f"left:{50 - pct}%"
    return f"width:{pct:.1f}%;{side};background:{color}"


def build_capture_diag_viewer(
    *,
    session_dir: Optional[Path] = None,
    out_html: Path,
    diag: Optional[CaptureDiag] = None,
    pos_file: Optional[Path] = None,
    frame_times: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
    log: Optional[LogFn] = None,
    **diag_kwargs,
) -> CaptureDiagViewerResult:
    """Build ``capture_diag.html`` for a session.

    Pass ``session_dir`` (resolved via ``compute_capture_diag``) or a
    pre-computed ``diag``. Extra ``diag_kwargs`` are forwarded to
    :func:`compute_capture_diag` when ``diag`` is not supplied (e.g. explicit
    ``container file=``, ``measurements_txt=``, ``the probe tool=``).

    ``chop_video_anchor`` — when diagnosing a cut ("segment") clip, pass the
    segment's own ``*.video_anchor.txt``; forwarded to
    :func:`compute_capture_diag`, where its min bootNs WINS over the parent
    capture_meta ``video_t0_boottime_ns`` (segment PTS are rebased to 0).

    Returns a :class:`CaptureDiagViewerResult` with the HTML/JS paths and the
    underlying :class:`CaptureDiag`.
    """
    log_ = make_logger(log)
    out_html = Path(out_html).resolve()

    if diag is None:
        diag = compute_capture_diag(
            session_dir=Path(session_dir) if session_dir else None,
            pos_file=Path(pos_file) if pos_file else None,
            frame_times=Path(frame_times) if frame_times else None,
            chop_video_anchor=(
                Path(chop_video_anchor) if chop_video_anchor else None
            ),
            **diag_kwargs,
        )

    d = diag

    # ---- Stat cards ---------------------------------------------------------
    focal_txt = (
        f"{d.focal_length:g}" if d.focal_length is not None else "unavailable"
    )
    cards = [
        ("Resolution", d.resolution or "unavailable", ""),
        ("Frame rate", _fmt(d.fps, " fps", 2), ""),
        ("Data rate", _fmt(d.mb_per_min, " MB/min", 1), ""),
        ("Duration", _fmt(d.duration_s, " s", 1), ""),
        ("File size", _fmt((d.file_size_bytes or 0) / (1024 * 1024), " MB", 1)
            if d.file_size_bytes else "unavailable", ""),
        ("Focal length", focal_txt, d.focal_source or "unavailable"),
    ]
    cards_html = "\n".join(
        f'<div class="card"><div class="k">{k}</div>'
        f'<div class="v">{v}</div>'
        f'<div class="sub">{sub}</div></div>'
        for k, v, sub in cards
    )

    # ---- Cut timeline ------------------------------------------------------
    if d.total_trim_s is not None and d.duration_s:
        dur = d.duration_s
        head = d.head_trim_s or 0.0
        tail = d.tail_trim_s or 0.0
        kept = max(0.0, dur - head - tail)
        head_pct = head / dur * 100.0
        kept_pct = kept / dur * 100.0
        tail_pct = tail / dur * 100.0
        trim_avail = True
    else:
        head = tail = kept = head_pct = kept_pct = tail_pct = 0.0
        trim_avail = False

    # ---- Sync indicator bars ------------------------------------------------
    data = {
        "audio_offset_ms": d.audio_gnss_offset_ms,
        "audio_drift_ppm": d.audio_gnss_drift_ppm,
        "video_offset_ms": d.video_gnss_offset_ms,
        "video_drift_ppm": d.video_gnss_drift_ppm,
        "diag": d.to_dict(),
    }
    data_js = json.dumps(data, separators=(",", ":"))

    # Bars: offset scaled to +/-500 ms, drift to +/-50 ppm.
    a_off = _bar(d.audio_gnss_offset_ms, 500.0)
    a_drf = _bar(d.audio_gnss_drift_ppm, 50.0)
    v_off = _bar(d.video_gnss_offset_ms, 500.0)
    v_drf = _bar(d.video_gnss_drift_ppm, 50.0)

    notes_html = "".join(f"<li>{_html_escape(n)}</li>" for n in d.notes) or "<li>(none)</li>"

    title = "Capture diagnostics"
    if session_dir:
        title += f" — {Path(session_dir).name}"

    html = _TEMPLATE
    html = html.replace("__TITLE__", _html_escape(title))
    html = html.replace("__CARDS__", cards_html)
    html = html.replace("__A_OFF_BAR__", a_off)
    html = html.replace("__A_DRF_BAR__", a_drf)
    html = html.replace("__V_OFF_BAR__", v_off)
    html = html.replace("__V_DRF_BAR__", v_drf)
    html = html.replace("__A_OFF_TXT__", _fmt(d.audio_gnss_offset_ms, " ms", 1))
    html = html.replace("__A_DRF_TXT__", _fmt(d.audio_gnss_drift_ppm, " ppm", 2))
    html = html.replace("__V_OFF_TXT__", _fmt(d.video_gnss_offset_ms, " ms", 1))
    html = html.replace("__V_DRF_TXT__", _fmt(d.video_gnss_drift_ppm, " ppm", 2))
    html = html.replace("__HEAD_PCT__", f"{head_pct:.2f}")
    html = html.replace("__KEPT_PCT__", f"{kept_pct:.2f}")
    html = html.replace("__TAIL_PCT__", f"{tail_pct:.2f}")
    html = html.replace("__HEAD_S__", f"{head:.3f}")
    html = html.replace("__KEPT_S__", f"{kept:.3f}")
    html = html.replace("__TAIL_S__", f"{tail:.3f}")
    html = html.replace(
        "__KEPT_LABEL__",
        (f"{d.pct_kept:.2f}% kept" if d.pct_kept is not None else "trim unavailable"),
    )
    html = html.replace("__TRIM_AVAIL__", "block" if trim_avail else "none")
    html = html.replace("__NOTES__", notes_html)
    html = html.replace("__DATA__", data_js)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)

    log_(
        f"[capture-diag] wrote {out_html}  "
        f"{d.resolution or '?'} {_fmt(d.fps,'fps',1)} "
        f"{_fmt(d.mb_per_min,'MB/min',1)}  "
        f"audio->GNSS off={_fmt(d.audio_gnss_offset_ms,'ms',1)} "
        f"drift={_fmt(d.audio_gnss_drift_ppm,'ppm',2)}  "
        f"video->GNSS off={_fmt(d.video_gnss_offset_ms,'ms',1)} "
        f"drift={_fmt(d.video_gnss_drift_ppm,'ppm',2)}  "
        f"trim={_fmt(d.total_trim_s,'s',3)} kept={_fmt(d.pct_kept,'%',2)}"
    )
    return CaptureDiagViewerResult(html_path=out_html, js_path=js, diag=d)


def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title>
<script src="plotly.min.js"></script>
<style>
html,body{margin:0;background:#0b0f17;color:#d8e0ee;font-family:system-ui,sans-serif;font-size:13px}
h1{margin:14px 18px 4px;font-size:18px;color:#e5e7eb}
h2{margin:18px 18px 8px;font-size:14px;color:#93c5fd;text-transform:uppercase;letter-spacing:.06em}
.wrap{max-width:1100px;margin:0 auto;padding:0 8px 40px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;padding:0 18px}
.card{background:#131a26;border:1px solid #1f2937;border-radius:10px;padding:14px}
.card .k{color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.card .v{color:#f1f5f9;font-size:22px;font-weight:600;margin-top:4px}
.card .sub{color:#64748b;font-size:11px;margin-top:4px;min-height:14px}
.sync{padding:0 18px}
.syncrow{display:grid;grid-template-columns:130px 1fr 120px;align-items:center;gap:12px;margin:10px 0}
.syncrow .lbl{color:#cbd5e1}
.track{position:relative;height:18px;background:#0f1521;border:1px solid #1f2937;border-radius:9px;overflow:hidden}
.track .mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#334155}
.track .fill{position:absolute;top:0;bottom:0;border-radius:9px}
.syncrow .num{text-align:right;color:#e2e8f0;font-variant-numeric:tabular-nums}
.trim{display:__TRIM_AVAIL__;padding:0 18px}
.trimbar{display:flex;height:34px;border-radius:8px;overflow:hidden;border:1px solid #1f2937}
.trimbar .seg{display:flex;align-items:center;justify-content:center;font-size:11px;color:#0b0f17;font-weight:600;overflow:hidden;white-space:nowrap}
.seg.head{background:#f59e0b}
.seg.kept{background:#34d399}
.seg.tail{background:#f87171}
.kept-label{margin:8px 0 0;color:#94a3b8}
.notes{padding:0 18px;color:#94a3b8;font-size:12px}
.notes ul{margin:6px 0;padding-left:18px}
.legend{padding:4px 18px;color:#64748b;font-size:11px}
</style></head><body>
<div class="wrap">
<h1>__TITLE__</h1>

<h2>Media stats</h2>
<div class="grid">
__CARDS__
</div>

<h2>Synchronisation (relative to GNSS)</h2>
<div class="sync">
<div class="syncrow"><div class="lbl">Audio &rarr; GNSS offset</div>
  <div class="track"><div class="mid"></div><div class="fill" style="__A_OFF_BAR__"></div></div>
  <div class="num">__A_OFF_TXT__</div></div>
<div class="syncrow"><div class="lbl">Audio &rarr; GNSS drift</div>
  <div class="track"><div class="mid"></div><div class="fill" style="__A_DRF_BAR__"></div></div>
  <div class="num">__A_DRF_TXT__</div></div>
<div class="syncrow"><div class="lbl">Video &rarr; GNSS offset</div>
  <div class="track"><div class="mid"></div><div class="fill" style="__V_OFF_BAR__"></div></div>
  <div class="num">__V_OFF_TXT__</div></div>
<div class="syncrow"><div class="lbl">Video &rarr; GNSS drift</div>
  <div class="track"><div class="mid"></div><div class="fill" style="__V_DRF_BAR__"></div></div>
  <div class="num">__V_DRF_TXT__</div></div>
</div>
<div class="legend">Bars are centred at 0; offset scale &plusmn;500&nbsp;ms, drift scale &plusmn;50&nbsp;ppm. Green = positive, red = negative.</div>

<h2>Video trim (GNSS/PPK coverage)</h2>
<div class="trim">
<div class="trimbar">
  <div class="seg head" style="width:__HEAD_PCT__%" title="head trim __HEAD_S__ s">__HEAD_S__s</div>
  <div class="seg kept" style="width:__KEPT_PCT__%" title="kept __KEPT_S__ s">kept __KEPT_S__s</div>
  <div class="seg tail" style="width:__TAIL_PCT__%" title="tail trim __TAIL_S__ s">__TAIL_S__s</div>
</div>
<p class="kept-label">__KEPT_LABEL__ &nbsp;|&nbsp; head __HEAD_S__s &middot; tail __TAIL_S__s</p>
</div>

<h2>Notes</h2>
<div class="notes"><ul>__NOTES__</ul></div>
</div>
<script>window.CAPTURE_DIAG = __DATA__;</script>
</body></html>
"""


def _main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m data_pipeline.stages.capture_diag_viewer",
        description="Build capture_diag.html for a session.",
    )
    ap.add_argument("session_dir", type=Path, help="session folder")
    ap.add_argument("out_html", type=Path, help="output HTML path")
    ap.add_argument("--pos", type=Path, default=None, help="optional .pos for coverage")
    ap.add_argument("--frame-times", type=Path, default=None,
                    help="optional extracted_frame_times.csv")
    ap.add_argument("--chop-video-anchor", type=Path, default=None,
                    help="trimmed ('chop') clip's own *.video_anchor.txt — "
                         "REQUIRED when diagnosing a chop so its min bootNs "
                         "overrides the parent capture_meta video t0")
    ap.add_argument("--ffprobe", type=str, default=None, help="explicit ffprobe path")
    args = ap.parse_args(argv)

    res = build_capture_diag_viewer(
        session_dir=args.session_dir,
        out_html=args.out_html,
        pos_file=args.pos,
        frame_times=args.frame_times,
        chop_video_anchor=args.chop_video_anchor,
        ffprobe=args.ffprobe,
        log=lambda m: print(m),
    )
    print(json.dumps(res.diag.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
