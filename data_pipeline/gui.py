"""Tkinter GUI orchestrating the four pipeline stages.

The UI is intentionally simple and stays inside the standard library:

* Each pipeline stage runs in a worker thread so the GUI stays responsive.
* The log widget is fed via a thread-safe ``queue.Queue`` polled from the
  Tk main loop - never write to Tk widgets from a worker thread.
* The "Run Post-processing" stage is *external*: the GUI just pauses, waiting for the
  user to drop a ``.pos`` file back in the input box before continuing.
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import coords as coords_mod
from .pipeline import RawInputs
from .stages import adaptive_frames as adaptive_stage
from .stages import frames as frames_stage
from .stages import kml_export as kml_stage
from .stages import georef as georef_stage
from .stages import ppk as ppk_stage
from .stages import multimask_ppk as multimask_stage
from .stages import rinex as rinex_stage
from .stages import t02 as t02_stage
from .stages import viewers as viewers_stage


# When the user leaves the android_rinex field empty, the Interchange-format stage falls
# back to the vendored copy under ``vendor/android_rinex/src``.
DEFAULT_ANDROID_RINEX_SRC = ""

# Recent-projects persistence file (last 5 RAW folders).
_RECENT_FILE = Path.home() / ".data_pipeline_recent.json"
_RECENT_MAX = 5

# Per-user GUI preferences (currently: complexity level).
_SETTINGS_FILE = Path.home() / ".data_pipeline_settings.json"

# Complexity levels. Each tab declares the minimum level it should appear at;
# the notebook is filtered against the current setting on every (re)build.
COMPLEXITY_LEVELS = ("basic", "medium", "complex")
COMPLEXITY_LABELS = {
    "basic":   "Basic — video → frames only",
    "medium":  "Medium — RINEX + frames + viewers (no PPK / smoothing)",
    "complex": "Complex — full PPK pipeline",
}


def _load_settings() -> dict:
    try:
        if _SETTINGS_FILE.is_file():
            return json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_settings(d: dict) -> None:
    try:
        _SETTINGS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except OSError:
        pass


def _get_complexity() -> str:
    lvl = _load_settings().get("complexity", "complex")
    return lvl if lvl in COMPLEXITY_LEVELS else "complex"


def _set_complexity(level: str) -> None:
    s = _load_settings()
    s["complexity"] = level
    _save_settings(s)


def _write_smoothed_csv(path, rows) -> None:
    """Persist a smoother output as a small CSV (atomic via .tmp + replace).

    Columns: utc_s, lat_deg, lon_deg, h_m, quality, vn, ve, vu, ns.
    Used by the Smoothers tab in the GUI; safe to call without holding
    the Tk main thread.
    """
    import csv as _csv
    import os as _os
    import tempfile as _tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = _tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                dir=str(path.parent))
    try:
        with _os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["utc_s", "lat_deg", "lon_deg", "h_m", "quality",
                        "vn", "ve", "vu", "ns"])
            for r in rows:
                w.writerow([
                    f"{r.utc_s:.6f}", f"{r.lat_deg:.9f}",
                    f"{r.lon_deg:.9f}", f"{r.h_m:.4f}",
                    r.quality, r.vn, r.ve, r.vu, r.ns,
                ])
        _os.replace(tmp, path)
    except Exception:
        try:
            _os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class _Tooltip:
    """Hover tooltip attached to any Tkinter widget."""

    _BG = "#1a2540"
    _FG = "#c8d8e8"

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._win: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", lambda e: self._show(e, text), add="+")
        widget.bind("<Leave>", lambda e: self._hide(), add="+")

    def _show(self, event: tk.Event, text: str) -> None:  # type: ignore[type-arg]
        self._hide()
        w = event.widget
        x = w.winfo_rootx() + 16
        y = w.winfo_rooty() + w.winfo_height() + 6
        self._win = tw = tk.Toplevel(w)
        tw.overrideredirect(True)
        tw.attributes("-topmost", True)
        tw.geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=text, justify="left",
            background=self._BG, foreground=self._FG,
            relief="solid", borderwidth=1,
            font=("Segoe UI", 8), wraplength=360, padx=6, pady=4,
        ).pack()

    def _hide(self) -> None:
        if self._win:
            self._win.destroy()
            self._win = None


@dataclass
class _Paths:
    """Mutable bundle of paths the GUI builds up across stages."""

    raw_folder: Path | None = None
    raw: RawInputs | None = None
    android_rinex_src: Path | None = None  # None => use vendored copy
    out_dir: Path | None = None

    obs_path: Path | None = None
    pos_path: Path | None = None

    frame_times_csv: Path | None = None
    georef_csv: Path | None = None


class App:
    POLL_MS = 80
    # Log-drain safety caps: a runaway producer (e.g. an external tool whose
    # captured output is re-logged) can enqueue hundreds of thousands of
    # messages. Draining them all in one `after` tick blocks the Tk mainloop
    # ("Not Responding") and an unbounded Text widget can exhaust Tk's text
    # B-tree allocator (Tcl_Panic -> hard process death). Drain at most
    # MAX_PER_TICK messages per tick and keep the widget under MAX_LOG_LINES.
    MAX_PER_TICK = 200
    MAX_LOG_LINES = 10_000
    # Sentinel for the client-export source chooser: keep the historical
    # behaviour (each smoother's *.client.csv carries that smoother's own
    # rows). Any other value is fed to user_export.resolve_export_rows.
    EXPORT_SOURCE_AS_RUN = "(as run)"

    def __init__(self) -> None:
        # Try drag-and-drop root (optional dependency); fall back to plain Tk.
        self._dnd_files = None
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES  # type: ignore[import-not-found]
            self.root = TkinterDnD.Tk()
            self._dnd_files = DND_FILES
        except Exception:
            self.root = tk.Tk()
        from . import __version__
        self.root.title(f"data_pipeline  v{__version__}")
        self.root.geometry("1100x900")
        self.root.minsize(960, 720)

        self.paths = _Paths()
        self._log_q: queue.Queue[str] = queue.Queue()
        self._busy = False
        self._buttons: list[tk.Widget] = []

        self._recent: list[str] = self._load_recent()
        self._recent_menu: Optional[tk.Menu] = None

        self._preview_win:       Optional[tk.Toplevel] = None
        self._preview_canvas:    Optional[tk.Canvas]  = None
        self._preview_rot_lbl:   Optional[ttk.Label]  = None
        self._preview_gen:       int                  = 0
        self._preview_last_path: Optional[str]        = None

        self._apply_style()
        self._build_ui()
        self.root.after(self.POLL_MS, self._drain_log_queue)

    def _apply_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        # Modernised palette: deeper backgrounds, brighter accent, dedicated
        # purple highlight for selected tabs / accent buttons, and a softer
        # body text colour for less eye-strain over long sessions.
        bg          = "#0e1525"   # window background
        bg_card     = "#172238"   # raised surfaces (labelframes, entries)
        bg_card_hi  = "#1c2942"   # hover surfaces
        bg_inset    = "#0a1020"   # text + log widgets
        bg_tab      = "#0c1424"   # inactive tab
        fg          = "#e2e8f0"   # primary text
        fg_dim      = "#94a3b8"   # secondary text
        accent      = "#7dd3fc"   # cyan headline accent
        accent_hi   = "#a78bfa"   # purple highlight (selected tabs / focus)
        good_green  = "#34d399"   # ready / success
        warn_amber  = "#fbbf24"   # busy / warn
        err_red     = "#f87171"   # failure
        border      = "#334155"
        border_soft = "#243352"

        # Expose key colours for other widgets / methods.
        self._bg          = bg
        self._bg_card     = bg_card
        self._bg_inset    = bg_inset
        self._accent      = accent
        self._accent_hi   = accent_hi
        self._good_green  = good_green
        self._warn_amber  = warn_amber
        self._err_red     = err_red
        self._fg          = fg
        self._fg_dim      = fg_dim

        self.root.configure(bg=bg)
        style.configure("TFrame",            background=bg)
        style.configure("Card.TFrame",       background=bg_card)
        style.configure("TLabelframe",       background=bg, foreground=fg,
                        bordercolor=border_soft, relief="groove", padding=6)
        style.configure("TLabelframe.Label", background=bg,
                        foreground=accent, font=("Segoe UI Semibold", 9, "bold"))
        style.configure("TLabel",            background=bg, foreground=fg,
                        font=("Segoe UI", 9))
        style.configure("Dim.TLabel",        background=bg, foreground=fg_dim,
                        font=("Segoe UI", 9))
        style.configure("Hint.TLabel",       background=bg, foreground=fg_dim,
                        font=("Segoe UI", 8))
        style.configure("TCheckbutton",      background=bg, foreground=fg,
                        font=("Segoe UI", 9), indicatormargin=4)
        style.configure("TRadiobutton",      background=bg, foreground=fg)
        style.configure("TSeparator",        background=border_soft)
        style.configure("TEntry",            fieldbackground=bg_card, foreground=fg,
                        insertcolor=accent, bordercolor=border,
                        selectbackground="#1e3a8a")
        style.configure("TSpinbox",          fieldbackground=bg_card, foreground=fg,
                        arrowcolor=fg_dim, bordercolor=border)
        style.configure("TCombobox",         fieldbackground=bg_card, foreground=fg,
                        arrowcolor=fg_dim, bordercolor=border,
                        selectbackground="#1e3a8a")
        style.configure("TButton",           background=bg_card, foreground=fg,
                        relief="flat", padding=[13, 6],
                        font=("Segoe UI", 9), borderwidth=1, bordercolor=border)
        style.map("TButton",
                  background=[("active", bg_card_hi), ("pressed", "#0c1a38"),
                              ("disabled", "#11192a")],
                  foreground=[("active", accent), ("disabled", "#3a4460")],
                  bordercolor=[("active", accent_hi)])
        # Notebook
        style.configure("TNotebook",         background=bg, borderwidth=0,
                        tabmargins=[2, 6, 0, 0])
        style.configure("TNotebook.Tab",     background=bg_tab, foreground=fg_dim,
                        padding=[20, 8], font=("Segoe UI", 9),
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", bg), ("active", bg_card_hi)],
                  foreground=[("selected", accent_hi), ("active", accent)],
                  font=[("selected", ("Segoe UI Semibold", 9, "bold"))],
                  expand=[("selected", [1, 1, 1, 0])])
        # Accent button (primary action)
        style.configure("Accent.TButton",    background="#1e3a8a", foreground=fg,
                        relief="flat", padding=[16, 7],
                        font=("Segoe UI Semibold", 9, "bold"),
                        borderwidth=1, bordercolor=accent_hi)
        style.map("Accent.TButton",
                  background=[("active", "#2a4eb8"), ("pressed", "#0f1e54"),
                              ("disabled", "#0c1e34")],
                  foreground=[("disabled", "#2a4a60")])
        # Progressbar
        style.configure("TProgressbar",      background=accent_hi,
                        troughcolor=bg_card, bordercolor=border_soft,
                        lightcolor=accent_hi, darkcolor=accent_hi,
                        thickness=8)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Menubar (File > Recent Projects) ─────────────────────────────────
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        self._recent_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Recent RAW folders", menu=self._recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)
        self._rebuild_recent_menu()

        # ── Title header ──────────────────────────────────────────────────────
        hdr_bg = "#080d1c"
        hdr = tk.Frame(self.root, bg=hdr_bg, height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        title_box = tk.Frame(hdr, bg=hdr_bg)
        title_box.pack(side="left", padx=20, pady=6, anchor="w")
        tk.Label(title_box, text="data_pipeline", bg=hdr_bg,
                 fg=self._accent,
                 font=("Segoe UI Semibold", 15, "bold")).pack(anchor="w")
        tk.Label(title_box,
                 text="Device PPK  →  georeferenced frame export",
                 bg=hdr_bg, fg=self._fg_dim,
                 font=("Segoe UI", 9)).pack(anchor="w")

        # Right-side build / version indicator
        meta_box = tk.Frame(hdr, bg=hdr_bg)
        meta_box.pack(side="right", padx=18, pady=10)
        tk.Label(meta_box, text="GUI build", bg=hdr_bg,
                 fg=self._fg_dim, font=("Segoe UI", 8)).pack(anchor="e")
        from . import __version__ as _v
        tk.Label(meta_box, text=f"v{_v}  ·  RTKLIB EX 2.5.0",
                 bg=hdr_bg, fg=self._accent_hi,
                 font=("Segoe UI Semibold", 9)).pack(anchor="e")

        # Two-tone accent strip for a clean visual divider.
        tk.Frame(self.root, bg=self._accent_hi, height=2).pack(fill="x")
        tk.Frame(self.root, bg=self._accent, height=1).pack(fill="x")

        # ── Complexity selector ──────────────────────────────────────────────
        # The chosen level decides which tabs the notebook exposes:
        #   basic   → just "Media → Samples" (no Reference at all)
        #   medium  → Inputs + Interchange-format + Samples+CSV + Viewers + Media (no Post-processing / no T02)
        #   complex → everything
        # Persisted in ~/.data_pipeline_settings.json so the choice survives
        # restarts. Default = complex (existing users see no change).
        self._complexity = tk.StringVar(value=_get_complexity())
        complexity_bar = tk.Frame(self.root, bg=self._bg)
        complexity_bar.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Label(complexity_bar, text="Mode:",
                  font=("Segoe UI Semibold", 9, "bold")).pack(side="left", padx=(4, 8))
        for lvl in COMPLEXITY_LEVELS:
            rb = ttk.Radiobutton(
                complexity_bar, text=COMPLEXITY_LABELS[lvl], value=lvl,
                variable=self._complexity,
                command=self._on_complexity_changed,
            )
            rb.pack(side="left", padx=4)

        # Notebook host — rebuilt whenever the complexity changes so the tab
        # set always matches the user's chosen mode.
        self._nb_host = tk.Frame(self.root, bg=self._bg)
        self._nb_host.pack(fill="both", expand=True, padx=8, pady=(4, 2))
        self._build_notebook()

        # ── Status strip ──────────────────────────────────────────────────────
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill="x", padx=12, pady=(4, 2))

        self._dot_canvas = tk.Canvas(
            status_bar, width=12, height=12, bg=self._bg, highlightthickness=0,
        )
        self._dot_canvas.pack(side="left")
        self._dot_item = self._dot_canvas.create_oval(
            1, 1, 11, 11, fill=self._good_green, outline="",
        )

        self._status_lbl = ttk.Label(
            status_bar, text="Ready",
            foreground=self._good_green,
            font=("Segoe UI Semibold", 9, "bold"),
        )
        self._status_lbl.pack(side="left", padx=(6, 0))
        self._progress_bar = ttk.Progressbar(
            status_bar, orient="horizontal", mode="determinate",
            length=220, style="TProgressbar",
        )
        self._progress_bar.pack(side="left", padx=(14, 0))
        self._progress_bar.pack_forget()  # shown by _show_progress_bar()
        self._progress_running = False  # tracks indeterminate animation

        # Right-side mini hint — mirrors the current complexity mode so the
        # operator always knows which workflow they're in.
        self._status_hint = ttk.Label(
            status_bar, text="", style="Hint.TLabel",
        )
        self._status_hint.pack(side="right")
        self._refresh_status_hint()

        # ── Log area ──────────────────────────────────────────────────────────
        bottom = ttk.LabelFrame(self.root, text="  Log  ")
        bottom.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        self.log_text = tk.Text(
            bottom, height=12, bg=self._bg_inset, fg=self._fg,
            insertbackground=self._accent, wrap="none",
            font=("Consolas", 9), relief="flat", borderwidth=0,
            selectbackground="#1e3a8a", padx=8, pady=4,
        )
        self.log_text.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(bottom, orient="vertical", command=self.log_text.yview)
        sb.pack(fill="y", side="right")
        self.log_text.configure(yscrollcommand=sb.set)

        # Log color tags
        self.log_text.tag_configure(
            "t_stage",  foreground=self._accent_hi, font=("Consolas", 9, "bold"))
        self.log_text.tag_configure(
            "t_done",   foreground=self._good_green, font=("Consolas", 9, "bold"))
        self.log_text.tag_configure(
            "t_error",  foreground=self._err_red, font=("Consolas", 9, "bold"))
        self.log_text.tag_configure(
            "t_step",   foreground=self._accent)
        self.log_text.tag_configure(
            "t_warn",   foreground=self._warn_amber)
        self.log_text.tag_configure(
            "t_normal", foreground=self._fg)

    # ------------------------------------------------------------------
    # Complexity-aware notebook construction
    # ------------------------------------------------------------------

    # Each entry: (level_min, builder_method_name). The notebook is
    # populated in this order so the tabs read left-to-right in the same
    # sequence regardless of which subset is currently visible.
    _TAB_SPEC = [
        ("medium",  "_build_inputs_tab"),
        ("medium",  "_build_rinex_tab"),
        ("complex", "_build_ppk_tab"),
        ("medium",  "_build_csv_tab"),
        ("medium",  "_build_smoothers_tab"),
        ("medium",  "_build_imu_calib_tab"),
        ("medium",  "_build_viewers_tab"),
        ("medium",  "_build_analysis_tab"),
        ("complex", "_build_t02_tab"),
        ("basic",   "_build_video_only_tab"),
    ]

    @staticmethod
    def _level_meets(current: str, required: str) -> bool:
        order = {lvl: i for i, lvl in enumerate(COMPLEXITY_LEVELS)}
        return order.get(current, 0) >= order.get(required, 0)

    def _build_notebook(self) -> None:
        nb = ttk.Notebook(self._nb_host)
        nb.pack(fill="both", expand=True)
        self._nb = nb
        level = self._complexity.get()
        for required, method_name in self._TAB_SPEC:
            # The media-only tab carries the lowest required level so
            # ``basic`` still sees it even when every higher-required tab
            # is filtered out.
            if self._level_meets(level, required):
                getattr(self, method_name)(nb)

    def _on_complexity_changed(self) -> None:
        new_level = self._complexity.get()
        _set_complexity(new_level)
        # Rebuild the notebook in place. The widgets inside the old notebook
        # are children of it; destroying the notebook clears them all.
        for child in list(self._nb_host.children.values()):
            child.destroy()
        self._build_notebook()
        self._refresh_status_hint()
        self._log(f"[mode] Switched to {new_level} — tabs reconfigured.")

    _MODE_HINTS = {
        "basic":   "Mode: BASIC — frame extraction only.",
        "medium":  "Mode: MEDIUM — RINEX + frames + viewers (no PPK).",
        "complex": "Mode: COMPLEX — full PPK pipeline + lab tools.",
    }

    def _refresh_status_hint(self) -> None:
        hint = self._MODE_HINTS.get(
            self._complexity.get(), "Tip: tabs flow left → right"
        )
        if hasattr(self, "_status_hint") and self._status_hint.winfo_exists():
            self._status_hint.configure(text=hint)

    def _make_scrollable_tab(self, nb: ttk.Notebook, text: str) -> ttk.Frame:
        """Add a tab whose contents scroll vertically.

        Wraps a :class:`ScrollableFrame` inside the notebook page so any
        builder that fills the returned sample gets scrolling for free —
        critical when the user runs the app at < 900 px tall.
        """
        from ._ui_helpers import ScrollableFrame
        page = ttk.Frame(nb)
        nb.add(page, text=text)
        sf = ScrollableFrame(page)
        sf.pack(fill="both", expand=True)
        return sf.body

    def _build_inputs_tab(self, nb: ttk.Notebook) -> None:
        f = self._make_scrollable_tab(nb, "Inputs")

        self.var_raw  = tk.StringVar()
        self.var_arnx = tk.StringVar(value=DEFAULT_ANDROID_RINEX_SRC)
        self.var_out  = tk.StringVar()

        grp = ttk.LabelFrame(f, text="Source Files")
        grp.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(10, 4))
        grp.columnconfigure(1, weight=1)

        raw_label = "RAW folder  (4 files)"
        if self._dnd_files is not None:
            raw_label = "RAW folder  (4 files — drop folder here)"
        raw_entry = self._row_path(grp, 0, label=raw_label,
                       var=self.var_raw, kind="dir",
                       on_change=self._on_raw_changed)
        self._register_dnd_folder(raw_entry, self.var_raw, self._on_raw_changed)
        self._row_path(grp, 1, label="android_rinex/src  (optional, vendored fallback)",
                       var=self.var_arnx, kind="dir",
                       on_change=self._on_arnx_changed)
        self._row_path(grp, 2, label="Output folder",
                       var=self.var_out, kind="dir",
                       on_change=self._on_out_changed)

        ttk.Separator(f, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=6,
        )

        info_grp = ttk.LabelFrame(f, text="Detected Files")
        info_grp.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
        info_grp.columnconfigure(0, weight=1)
        self.detected = ttk.Label(
            info_grp, text="(no RAW folder selected)", foreground="#888",
            wraplength=900,
        )
        self.detected.grid(row=0, column=0, sticky="w", padx=10, pady=6)

        f.columnconfigure(1, weight=1)

    def _build_rinex_tab(self, nb: ttk.Notebook) -> None:
        f = self._make_scrollable_tab(nb, "RINEX")

        self.var_skip_edit   = tk.BooleanVar(value=True)
        self.var_fix_bias    = tk.BooleanVar(value=True)
        self.var_marker      = tk.StringVar(value="UNKN")
        self.var_observer    = tk.StringVar(value="UNKN")
        self.var_agency      = tk.StringVar(value="UNKN")
        self.var_filter_mode = tk.StringVar(value="sync")
        self.var_keep_level  = tk.StringVar(value="relaxed")
        self.var_obs_path    = tk.StringVar()

        # Options group
        grp1 = ttk.LabelFrame(f, text="Conversion Options")
        grp1.grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(10, 4))
        grp1.columnconfigure(1, weight=1); grp1.columnconfigure(3, weight=1)

        cb_skip = ttk.Checkbutton(grp1, text="--skip-edit  (skip pseudorange range check)",
                                  variable=self.var_skip_edit)
        cb_skip.grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 2))
        _Tooltip(cb_skip, "Skips the pseudorange range edit step in gnsslogger_to_rnx.\n"
                 "Recommended: most Android logs include marginal measurements that pass with --skip-edit.")

        cb_bias = ttk.Checkbutton(grp1, text="--fix-bias  (hold first FullBiasNanos)",
                                  variable=self.var_fix_bias)
        cb_bias.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 6))
        _Tooltip(cb_bias, "Pins FullBiasNanos to the first sample to prevent clock resets\n"
                 "causing RINEX time jumps. Recommended for Android GNSS Logger output.")

        ttk.Label(grp1, text="marker").grid(row=0, column=2, sticky="e", padx=(16, 4))
        ttk.Entry(grp1, textvariable=self.var_marker, width=12).grid(
            row=0, column=3, sticky="w", padx=(0, 8), pady=(6, 2))

        ttk.Label(grp1, text="observer").grid(row=1, column=2, sticky="e", padx=(16, 4))
        ttk.Entry(grp1, textvariable=self.var_observer, width=12).grid(
            row=1, column=3, sticky="w", padx=(0, 8))

        ttk.Label(grp1, text="agency").grid(row=2, column=0, sticky="e", padx=(8, 4), pady=(2, 6))
        ttk.Entry(grp1, textvariable=self.var_agency, width=12).grid(
            row=2, column=1, sticky="w", padx=(0, 16), pady=(2, 6))

        ttk.Label(grp1, text="filter mode").grid(row=2, column=2, sticky="e", padx=(16, 4))
        ttk.Combobox(grp1, textvariable=self.var_filter_mode,
                     values=["sync", "trck"], state="readonly", width=10).grid(
            row=2, column=3, sticky="w", padx=(0, 8), pady=(2, 6))

        ttk.Label(grp1, text="strictness").grid(row=3, column=0, sticky="e", padx=(8, 4), pady=(2, 6))
        cb_keep = ttk.Combobox(
            grp1, textvariable=self.var_keep_level,
            values=["strict", "relaxed", "permissive"],
            state="readonly", width=12,
        )
        cb_keep.grid(row=3, column=1, sticky="w", padx=(0, 16), pady=(2, 6))
        _Tooltip(
            cb_keep,
            "Signal-quality strictness for the Raw measurement filter.\n"
            "  strict      = Google decimeter defaults (CNo>=20, SVT-unc<=500 ns,\n"
            "                drop multipath, drop ADR cycle slip).\n"
            "  relaxed     = CNo>=15, SVT-unc<=1 us, ignore Android multipath flag.\n"
            "  permissive  = CNo>=10, SVT-unc<=2 us, no multipath / slip / ADR-invalid\n"
            "                checks. Use when the recording is the only one you'll get.\n"
            "Hard filters (code lock, TOW/TOD, known constellation) are always enforced."
        )

        # Output group
        grp2 = ttk.LabelFrame(f, text="RINEX Output")
        grp2.grid(row=1, column=0, columnspan=4, sticky="ew", padx=8, pady=4)
        grp2.columnconfigure(1, weight=1)

        ttk.Label(grp2, text=".obs output path").grid(row=0, column=0, sticky="e", padx=8, pady=6)
        ttk.Entry(grp2, textvariable=self.var_obs_path).grid(
            row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(grp2, text="Browse...", command=self._pick_obs).grid(
            row=0, column=2, sticky="w", padx=(0, 8))

        btn_rinex = ttk.Button(f, text="Convert to RINEX", style="Accent.TButton",
                               command=self._run_rinex)
        btn_rinex.grid(row=2, column=0, columnspan=2, padx=8, pady=10, sticky="w")
        self._buttons.append(btn_rinex)

        # Post-processing group
        grp3 = ttk.LabelFrame(f, text="PPK  —  External Processing")
        grp3.grid(row=3, column=0, columnspan=4, sticky="ew", padx=8, pady=4)
        grp3.columnconfigure(1, weight=1)

        ttk.Label(
            grp3,
            text=(
                "Run PPK externally (RTKLIB / RTKPost / RTKEXPLORER) on the .obs above "
                "with your base station .obs / .nav.  "
                "When you have the resulting .pos file, load it here to continue."
            ),
            foreground="#888", wraplength=860,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(6, 4))

        self.var_pos_path = tk.StringVar()
        self._row_path(
            grp3, 1, label=".pos file  (RTKLIB output)",
            var=self.var_pos_path, kind="file_open",
            file_types=[("RTKLIB pos", "*.pos"), ("All", "*.*")],
            on_change=self._on_pos_changed,
        )

        f.columnconfigure(1, weight=1)

    def _build_ppk_tab(self, nb: ttk.Notebook) -> None:
        """Post-processing panel wrapping the external solver binary.

        The user supplies (or auto-detects) a subject .obs, a base .obs, one or
        more nav / SP3 / EPH files, and a config .conf. The button runs::

            <solver> -k <config> -o <out.pos> <subject> <base> <nav1> [<nav2>...]

        On success the resulting .pos is wired back into the pipeline state
        (``paths.pos_path`` + the CSV tab's .pos field) so the next stage is
        ready immediately.
        """
        f = self._make_scrollable_tab(nb, "PPK")

        self.var_ppk_rover    = tk.StringVar()
        self.var_ppk_base     = tk.StringVar()
        self.var_ppk_nav      = tk.StringVar()
        self.var_ppk_config   = tk.StringVar()
        self.var_ppk_output   = tk.StringVar()
        # Default the solver binary path — resolve at runtime via lab_tools chain
        # (env var → vendor/rtklib → PATH). Leave empty here so the user
        # sees "not set" and can pick the binary via the file picker.
        try:
            from . import lab_tools as _lt
            _default_rnx = str(_lt.resolve_tool("rnx2rtkp"))
        except Exception:
            _default_rnx = ""
        self.var_ppk_rnxexe = tk.StringVar(value=_default_rnx)

        ttk.Label(
            f,
            text=(
                "Run PPK with RTKLIB's rnx2rtkp.exe.  "
                "Rover auto-fills from the RINEX step (you can override).  "
                "Drop a config (.conf) prepared in RTKPost or the "
                "gnss_automation project — every processing option comes "
                "from there, so this panel never hides settings from you.  "
                "Output .pos is wired straight into the Frames + CSV tab."
            ),
            foreground="#888", wraplength=920,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        # ── Group 1: Files ──────────────────────────────────────────────────
        g_files = ttk.LabelFrame(f, text="Input Files")
        g_files.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        g_files.columnconfigure(1, weight=1)

        rover_label = "Rover .obs  (auto from RINEX step)"
        if self._dnd_files is not None:
            rover_label = "Rover .obs  (auto / drop file here)"
        rover_entry = self._row_path(
            g_files, 0, label=rover_label,
            var=self.var_ppk_rover, kind="file_open",
            file_types=[("RINEX OBS", "*.obs *.??o"), ("All", "*.*")],
            on_change=self._on_ppk_rover_changed,
        )
        self._register_dnd_video(rover_entry, self.var_ppk_rover,
                                 self._on_ppk_rover_changed)

        base_label = "Base .obs"
        if self._dnd_files is not None:
            base_label = "Base .obs  (drop file here)"
        base_entry = self._row_path(
            g_files, 1, label=base_label,
            var=self.var_ppk_base, kind="file_open",
            file_types=[("RINEX OBS", "*.obs *.??o"), ("All", "*.*")],
            on_change=self._on_ppk_base_changed,
        )
        self._register_dnd_video(base_entry, self.var_ppk_base,
                                 self._on_ppk_base_changed)

        # Multi-line nav list with a smart auto-detect button
        ttk.Label(g_files, text="Nav / Eph files\n(one per line)").grid(
            row=2, column=0, sticky="ne", padx=8, pady=4)
        nav_holder = ttk.Frame(g_files)
        nav_holder.grid(row=2, column=1, sticky="ew", padx=8, pady=4)
        nav_holder.columnconfigure(0, weight=1)
        self._ppk_nav_text = tk.Text(
            nav_holder, height=4, bg="#1a2540", fg="#dde4f0",
            insertbackground="#4cc9f0", wrap="none", font=("Consolas", 9),
            relief="flat", borderwidth=1, highlightthickness=1,
            highlightbackground="#28375a",
        )
        self._ppk_nav_text.grid(row=0, column=0, sticky="ew")
        _Tooltip(self._ppk_nav_text,
                 "One nav/ephemeris file per line.\n"
                 "Accepted: .nav .gnav .hnav .lnav .sp3 .eph .clk .ionex "
                 "and RINEX 2.xx variants (e.g. base.24n / base.24g).\n"
                 "Wildcards work — wrap in double quotes if expanded by shell.")

        nav_btns = ttk.Frame(g_files)
        nav_btns.grid(row=2, column=2, sticky="nw", padx=(0, 8))
        ttk.Button(nav_btns, text="Auto-detect",
                   command=self._ppk_autodetect_nav).grid(row=0, column=0, pady=2)
        ttk.Button(nav_btns, text="Add file…",
                   command=self._ppk_add_nav).grid(row=1, column=0, pady=2)
        ttk.Button(nav_btns, text="Clear",
                   command=lambda: self._ppk_nav_text.delete("1.0", "end")
                   ).grid(row=2, column=0, pady=2)
        self._register_dnd_nav(self._ppk_nav_text)

        # Config file row with preset combobox + view button
        ttk.Label(g_files, text="Config  (.conf)").grid(
            row=3, column=0, sticky="e", padx=8, pady=4)
        cfg_row = ttk.Frame(g_files)
        cfg_row.grid(row=3, column=1, columnspan=2, sticky="ew", padx=8, pady=4)
        cfg_row.columnconfigure(0, weight=1)
        cfg_entry = ttk.Entry(cfg_row, textvariable=self.var_ppk_config)
        cfg_entry.grid(row=0, column=0, sticky="ew")
        self._register_dnd_video(cfg_entry, self.var_ppk_config, None)

        presets = ppk_stage.list_config_files()
        preset_names = ["(pick preset)"] + [p.name for p in presets]
        self._ppk_preset_paths = {p.name: p for p in presets}
        self._ppk_preset_cb = ttk.Combobox(
            cfg_row, values=preset_names, state="readonly", width=22,
        )
        self._ppk_preset_cb.current(0)
        self._ppk_preset_cb.grid(row=0, column=1, padx=(6, 0))
        self._ppk_preset_cb.bind(
            "<<ComboboxSelected>>", lambda _e: self._ppk_pick_preset()
        )
        _Tooltip(self._ppk_preset_cb,
                 f"Presets from {ppk_stage.DEFAULT_RTKLIB_DIR}.\n"
                 "Pick one to populate the Config field, then edit if needed.")
        ttk.Button(cfg_row, text="Browse…",
                   command=self._ppk_browse_config).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(cfg_row, text="View",
                   command=self._ppk_view_config).grid(row=0, column=3, padx=(6, 0))

        # Output .pos row
        out_label = "Output .pos"
        out_entry = self._row_path(
            g_files, 4, label=out_label,
            var=self.var_ppk_output, kind="file_save",
            file_types=[("RTKLIB pos", "*.pos"), ("All", "*.*")],
            on_change=None,
        )
        del out_entry  # silence unused

        # ── Group 2: Engine ─────────────────────────────────────────────────
        g_eng = ttk.LabelFrame(f, text="RTKLIB Executable")
        g_eng.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        g_eng.columnconfigure(1, weight=1)
        ttk.Label(g_eng, text="rnx2rtkp.exe").grid(
            row=0, column=0, sticky="e", padx=8, pady=6)
        ttk.Entry(g_eng, textvariable=self.var_ppk_rnxexe).grid(
            row=0, column=1, sticky="ew", padx=8)

        def _browse_exe() -> None:
            p = filedialog.askopenfilename(
                filetypes=[
                    ("Executables", "*.exe"),
                    ("All", "*.*"),
                ],
                initialdir=str(ppk_stage.DEFAULT_RTKLIB_DIR),
            )
            if p:
                self.var_ppk_rnxexe.set(p)

        ttk.Button(g_eng, text="Browse…", command=_browse_exe).grid(
            row=0, column=2, sticky="w", padx=(0, 8))
        _Tooltip(
            g_eng,
            f"Defaults to {ppk_stage.DEFAULT_RTKLIB_DIR}\\rnx2rtkp.exe.\n"
            "You can also point the RNX2RTKP environment variable at any "
            "rnx2rtkp build to override this without editing the GUI."
        )

        # ── Group 3: Reference input Position ─────────────────────────────────
        self._build_ppk_base_position_group(f, row=3)

        # ── Buttons ─────────────────────────────────────────────────────────
        btn_row = ttk.Frame(f)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 8))

        btn_run = ttk.Button(btn_row, text="Run PPK", style="Accent.TButton",
                             command=self._run_ppk)
        btn_run.grid(row=0, column=0, padx=(0, 6))
        btn_show = ttk.Button(btn_row, text="Show command",
                              command=self._ppk_show_command)
        btn_show.grid(row=0, column=1, padx=6)
        btn_open = ttk.Button(btn_row, text="Open .pos folder ↗",
                              command=self._ppk_open_output)
        btn_open.grid(row=0, column=2, padx=6)
        btn_mm = ttk.Button(btn_row, text="Multi-mask PPK (elevation sweep)",
                            command=self._run_multimask_ppk)
        btn_mm.grid(row=0, column=3, padx=6)
        self._buttons.append(btn_run)
        self._buttons.append(btn_mm)

        if not hasattr(self, "var_auto_build"):
            self.var_auto_build = tk.BooleanVar(value=True)
        cb_auto_build = ttk.Checkbutton(
            f, text="Auto-build smoothers + viewers after PPK completes",
            variable=self.var_auto_build,
        )
        cb_auto_build.grid(row=5, column=0, columnspan=2, sticky="w",
                           padx=10, pady=(0, 2))
        _Tooltip(
            cb_auto_build,
            "When ON (default): as soon as PPK (or Multi-mask PPK) finishes,\n"
            "the GUI automatically runs the selected Smoothers and builds the\n"
            "velocity + geo viewers using the .pos / sensors / output folder\n"
            "that were just produced -- no tab-switching or extra clicks.\n"
            "Each step is independent; a failure in one does not block the\n"
            "others or the GUI. Turn OFF to drive every stage manually.",
        )

        ttk.Label(
            f,
            text=(
                "After a successful run the .pos file is auto-loaded into "
                "Frames + CSV -- just switch tabs and click 'Build Georef "
                "CSV' or 'Run both'. With auto-build ON, Smoothers + the "
                "Velocity/Geo viewers also run automatically."
            ),
            foreground="#4cc9f0", font=("Segoe UI", 8), wraplength=900,
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

    def _build_ppk_base_position_group(self, parent: ttk.Frame, *, row: int) -> None:
        """Reference input coordinate input.

        Leaving ``Override`` un-checked means the config file's existing
        ``ant2-*`` keys are used as-is (typical when the .conf already
        contains the right base position).  Ticking the checkbox swaps in
        a patched copy of the config with the user-supplied position.
        """
        g = ttk.LabelFrame(parent, text="Base Station Position")
        g.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        g.columnconfigure(1, weight=1)
        g.columnconfigure(3, weight=1)

        self.var_ppk_base_override = tk.BooleanVar(value=False)
        self.var_ppk_base_format = tk.StringVar(value="dd")
        self.var_ppk_base_a = tk.StringVar()
        self.var_ppk_base_b = tk.StringVar()
        self.var_ppk_base_c = tk.StringVar()
        self.var_ppk_base_h = tk.StringVar(value="0.0")
        self.var_ppk_base_zone = tk.StringVar()  # Grid only

        cb = ttk.Checkbutton(
            g, text="Override base position (patches the .conf for this run)",
            variable=self.var_ppk_base_override,
            command=self._ppk_base_pos_refresh_state,
        )
        cb.grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(6, 2))
        _Tooltip(
            cb,
            "When ticked, the GUI writes a temp .conf next to the output .pos\n"
            "with ant2-postype=xyz and ant2-pos1/2/3 set from the values below.\n"
            "Use this when your preset config doesn't already contain the\n"
            "exact base position, or when you need to swap base receivers\n"
            "between runs without editing the .conf file.",
        )

        ttk.Label(g, text="Format").grid(row=1, column=0, sticky="e", padx=(8, 4))
        cb_fmt = ttk.Combobox(
            g, textvariable=self.var_ppk_base_format,
            values=["dd", "dms", "utm", "ecef"],
            state="readonly", width=8,
        )
        cb_fmt.grid(row=1, column=1, sticky="w", pady=4)
        cb_fmt.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._ppk_base_pos_refresh_state(),
        )
        _Tooltip(
            cb_fmt,
            "dd   — decimal degrees: lat, lon, height (m)\n"
            "dms  — degrees/min/sec per field (e.g. '47 07 24.4 N'), height in m\n"
            "utm  — UTM zone + letter (e.g. '10T'), easting (m), northing (m), height (m)\n"
            "ecef — Earth-centred XYZ in metres, straight into the .conf",
        )

        self._ppk_base_lbl_a = ttk.Label(g, text="Latitude")
        self._ppk_base_lbl_a.grid(row=2, column=0, sticky="e", padx=(8, 4), pady=2)
        self._ppk_base_ent_a = ttk.Entry(g, textvariable=self.var_ppk_base_a, width=24)
        self._ppk_base_ent_a.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=2)

        self._ppk_base_lbl_b = ttk.Label(g, text="Longitude")
        self._ppk_base_lbl_b.grid(row=2, column=2, sticky="e", padx=(8, 4), pady=2)
        self._ppk_base_ent_b = ttk.Entry(g, textvariable=self.var_ppk_base_b, width=24)
        self._ppk_base_ent_b.grid(row=2, column=3, sticky="ew", padx=(0, 8), pady=2)

        self._ppk_base_lbl_c = ttk.Label(g, text="Height (m)")
        self._ppk_base_lbl_c.grid(row=3, column=0, sticky="e", padx=(8, 4), pady=2)
        self._ppk_base_ent_c = ttk.Entry(g, textvariable=self.var_ppk_base_c, width=24)
        self._ppk_base_ent_c.grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=2)

        self._ppk_base_lbl_zone = ttk.Label(g, text="UTM zone")
        self._ppk_base_lbl_zone.grid(row=3, column=2, sticky="e", padx=(8, 4), pady=2)
        self._ppk_base_ent_zone = ttk.Entry(g, textvariable=self.var_ppk_base_zone, width=12)
        self._ppk_base_ent_zone.grid(row=3, column=3, sticky="w", padx=(0, 8), pady=2)

        self._ppk_base_pos_refresh_state()

    def _ppk_base_pos_refresh_state(self) -> None:
        """Enable/disable fields and relabel based on the chosen format."""
        on = bool(self.var_ppk_base_override.get())
        fmt = self.var_ppk_base_format.get()
        labels = {
            "dd":   ("Latitude (deg)", "Longitude (deg)", "Height (m)"),
            "dms":  ("Latitude (DMS, e.g. 47 07 24.4 N)",
                     "Longitude (DMS, e.g. 122 39 15.5 W)",
                     "Height (m)"),
            "utm":  ("Easting (m)", "Northing (m)", "Height (m)"),
            "ecef": ("X (m)", "Y (m)", "Z (m)"),
        }.get(fmt, ("a", "b", "c"))
        self._ppk_base_lbl_a.configure(text=labels[0])
        self._ppk_base_lbl_b.configure(text=labels[1])
        self._ppk_base_lbl_c.configure(text=labels[2])

        state_ent = "normal" if on else "disabled"
        for w in (
            self._ppk_base_ent_a, self._ppk_base_ent_b,
            self._ppk_base_ent_c, self._ppk_base_ent_zone,
        ):
            w.configure(state=state_ent)
        zone_visible = on and fmt == "utm"
        zone_state = "normal" if zone_visible else "disabled"
        self._ppk_base_lbl_zone.configure(state=("normal" if zone_visible else "disabled"))
        self._ppk_base_ent_zone.configure(state=zone_state)

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def _register_dnd_nav(self, text_widget: tk.Text) -> None:
        if self._dnd_files is None:
            return

        def _handle(event: "tk.Event") -> None:  # type: ignore[type-arg]
            data = event.data or ""
            paths = self.root.tk.splitlist(data)  # type: ignore[attr-defined]
            added: list[str] = []
            for raw in paths:
                p = Path(raw)
                if p.is_dir():
                    added.extend(str(x) for x in ppk_stage.detect_nav_files(p))
                elif p.is_file():
                    added.append(str(p))
            if added:
                current = text_widget.get("1.0", "end").strip()
                existing = set(current.splitlines()) if current else set()
                merged = (current.splitlines() if current else []) + [
                    a for a in added if a not in existing
                ]
                text_widget.delete("1.0", "end")
                text_widget.insert("1.0", "\n".join(merged) + "\n")

        try:
            text_widget.drop_target_register(self._dnd_files)  # type: ignore[attr-defined]
            text_widget.dnd_bind("<<Drop>>", _handle)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _ppk_get_nav_files(self) -> list[Path]:
        raw = self._ppk_nav_text.get("1.0", "end").strip()
        if not raw:
            return []
        return [Path(line.strip()) for line in raw.splitlines() if line.strip()]

    def _ppk_set_nav_files(self, files: list[Path]) -> None:
        self._ppk_nav_text.delete("1.0", "end")
        if files:
            self._ppk_nav_text.insert("1.0", "\n".join(str(p) for p in files) + "\n")

    def _ppk_autodetect_nav(self) -> None:
        dirs: list[Path] = []
        for s in (self.var_ppk_rover.get(), self.var_ppk_base.get()):
            s = s.strip()
            if s:
                p = Path(s)
                if p.is_file():
                    dirs.append(p.parent)
        if not dirs:
            messagebox.showinfo(
                "No reference dirs",
                "Pick a rover or base .obs first so I know where to search.",
            )
            return
        found = ppk_stage.detect_nav_files(*dirs)
        if not found:
            messagebox.showinfo(
                "No nav files found",
                "Scanned:\n  " + "\n  ".join(str(d) for d in dirs)
                + "\n\nNothing matching .nav/.sp3/.eph/.clk/RINEX 2.xx patterns.",
            )
            return
        # Merge with existing (avoid duplicates)
        existing = self._ppk_get_nav_files()
        seen = {p.resolve() for p in existing}
        merged = list(existing)
        for p in found:
            if p.resolve() not in seen:
                merged.append(p)
                seen.add(p.resolve())
        self._ppk_set_nav_files(merged)
        self._log(f"[ppk] auto-detected {len(found)} nav file(s) "
                  f"({len(merged) - len(existing)} new)")

    def _ppk_add_nav(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Pick nav / ephemeris files",
            filetypes=[
                ("Nav / Eph", "*.nav *.gnav *.hnav *.lnav *.sp3 *.eph *.clk "
                              "*.ionex *.??n *.??g *.??p *.??l"),
                ("All", "*.*"),
            ],
        )
        if not paths:
            return
        existing = self._ppk_get_nav_files()
        seen = {p.resolve() for p in existing}
        for raw in paths:
            p = Path(raw)
            if p.resolve() not in seen:
                existing.append(p)
                seen.add(p.resolve())
        self._ppk_set_nav_files(existing)

    def _ppk_pick_preset(self) -> None:
        name = self._ppk_preset_cb.get()
        path = self._ppk_preset_paths.get(name)
        if path is not None:
            self.var_ppk_config.set(str(path))

    def _ppk_browse_config(self) -> None:
        p = filedialog.askopenfilename(
            title="Pick RTKLIB config",
            filetypes=[("RTKLIB config", "*.conf"), ("All", "*.*")],
            initialdir=str(ppk_stage.DEFAULT_RTKLIB_DIR),
        )
        if p:
            self.var_ppk_config.set(p)

    def _ppk_view_config(self) -> None:
        s = self.var_ppk_config.get().strip()
        if not s:
            messagebox.showinfo("No config", "Pick a config file first.")
            return
        p = Path(s)
        if not p.is_file():
            messagebox.showerror("Missing", f"Not a file:\n{p}")
            return
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            messagebox.showerror("Read error", str(e))
            return
        win = tk.Toplevel(self.root)
        win.title(f"Config — {p.name}")
        win.geometry("780x520")
        win.configure(bg="#0c1020")
        t = tk.Text(win, bg="#090e1c", fg="#c0cce0",
                    insertbackground="#c0cce0", wrap="none",
                    font=("Consolas", 9), relief="flat")
        t.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(win, orient="vertical", command=t.yview)
        sb.pack(fill="y", side="right")
        t.configure(yscrollcommand=sb.set, state="normal")
        t.insert("1.0", txt)
        t.configure(state="disabled")

    def _ppk_open_output(self) -> None:
        s = self.var_ppk_output.get().strip()
        if not s:
            messagebox.showinfo("No output", "Pick (or run with) an output .pos first.")
            return
        p = Path(s).parent
        if not p.is_dir():
            messagebox.showerror("Missing", f"Not a folder:\n{p}")
            return
        self._open_folder(p)

    def _ppk_show_command(self) -> None:
        try:
            cmd = self._build_ppk_cmd_preview()
        except Exception as e:
            messagebox.showerror("Cannot preview", str(e))
            return
        win = tk.Toplevel(self.root)
        win.title("rnx2rtkp command preview")
        win.geometry("900x180")
        win.configure(bg="#0c1020")
        t = tk.Text(win, bg="#090e1c", fg="#c0cce0",
                    insertbackground="#c0cce0", wrap="word",
                    font=("Consolas", 9), height=6)
        t.pack(fill="both", expand=True, padx=8, pady=8)
        t.insert("1.0", " ".join(cmd))
        t.configure(state="disabled")

    def _build_ppk_cmd_preview(self) -> list[str]:
        exe = self.var_ppk_rnxexe.get().strip() or "rnx2rtkp"
        rover = self.var_ppk_rover.get().strip()
        base = self.var_ppk_base.get().strip()
        cfg = self.var_ppk_config.get().strip()
        out = self.var_ppk_output.get().strip()
        navs = [str(p) for p in self._ppk_get_nav_files()]
        if not (rover and base and cfg and out and navs):
            raise ValueError(
                "Fill rover, base, config, output, and at least one nav file."
            )
        cmd = [exe, "-k", cfg, "-o", out, rover, base, *navs]
        return cmd

    def _on_ppk_rover_changed(self) -> None:
        s = self.var_ppk_rover.get().strip()
        if not s:
            return
        # If output not yet set, default to <rover_dir>/<rover_stem>.pos
        if not self.var_ppk_output.get().strip():
            p = Path(s)
            self.var_ppk_output.set(str(p.with_suffix(".pos")))

    def _on_ppk_base_changed(self) -> None:
        # Hook reserved for future autodetect; deliberate no-op for now.
        pass

    def _sync_ppk_rover_from_obs(self) -> None:
        """Called after a Interchange-format run: pre-fill the Post-processing subject field."""
        if self.paths.obs_path is None:
            return
        # Only overwrite if user hasn't typed something else.
        current = self.var_ppk_rover.get().strip()
        if current and Path(current).resolve() != self.paths.obs_path.resolve():
            return
        self.var_ppk_rover.set(str(self.paths.obs_path))
        self._on_ppk_rover_changed()
        # Best-effort nav autodetection in the subject directory.
        try:
            found = ppk_stage.detect_nav_files(self.paths.obs_path.parent)
        except Exception:
            found = []
        if found and not self._ppk_get_nav_files():
            self._ppk_set_nav_files(found)
            self._log(f"[ppk] auto-detected {len(found)} nav file(s) "
                      f"in {self.paths.obs_path.parent}")

    def _collect_ppk_base_override(self) -> Optional[tuple[float, float, float]]:
        """Read the Reference input group and return Cartesian XYZ XYZ or ``None``.

        Returns ``None`` when the override checkbox is off, signalling to
        :func:`ppk_stage.run` that the config file's existing base
        position should be used unchanged. Raises ``ValueError`` with a
        human-readable message on malformed input.
        """
        if not bool(self.var_ppk_base_override.get()):
            return None
        fmt = self.var_ppk_base_format.get()
        a = self.var_ppk_base_a.get().strip()
        b = self.var_ppk_base_b.get().strip()
        c = self.var_ppk_base_c.get().strip()
        z = self.var_ppk_base_zone.get().strip()
        if fmt == "dd":
            if not (a and b and c):
                raise ValueError("Decimal degrees: fill latitude, longitude and height.")
            return coords_mod.parse_dd(f"{a} {b} {c}")
        if fmt == "dms":
            if not (a and b and c):
                raise ValueError("DMS: fill latitude, longitude and height.")
            return coords_mod.parse_dms(a, b, c)
        if fmt == "utm":
            if not (z and a and b and c):
                raise ValueError("UTM: fill zone, easting, northing and height.")
            return coords_mod.parse_utm(z, a, b, c)
        if fmt == "ecef":
            if not (a and b and c):
                raise ValueError("ECEF: fill X, Y and Z (metres).")
            return coords_mod.parse_ecef(f"{a} {b} {c}")
        raise ValueError(f"Unknown base-position format '{fmt}'")

    def _run_ppk(self) -> None:
        try:
            cmd = self._build_ppk_cmd_preview()
        except Exception as e:
            messagebox.showerror("Missing inputs", str(e))
            return
        rover = Path(self.var_ppk_rover.get().strip())
        base = Path(self.var_ppk_base.get().strip())
        cfg = Path(self.var_ppk_config.get().strip())
        out = Path(self.var_ppk_output.get().strip())
        navs = self._ppk_get_nav_files()
        exe_raw = self.var_ppk_rnxexe.get().strip()
        exe_path: Optional[Path] = Path(exe_raw) if exe_raw else None
        try:
            base_ecef = self._collect_ppk_base_override()
        except ValueError as e:
            messagebox.showerror("Base position", str(e))
            return
        del cmd  # only used as up-front validation

        def go() -> None:
            res = ppk_stage.run(
                rover_obs=rover,
                base_obs=base,
                nav_files=navs,
                config_file=cfg,
                output_pos=out,
                rnx2rtkp_exe=exe_path,
                base_ecef_xyz=base_ecef,
                log=self._log,
            )
            # Auto-wire result into CSV stage so user can switch tabs and go.
            # Mutate self.paths on the main thread to keep Tk single-thread
            # discipline (Python attr assignment is atomic, but Tk widgets
            # downstream that read the path will see a consistent value).
            self.root.after(
                0, lambda r=res: (
                    setattr(self.paths, "pos_path", r.pos_path),
                    self.var_pos_path.set(str(r.pos_path)),
                    self._auto_build_all(),
                ),
            )

        self._run_async(go, "PPK (rnx2rtkp)")

    def _run_multimask_ppk(self) -> None:
        """Run the multi-elevation-mask Post-processing ensemble + GT-free fusion.

        Reuses the Post-processing tab inputs (subject / base / nav / config + optional base
        override). Writes per-mask .pos + fused.pos + disagreement.csv (+ an
        offline HTML report) into a ``multimask`` folder next to the chosen
        Post-processing output. GT-free; thread-safe via ``_run_async`` + the log queue.
        """
        try:
            self._build_ppk_cmd_preview()   # up-front input validation
        except Exception as e:
            messagebox.showerror("Missing inputs", str(e))
            return
        rover = Path(self.var_ppk_rover.get().strip())
        base = Path(self.var_ppk_base.get().strip())
        cfg = Path(self.var_ppk_config.get().strip())
        navs = self._ppk_get_nav_files()
        exe_raw = self.var_ppk_rnxexe.get().strip()
        exe_path: Optional[Path] = Path(exe_raw) if exe_raw else None
        out_raw = self.var_ppk_output.get().strip()
        if out_raw:
            workdir = Path(out_raw).parent / "multimask"
        else:
            workdir = rover.parent / "multimask"
        try:
            base_ecef = self._collect_ppk_base_override()
        except ValueError as e:
            messagebox.showerror("Base position", str(e))
            return

        def go() -> None:
            workdir.mkdir(parents=True, exist_ok=True)
            conf_used = cfg
            # When the user overrides the base, patch it into a temp conf so the
            # ensemble runs against the user-supplied base position.
            if base_ecef is not None:
                conf_used = ppk_stage.write_patched_config(
                    cfg, workdir / "base_patched.conf",
                    base_ecef_xyz=base_ecef, log=self._log,
                )
            res = multimask_stage.run_multimask_ppk(
                rover, base, navs, conf_used,
                masks=multimask_stage.DEFAULT_MASKS,
                workdir=workdir,
                rnx2rtkp_exe=exe_path,
                log=self._log,
            )
            self._log(f"[multimask] per-mask solutions: "
                      f"{sorted(res.per_mask.keys())}")
            if res.fused_pos is not None:
                self._log(f"[multimask] fused .pos -> {res.fused_pos}")
                self._log(f"[multimask] disagreement CSV -> {res.disagreement_csv}")
                if res.report_html is not None:
                    self._log(f"[multimask] report -> {res.report_html}")
                # Auto-wire the fused path into the CSV stage.
                self.root.after(
                    0, lambda r=res: (
                        setattr(self.paths, "pos_path", r.fused_pos),
                        self.var_pos_path.set(str(r.fused_pos)),
                        self._auto_build_all(),
                    ),
                )
            else:
                self._log("[multimask] no fused output (fewer than 2 masks "
                          "produced a common-epoch solution)")

        self._run_async(go, "Multi-mask PPK (elevation sweep)")

    def _build_csv_tab(self, nb: ttk.Notebook) -> None:
        f = self._make_scrollable_tab(nb, "Frames + CSV")

        self.var_fps                = tk.DoubleVar(value=6.0)
        self.var_pts_name_decimals  = tk.IntVar(value=6)
        self.var_format             = tk.StringVar(value="png")
        self.var_rotation           = tk.StringVar(value="0")
        # Adaptive (Rate-signal + CV) selection options.
        self.var_fps_mode           = tk.StringVar(value="fixed")  # "fixed" | "adaptive"
        self.var_adapt_spacing_m    = tk.DoubleVar(value=2.0)
        self.var_adapt_turn_overlap = tk.DoubleVar(value=0.80)
        self.var_adapt_yawrate      = tk.DoubleVar(value=5.0)
        self.var_adapt_min_dt       = tk.DoubleVar(value=0.10)
        self.var_adapt_max_dt       = tk.DoubleVar(value=30.0)
        self.var_smoothing          = tk.StringVar(value="car")
        self.var_confidence_gate    = tk.StringVar(value="off")
        self.var_xy_sigma           = tk.DoubleVar(value=2.0)
        self.var_z_sigma            = tk.DoubleVar(value=10.0)
        self.var_add_ypr            = tk.BooleanVar(value=True)
        self.var_gravity_orient     = tk.BooleanVar(value=False)
        self.var_imu_fusion         = tk.BooleanVar(value=False)
        self.var_include_alt        = tk.BooleanVar(value=False)
        self.var_z_sigma_override   = tk.DoubleVar(value=30.0)
        # Explicit opt-in to smoothing the altitude (Z) channel. Default OFF:
        # device Z is noisy and smoothing it is a deliberate choice. When on,
        # var_alt_smooth_sigma is the Gaussian window (s) for Z.
        self.var_smooth_alt         = tk.BooleanVar(value=False)
        self.var_alt_smooth_sigma   = tk.DoubleVar(value=30.0)
        self.var_pitch_prior        = tk.StringVar(value="0")
        self.var_roll_prior         = tk.StringVar(value="0")
        self.var_use_pitch_prior    = tk.BooleanVar(value=True)
        self.var_acc_xy             = tk.DoubleVar(value=0.10)
        self.var_acc_z              = tk.DoubleVar(value=0.30)
        self.var_max_gap            = tk.DoubleVar(value=2.0)
        # Client path export (stages.user_export) options.
        # Coordinate-system chooser: default Datum-based + Cartesian XYZ matches
        # user_export.DEFAULT_COORD_SYSTEMS (legacy column set unchanged).
        self.var_exp_coord_geodetic = tk.BooleanVar(value=True)
        self.var_exp_coord_ecef     = tk.BooleanVar(value=True)
        self.var_exp_coord_utm      = tk.BooleanVar(value=False)
        self.var_exp_coord_enu      = tk.BooleanVar(value=False)
        # Height (Z) smoothing for the client export — DEFAULT ON, matching
        # the export_trajectory/export_kml backend default (z_sigma_s=3.0 s).
        self.var_exp_smooth_z       = tk.BooleanVar(value=True)
        self.var_exp_z_sigma_s      = tk.DoubleVar(value=3.0)
        # Time-basis chooser for the client export — which TIME column(s) the
        # CSV carries (user_export ``time_bases``). Default Reference time-only matches
        # user_export.DEFAULT_TIME_BASES so the output stays byte-identical.
        self.var_tb_gpst            = tk.BooleanVar(value=True)
        self.var_tb_utc             = tk.BooleanVar(value=False)
        self.var_tb_audio           = tk.BooleanVar(value=False)
        self.var_tb_iso             = tk.BooleanVar(value=False)
        # Export-source chooser + final-velocity block (user_export F1+F5).
        # Defaults are all neutral so the default client export stays
        # byte-identical: "(as run)" keeps each smoother's own rows, final
        # velocity off, disagreement threshold blank (= None, gate disabled).
        self.var_exp_source         = tk.StringVar(
            value=self.EXPORT_SOURCE_AS_RUN)
        self.var_emit_final_vel     = tk.BooleanVar(value=False)
        self.var_vel_disagree       = tk.StringVar(value="")
        # Variant comparison state.
        # Holds (label -> csv_path) after Build-all-variants completes so the
        # eval / "use best for Coordinate output" button can find them.
        self._variant_csvs: dict[str, Path] = {}

        ttk.Label(
            f,
            text=(
                "Extract lossless frames from the MP4 at the chosen fps, then interpolate "
                "and smooth the PPK trajectory (1 Hz) to every frame for Georef import.  "
                "Output CSV includes ENU Doppler velocity components (ve, vn, vu) per frame."
            ),
            foreground="#888", wraplength=920,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        # ── Group 1: Sample Extraction ────────────────────────────────────────
        g1 = ttk.LabelFrame(f, text="Frame Extraction")
        g1.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 2))
        g1.columnconfigure(1, weight=1)
        g1.columnconfigure(3, weight=1)

        ttk.Label(g1, text="fps").grid(row=0, column=0, sticky="e", padx=(10, 4), pady=6)
        sb_fps = ttk.Spinbox(g1, textvariable=self.var_fps,
                             from_=0.5, to=60.0, increment=0.5, width=8)
        sb_fps.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=6)
        _Tooltip(sb_fps, "Frames per second to extract from the video.\n"
                 "Higher fps → more frames, larger output, longer processing.\n"
                 "Typical: 1–6 fps for vehicle survey, 0.5–2 fps for slow aerial.")

        ttk.Label(g1, text="filename decimals").grid(row=0, column=2, sticky="e", padx=(0, 4))
        sb_dec = ttk.Spinbox(g1, textvariable=self.var_pts_name_decimals,
                             from_=3, to=12, increment=1, width=6)
        sb_dec.grid(row=0, column=3, sticky="w", padx=(0, 10))
        _Tooltip(sb_dec, "Decimal places in the frame filename timestamp.\n"
                 "6 → 1 µs resolution (e.g. frame_1462018987.123456.jpg).")

        ttk.Label(g1, text="image format").grid(row=1, column=0, sticky="e",
                                                padx=(10, 4), pady=(0, 6))
        cb_fmt = ttk.Combobox(g1, textvariable=self.var_format,
                              values=["png", "tiff", "jpeg1"],
                              state="readonly", width=10)
        cb_fmt.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(0, 6))
        _Tooltip(cb_fmt, "Output frame image format.\n"
                 "• png  — lossless, large files (~500 KB/frame)\n"
                 "• tiff — lossless, compatible with older tools\n"
                 "• jpeg1 — ffmpeg -q:v 1, near-lossless, ~10× smaller than png\n"
                 "  (visually identical for georeferencing purposes)")
        ttk.Label(g1, text="png / tiff = lossless   |   jpeg1 = near-lossless  (-q:v 1,  ~10× smaller)",
                  foreground="#777").grid(row=1, column=2, columnspan=2, sticky="w",
                                         padx=4, pady=(0, 6))

        ttk.Label(g1, text="rotation").grid(row=2, column=0, sticky="e",
                                            padx=(10, 4), pady=(0, 8))
        cb_rot = ttk.Combobox(g1, textvariable=self.var_rotation,
                              values=["0", "90", "180", "270"],
                              state="readonly", width=8)
        cb_rot.grid(row=2, column=1, sticky="w", padx=(0, 20), pady=(0, 8))
        _Tooltip(cb_rot, "Clockwise rotation applied to every extracted frame.\n"
                 "  0°  — no rotation (default)\n"
                 "  90° — 90° clockwise  (device held portrait, right side up)\n"
                 " 180° — flip 180° (device upside-down)\n"
                 " 270° — 90° counter-clockwise  (portrait, left side up)\n\n"
                 "A live preview window opens automatically and updates on every change.")
        ttk.Label(g1, text="preview updates automatically",
                  foreground="#4a5a7a", font=("Segoe UI", 8)).grid(
            row=2, column=2, columnspan=2, sticky="w", padx=4, pady=(0, 8))
        self.var_rotation.trace_add("write", lambda *_: self._update_rotation_preview())

        # FPS mode toggle (fixed vs adaptive). The "adaptive" mode uses Post-processing
        # Rate-signal speed for straight-line spacing and Keypoint feature overlap
        # during turns; static stops produce far fewer samples automatically.
        mode_row = ttk.Frame(g1)
        mode_row.grid(row=3, column=0, columnspan=4, sticky="w",
                      padx=8, pady=(2, 8))
        ttk.Label(mode_row, text="extraction mode:",
                  style="Dim.TLabel").grid(row=0, column=0, padx=(0, 8))
        rb_fixed = ttk.Radiobutton(
            mode_row, text="Fixed FPS",
            variable=self.var_fps_mode, value="fixed",
            command=self._on_fps_mode_changed,
        )
        rb_fixed.grid(row=0, column=1, padx=(0, 14))
        rb_adapt = ttk.Radiobutton(
            mode_row, text="Adaptive  (PPK speed + CV overlap)",
            variable=self.var_fps_mode, value="adaptive",
            command=self._on_fps_mode_changed,
        )
        rb_adapt.grid(row=0, column=2, padx=(0, 8))
        _Tooltip(rb_adapt,
                 "Adaptive mode integrates PPK Doppler speed between source "
                 "frames and keeps a new frame each time the cumulative "
                 "along-track distance reaches the spacing target.  During "
                 "turns (heading rate > threshold) the rule switches to an "
                 "ORB-feature overlap check, keeping a new frame as soon as "
                 "≤ N % of the new view overlaps the previous kept frame.\n"
                 "Static stops produce very few frames; the max-interval "
                 "guard still anchors one frame every N seconds.")

        # Adaptive options sub-group — visible always so the user can preview
        # what would happen; only consulted when mode == "adaptive".
        ga = ttk.LabelFrame(f, text="Adaptive Extraction  (used when mode = Adaptive)")
        ga.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 2))
        ga.columnconfigure(1, weight=1)
        ga.columnconfigure(3, weight=1)

        ttk.Label(ga, text="straight spacing (m)").grid(
            row=0, column=0, sticky="e", padx=(10, 4), pady=6)
        sb_sp = ttk.Spinbox(ga, textvariable=self.var_adapt_spacing_m,
                            from_=0.1, to=50.0, increment=0.1, width=8)
        sb_sp.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=6)
        _Tooltip(sb_sp,
                 "Target along-track spacing in meters while driving straight.\n"
                 "Recommended:\n"
                 "  • 1.0–1.5 m for tight building/façade reconstruction\n"
                 "  • 2.0 m   for general street-level survey\n"
                 "  • 4.0+ m  for highway / aerial-style coverage")

        ttk.Label(ga, text="turn overlap target").grid(
            row=0, column=2, sticky="e", padx=(0, 4))
        sb_ov = ttk.Spinbox(ga, textvariable=self.var_adapt_turn_overlap,
                            from_=0.50, to=0.95, increment=0.01, width=8,
                            format="%.2f")
        sb_ov.grid(row=0, column=3, sticky="w", padx=(0, 10))
        _Tooltip(sb_ov,
                 "During turns, the new frame is kept as soon as ORB-feature "
                 "overlap with the previous kept frame drops to this value.\n"
                 "0.80–0.85 is the georeferencing sweet spot.")

        ttk.Label(ga, text="yaw-rate threshold (°/s)").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=(0, 8))
        sb_yr = ttk.Spinbox(ga, textvariable=self.var_adapt_yawrate,
                            from_=0.5, to=60.0, increment=0.5, width=8)
        sb_yr.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(0, 8))
        _Tooltip(sb_yr,
                 "Heading rate (deg/s) above which the turn branch kicks in.\n"
                 "Below this, straight-line distance rule applies.")

        intv_row = ttk.Frame(ga)
        intv_row.grid(row=1, column=2, columnspan=2, sticky="w", padx=(0, 10), pady=(0, 8))
        ttk.Label(intv_row, text="min Δt").grid(row=0, column=0, sticky="e")
        ttk.Spinbox(intv_row, textvariable=self.var_adapt_min_dt,
                    from_=0.01, to=2.0, increment=0.01, width=6,
                    format="%.2f").grid(row=0, column=1, padx=(2, 12))
        ttk.Label(intv_row, text="max Δt").grid(row=0, column=2, sticky="e")
        ttk.Spinbox(intv_row, textvariable=self.var_adapt_max_dt,
                    from_=1.0, to=300.0, increment=1.0, width=6).grid(
            row=0, column=3, padx=(2, 0))
        _Tooltip(intv_row,
                 "Hard time guards (seconds):\n"
                 "  min Δt — never keep two frames closer than this\n"
                 "  max Δt — always anchor one frame after this long without one\n"
                 "         (useful for long stops to keep continuity).")

        ttk.Label(
            ga,
            text="Adaptive mode requires a .pos file and the original "
                 "recording_*.txt time-anchor.",
            foreground=self._fg_dim, font=("Segoe UI", 8),
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 6))

        # ── Group 2: Path Smoothing ────────────────────────────────────
        g2 = ttk.LabelFrame(f, text="Trajectory Smoothing")
        g2.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=2)
        g2.columnconfigure(1, weight=1)
        g2.columnconfigure(3, weight=1)

        ttk.Label(g2, text="profile").grid(row=0, column=0, sticky="e",
                                           padx=(10, 4), pady=6)
        cb_sm = ttk.Combobox(
            g2, textvariable=self.var_smoothing,
            values=["car", "none", "gentle", "aggressive",
                    "custom", "fused-bent",
                    "cv_rts", "cv_rts_pv", "gate_then_cv",
                    "epoch_weighted", "epoch_weighted_v2",
                    "ekf_smoothed", "kalman_simple_cv",
                    "ns_adaptive", "fgo"],
            state="readonly", width=22,
        )
        cb_sm.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=6)
        _Tooltip(cb_sm, "Trajectory shaping profile:\n"
                 "Gaussian (per-axis Gaussian in ENU):\n"
                 "• none       — raw PPK, no smoothing\n"
                 "• gentle     — xy 0.5 s / z 2 s  (walking)\n"
                 "• car        — xy 2 s  / z 10 s  (default road)\n"
                 "• aggressive — xy 5 s  / z 20 s  (highway)\n"
                 "• custom     — use xy / z sigma spinboxes below\n"
                 "• fused-bent — device IMU-fused shape warped onto PPK\n"
                 "\n"
                 "Kalman smoothers (pre-smooth .pos then interpolate):\n"
                 "• cv_rts          — CV+RTS per-axis (position only)\n"
                 "• cv_rts_pv       — CV+RTS position + Doppler velocity\n"
                 "                      (no-video champion: 2.416m)\n"
                 "• gate_then_cv    — Doppler MAD-gate + CV+RTS (2.747m)\n"
                 "• epoch_weighted  — Recipe 1+3 scalar (2.330m)\n"
                 "• epoch_weighted_v2 — 6D Kalman + ZUPT + NHC\n"
                 "• ekf_smoothed    — 9-state EKF+RTS (needs IMU, 2.892m)\n"
                 "• kalman_simple_cv — simple constant-velocity Kalman\n"
                 "\n"
                 "Other:\n"
                 "• ns_adaptive — adaptive Gaussian bandwidth from ns\n"
                 "• fgo         — GTSAM factor-graph (needs gtsam + IMU)\n"
                 "\n"
                 "Kalman/FGO smoothers run BEFORE frame interpolation.\n"
                 "xy/z sigma spinboxes are ignored for those profiles.")

        ttk.Label(g2, text="max interp gap (s)").grid(row=0, column=2, sticky="e",
                                                      padx=(0, 4))
        sb_gap = ttk.Spinbox(g2, textvariable=self.var_max_gap,
                             from_=0.5, to=30.0, increment=0.5, width=8)
        sb_gap.grid(row=0, column=3, sticky="w", padx=(0, 10))
        _Tooltip(sb_gap, "Maximum time gap between consecutive PPK fixes\n"
                 "that the pipeline will interpolate across (seconds).\n"
                 "Gaps larger than this leave the frame with no position in the CSV.")

        ttk.Label(g2, text="xy sigma (s)").grid(row=1, column=0, sticky="e",
                                                padx=(10, 4), pady=(0, 8))
        sb_xys = ttk.Spinbox(g2, textvariable=self.var_xy_sigma,
                             from_=0.0, to=60.0, increment=0.5, width=8)
        sb_xys.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(0, 8))
        _Tooltip(sb_xys, "Horizontal smoothing standard deviation in seconds (custom profile).\n"
                 "Converted to frame-domain samples: σ_samples = xy_sigma × fps.")

        ttk.Label(g2, text="z sigma (s)").grid(row=1, column=2, sticky="e", padx=(0, 4))
        sb_zs = ttk.Spinbox(g2, textvariable=self.var_z_sigma,
                            from_=0.0, to=120.0, increment=1.0, width=8)
        sb_zs.grid(row=1, column=3, sticky="w", padx=(0, 10), pady=(0, 8))
        _Tooltip(sb_zs, "Vertical smoothing standard deviation in seconds (custom profile).\n"
                 "Larger than xy because device altitude is inherently noisier\n"
                 "(VDOP typically 2–3× HDOP, plus multipath on up component).")

        ttk.Label(g2, text="confidence gate").grid(row=2, column=0, sticky="e",
                                                      padx=(10, 4), pady=(0, 8))
        cb_cg = ttk.Combobox(
            g2, textvariable=self.var_confidence_gate,
            values=["off", "sd_h", "combo", "eff_sig"],
            state="readonly", width=18,
        )
        cb_cg.grid(row=2, column=1, sticky="w", padx=(0, 20), pady=(0, 8))
        _Tooltip(cb_cg, "Epoch confidence gate — reject epochs unlikely to meet\n"
                 "a 6 m horizontal accuracy ceiling. Only active with\n"
                 "epoch_weighted_v2 smoother.\n"
                 "\n"
                 "• off     — keep all epochs (default)\n"
                 "• sd_h    — session gate + RTKLIB sd_h < 0.243 m\n"
                 "            (10% retention, validated max 4.75 m)\n"
                 "• combo   — session gate + pred_std + innovation\n"
                 "            (11% retention, validated max 5.58 m)\n"
                 "• eff_sig — effective sigma < 0.727 m\n"
                 "            (8% retention, validated max 5.47 m)\n"
                 "\n"
                 "All strategies are conservative keep/reject gates.\n"
                 "BAD epochs are typically cold-start ambiguity (first\n"
                 "30-50 epochs) and multipath drift clusters.")

        # ── Group 3: Orientation ─────────────────────────────────────────────
        g3 = ttk.LabelFrame(f, text="Orientation  (Yaw / Pitch / Roll)")
        g3.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=2)
        g3.columnconfigure(0, weight=1)

        cb_ypr = ttk.Checkbutton(
            g3,
            text="Add Yaw / Pitch / Roll  "
                 "(yaw from PPK Doppler heading, pitch + roll from device IMU)",
            variable=self.var_add_ypr,
        )
        cb_ypr.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        _Tooltip(cb_ypr, "Attaches orientation columns to each CSV row.\n\n"
                 "Yaw: derived from PPK Doppler velocity (ve, vn) — atan2(ve, vn).\n"
                 "  Low-noise, decoupled from position; NaN when speed < 0.5 m/s.\n\n"
                 "Pitch / Roll: from device IMU (OrientationDeg lines), decimated to\n"
                 "10 Hz and Gaussian-smoothed.")

        cb_grav = ttk.Checkbutton(
            g3,
            text="Gravity orientation at stops  "
                 "(sensors_*.txt — averages accelerometer gravity during static periods; "
                 "absolute pitch/roll, zero gyro drift)",
            variable=self.var_gravity_orient,
        )
        cb_grav.grid(row=1, column=0, sticky="w", padx=10, pady=2)
        _Tooltip(cb_grav, "During stationary periods the accelerometer measures pure gravity\n"
                 "→ exact absolute pitch/roll with no gyro drift.\n\n"
                 "The pipeline detects static intervals from PPK velocity, computes the\n"
                 "gravity vector for each interval, and replaces IMU orientation for all\n"
                 "frames captured while the vehicle is stopped (e.g. at intersections).")

        cb_fus = ttk.Checkbutton(
            g3,
            text="IMU/GNSS fusion  "
                 "(Mahony 200 Hz attitude filter — continuous pitch/roll/yaw; "
                 "requires sensors_*.txt)",
            variable=self.var_imu_fusion,
        )
        cb_fus.grid(row=2, column=0, sticky="w", padx=10, pady=(2, 4))
        _Tooltip(cb_fus, "Runs a Mahony complementary filter fusing gyro + accelerometer + GNSS heading.\n\n"
                 "Gives smooth orientation at IMU rate (~200 Hz) instead of 1 Hz GNSS only.\n"
                 "Yaw seeded from GNSS velocity when speed > 0.5 m/s; self-correcting.\n\n"
                 "Note: EKF position fusion is intentionally disabled — device IMU accel\n"
                 "noise degrades trajectory vs Gaussian smoothing of PPK alone.")

        # Pitch/roll prior row
        prior_row = ttk.Frame(g3)
        prior_row.grid(row=3, column=0, sticky="w", padx=10, pady=(4, 8))

        cb_prior = ttk.Checkbutton(
            prior_row,
            text="Camera pitch/roll prior  (when no IMU data)",
            variable=self.var_use_pitch_prior,
        )
        cb_prior.grid(row=0, column=0, sticky="w")
        _Tooltip(cb_prior,
                 "Writes a constant Pitch and Roll to the CSV for frames that have no\n"
                 "orientation from data log or IMU fusion.\n\n"
                 "CRITICAL for dashcam/forward-facing device:\n"
                 "  Pitch = 0° → camera looks horizontally (correct for dashcam)\n"
                 "  Without this + altitude, Georef can flip the scene upside-down\n"
                 "  ('street appears in the sky').\n\n"
                 "For nadir/drone: set Pitch = -90° (camera looking straight down).")

        ttk.Label(prior_row, text="  Pitch (°):", foreground="#888").grid(
            row=0, column=1, padx=(16, 2))
        sb_pitch = ttk.Spinbox(prior_row, textvariable=self.var_pitch_prior,
                               from_=-90, to=90, increment=1, width=6)
        sb_pitch.grid(row=0, column=2)
        ttk.Label(prior_row, text="  Roll (°):", foreground="#888").grid(
            row=0, column=3, padx=(10, 2))
        sb_roll = ttk.Spinbox(prior_row, textvariable=self.var_roll_prior,
                              from_=-180, to=180, increment=1, width=6)
        sb_roll.grid(row=0, column=4)

        # ── Group 4: Output Columns & Accuracy ───────────────────────────────
        g4 = ttk.LabelFrame(f, text="Output Columns & Accuracy")
        g4.grid(row=5, column=0, columnspan=2, sticky="ew", padx=8, pady=2)
        g4.columnconfigure(1, weight=1)
        g4.columnconfigure(3, weight=1)

        alt_row = ttk.Frame(g4)
        alt_row.grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(8, 4))

        cb_alt = ttk.Checkbutton(
            alt_row, text="Include Altitude / AccuracyZ  ← RECOMMENDED for dashcam",
            variable=self.var_include_alt,
        )
        cb_alt.grid(row=0, column=0, sticky="w")
        _Tooltip(cb_alt, "Adds Altitude and AccuracyZ columns to the CSV.\n\n"
                 "RECOMMENDED for forward-facing / dashcam setups:\n"
                 "  Without altitude, Georef has no vertical constraint on the\n"
                 "  reconstruction — it can flip the scene 180° ('street in sky').\n\n"
                 "Device vertical accuracy after PPK is typically 0.15–0.50 m.\n"
                 "Set AccuracyZ to 0.30–0.50 m to let Georef trust it appropriately.")

        ttk.Label(alt_row, text="   Z smooth σ (s):", foreground="#888").grid(
            row=0, column=1, padx=(16, 2))
        sb_zsig = ttk.Spinbox(alt_row, textvariable=self.var_z_sigma_override,
                              from_=0, to=300, increment=5, width=7)
        sb_zsig.grid(row=0, column=2)
        _Tooltip(sb_zsig, "Gaussian smoothing window for altitude (σ in seconds), "
                 "independent of the XY profile.\n\n"
                 "PPK vertical noise is 3–5× worse than horizontal.\n"
                 "30s recommended for road driving (slow altitude change).\n"
                 "0 = use profile default.  Higher = smoother but less responsive to hills.")

        # Explicit opt-in to smoothing altitude (Z). Default OFF — device Z is
        # noisy, so smoothing it is a deliberate choice independent of XY.
        cb_smooth_alt = ttk.Checkbutton(
            alt_row, text="Smooth altitude (Z)",
            variable=self.var_smooth_alt,
        )
        cb_smooth_alt.grid(row=1, column=0, sticky="w", pady=(4, 0))
        _Tooltip(cb_smooth_alt,
                 "Opt in to Gaussian-smoothing the altitude (Z) channel,\n"
                 "independently of horizontal (XY) smoothing.\n\n"
                 "OFF by default: device vertical position is noisy, and smoothing\n"
                 "it is a deliberate choice. When ON, the σ on the right is used\n"
                 "(falls back to the Z-override / profile when 0).")

        ttk.Label(alt_row, text="   alt σ (s):", foreground="#888").grid(
            row=1, column=1, padx=(16, 2), pady=(4, 0))
        sb_altsig = ttk.Spinbox(alt_row, textvariable=self.var_alt_smooth_sigma,
                                from_=0, to=300, increment=5, width=7)
        sb_altsig.grid(row=1, column=2, pady=(4, 0))
        _Tooltip(sb_altsig,
                 "Gaussian window (σ in seconds) applied to altitude when\n"
                 "'Smooth altitude (Z)' is checked. 0 = fall back to the\n"
                 "Z-override / profile Z sigma.")

        ttk.Label(g4, text="accuracy XY (m)").grid(row=1, column=0, sticky="e",
                                                   padx=(10, 4), pady=(0, 8))
        sb_axy = ttk.Spinbox(g4, textvariable=self.var_acc_xy,
                             from_=0.01, to=50.0, increment=0.05, width=8)
        sb_axy.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(0, 8))
        _Tooltip(sb_axy, "Horizontal position accuracy written to AccuracyX / AccuracyY (metres).\n"
                 "Georef uses this to weight GPS reference against image tie-points.\n"
                 "Device PPK fixed: ~0.05–0.15 m.  Float: ~0.30–1.00 m.")

        ttk.Label(g4, text="accuracy Z (m)").grid(row=1, column=2, sticky="e", padx=(0, 4))
        sb_az = ttk.Spinbox(g4, textvariable=self.var_acc_z,
                            from_=0.01, to=500.0, increment=0.05, width=8)
        sb_az.grid(row=1, column=3, sticky="w", padx=(0, 10), pady=(0, 8))
        _Tooltip(sb_az, "Vertical position accuracy for AccuracyZ (metres).\n"
                 "Typically 2–3× the horizontal accuracy for data GNSS.")

        ttk.Label(
            g4,
            text="CSV always includes ENU Doppler velocity: "
                 "DopplerVe, DopplerVn, DopplerVu (m/s) + DopplerSpeed + CoordsSpeed per frame.",
            foreground="#4cc9f0", font=("Segoe UI", 8),
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 8))

        # ── Group 5: Client path export ────────────────────────────────
        # Options consumed by stages.user_export.export_trajectory/export_kml
        # (used wherever the GUI writes a client-facing path, e.g. the
        # Smoothers tab per-smoother *.client.csv / *.client.export format outputs).
        g5 = ttk.LabelFrame(f, text="Client trajectory export (coordinate systems + Z)")
        g5.grid(row=6, column=0, columnspan=2, sticky="ew", padx=8, pady=2)

        coord_row = ttk.Frame(g5)
        coord_row.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))
        ttk.Label(coord_row, text="Coordinate systems:").grid(
            row=0, column=0, sticky="w", padx=(0, 8))

        cb_geo = ttk.Checkbutton(coord_row, text="Geodetic (lat/lon/h)",
                                 variable=self.var_exp_coord_geodetic)
        cb_geo.grid(row=0, column=1, sticky="w", padx=(0, 10))
        _Tooltip(cb_geo, "WGS84 geodetic block: lat_deg, lon_deg, h_m\n"
                 "(degrees + ellipsoidal metres). Part of the legacy default.")

        cb_ecef = ttk.Checkbutton(coord_row, text="ECEF",
                                  variable=self.var_exp_coord_ecef)
        cb_ecef.grid(row=0, column=2, sticky="w", padx=(0, 10))
        _Tooltip(cb_ecef, "WGS84 Earth-Centred Earth-Fixed block:\n"
                 "x_ecef_m, y_ecef_m, z_ecef_m (metres).\n"
                 "Part of the legacy default.")

        cb_utm = ttk.Checkbutton(coord_row, text="UTM",
                                 variable=self.var_exp_coord_utm)
        cb_utm.grid(row=0, column=3, sticky="w", padx=(0, 10))
        _Tooltip(cb_utm, "UTM block: utm_easting_m, utm_northing_m, utm_zone, h_m.\n"
                 "Zone auto-picked from the trajectory's mean lon/lat\n"
                 "(EPSG recorded in a '#' header comment). Requires pyproj.")

        cb_enu = ttk.Checkbutton(coord_row, text="ENU",
                                 variable=self.var_exp_coord_enu)
        cb_enu.grid(row=0, column=4, sticky="w")
        _Tooltip(cb_enu, "Local East-North-Up block: e_m, n_m, u_m (metres).\n"
                 "Origin = first valid fix of the exported trajectory.")

        zrow = ttk.Frame(g5)
        zrow.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))
        cb_expz = ttk.Checkbutton(zrow, text="Smooth height (Z)",
                                  variable=self.var_exp_smooth_z)
        cb_expz.grid(row=0, column=0, sticky="w")
        _Tooltip(cb_expz, "Gaussian-smooth the exported height (h_m) over time\n"
                 "(time-weighted; splits at gaps). Applied consistently to every\n"
                 "selected coordinate system and the KML altitude.\n"
                 "DEFAULT ON — matches the export backend default.")
        ttk.Label(zrow, text="   z σ (s):", foreground="#888").grid(
            row=0, column=1, padx=(16, 2))
        sb_expz = ttk.Spinbox(zrow, textvariable=self.var_exp_z_sigma_s,
                              from_=0.1, to=60.0, increment=0.5, width=7)
        sb_expz.grid(row=0, column=2)
        _Tooltip(sb_expz, "Gaussian window (σ, seconds) for the exported height\n"
                 "when 'Smooth height (Z)' is checked. Default 3.0 s.")

        # Time-basis chooser (user_export ``time_bases``): which time
        # column(s) each exported row carries. Reference time-only is the legacy
        # default — output byte-identical when nothing else is ticked.
        tbrow = ttk.Frame(g5)
        tbrow.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 8))
        ttk.Label(tbrow, text="Time basis:").grid(
            row=0, column=0, sticky="w", padx=(0, 8))

        cb_tb_gpst = ttk.Checkbutton(tbrow, text="GPST",
                                     variable=self.var_tb_gpst)
        cb_tb_gpst.grid(row=0, column=1, sticky="w", padx=(0, 10))
        _Tooltip(cb_tb_gpst, "GPS time column: gpst_s (seconds of GPS time).\n"
                 "The legacy default — leaving only this checked keeps the\n"
                 "export byte-identical to previous versions.")

        cb_tb_utc = ttk.Checkbutton(tbrow, text="UTC",
                                    variable=self.var_tb_utc)
        cb_tb_utc.grid(row=0, column=2, sticky="w", padx=(0, 10))
        _Tooltip(cb_tb_utc, "UTC column: utc_s (Unix epoch seconds,\n"
                 "leap-second corrected from GPST).")

        cb_tb_audio = ttk.Checkbutton(tbrow, text="Audio",
                                      variable=self.var_tb_audio)
        cb_tb_audio.grid(row=0, column=3, sticky="w", padx=(0, 10))
        _Tooltip(cb_tb_audio, "Audio-relative column: t_audio_s = seconds since\n"
                 "audio sample 0 of the session's global WAV.\n"
                 "Needs the loaded RAW session's audio anchor\n"
                 "(audio_anchor_*.txt) + a boot→UTC anchor; if either is\n"
                 "missing the basis is dropped with a logged warning.")

        cb_tb_iso = ttk.Checkbutton(tbrow, text="ISO 8601",
                                    variable=self.var_tb_iso)
        cb_tb_iso.grid(row=0, column=4, sticky="w")
        _Tooltip(cb_tb_iso, "Human-readable UTC timestamp column:\n"
                 "iso_time (ISO 8601, e.g. 2026-06-29T12:34:56.123456Z).")

        # Export-source chooser + final-velocity gate (user_export
        # resolve_export_rows / emit_final_velocity / vel_disagree gate).
        # All defaults neutral — the client export stays byte-identical
        # until the user changes them.
        srow = ttk.Frame(g5)
        srow.grid(row=3, column=0, sticky="w", padx=10, pady=(0, 8))
        ttk.Label(srow, text="Export source:").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        try:
            from .smoothers import list_smoothers as _list_sm
            _src_values = ([self.EXPORT_SOURCE_AS_RUN, "raw"]
                           + list(_list_sm()))
        except Exception:
            _src_values = [self.EXPORT_SOURCE_AS_RUN, "raw"]
        cb_src = ttk.Combobox(srow, textvariable=self.var_exp_source,
                              values=_src_values, state="readonly",
                              width=20)
        cb_src.grid(row=0, column=1, sticky="w", padx=(0, 14))
        _Tooltip(cb_src,
                 "Which trajectory the client CSV/KML carries:\n"
                 "• (as run) — each smoother's own output (legacy default)\n"
                 "• raw — the unsmoothed PPK rows, exactly as parsed\n"
                 "• <smoother> — run that smoother once on the raw rows and\n"
                 "  export ITS trajectory, decoupled from whichever\n"
                 "  smoothers the comparison table ran.")

        cb_fv = ttk.Checkbutton(srow, text="Emit final velocity",
                                variable=self.var_emit_final_vel)
        cb_fv.grid(row=0, column=2, sticky="w", padx=(0, 10))
        _Tooltip(cb_fv,
                 "Adds final_vn/ve/vu_mps + final_speed_mps (raw Doppler\n"
                 "velocity) plus vel_disagree_mps + coords_dropped columns\n"
                 "to the client CSV. OFF (default) keeps the export\n"
                 "byte-identical to previous versions.")

        ttk.Label(srow, text="vel disagree thr (m/s):",
                  foreground="#888").grid(row=0, column=3, padx=(0, 2))
        ent_vd = ttk.Entry(srow, textvariable=self.var_vel_disagree,
                           width=7)
        ent_vd.grid(row=0, column=4, sticky="w")
        _Tooltip(ent_vd,
                 "Optional coordinate-vs-Doppler velocity disagreement gate\n"
                 "(m/s). Rows whose coordinate-derived velocity disagrees\n"
                 "with raw Doppler beyond this threshold get blank\n"
                 "coordinates + blank final_v* (coords_dropped=1).\n"
                 "Blank = gate disabled (default).")

        # ── Buttons ──────────────────────────────────────────────────────────
        ttk.Separator(f, orient="horizontal").grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=8, pady=8,
        )
        btn_row = ttk.Frame(f)
        btn_row.grid(row=8, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))

        btn1 = ttk.Button(btn_row, text="Extract frames", command=self._run_frames)
        btn1.grid(row=0, column=0, padx=(0, 6))
        btn2 = ttk.Button(btn_row, text="Build Georef CSV",
                          style="Accent.TButton", command=self._run_csv)
        btn2.grid(row=0, column=1, padx=6)
        btn3 = ttk.Button(btn_row, text="Run both", command=self._run_frames_and_csv)
        btn3.grid(row=0, column=2, padx=6)
        btn_open_out = ttk.Button(
            btn_row, text="Open last output ↗", command=self._open_last_output,
        )
        btn_open_out.grid(row=0, column=3, padx=6)
        _Tooltip(btn_open_out, "Open the current Output folder in Explorer / Finder / xdg-open.")
        btn4 = ttk.Button(btn_row, text="Plot velocity...", command=self._show_vel_plot)
        btn4.grid(row=0, column=4, padx=6)
        self._buttons.extend([btn1, btn2, btn3, btn4])

        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

    def _build_t02_tab(self, nb: ttk.Notebook) -> None:
        """T02 / JPS → Interchange-format converter panel.

        Wraps ``jps2rin.exe`` (default) and The external solver's ``the converter binary.exe`` so the
        user can convert any The reference unit/Topcon binary log to Interchange-format OBS + NAV in
        one click. Discovered OBS / NAV files are surfaced so they can be
        copy-pasted (or in a future step auto-fed) into the Post-processing tab.
        """
        f = self._make_scrollable_tab(nb, "T02 Tools")

        self.var_t02_input    = tk.StringVar()
        self.var_t02_out      = tk.StringVar()
        self.var_t02_version  = tk.StringVar(value="3.05")
        self.var_t02_conv     = tk.StringVar(value="trimble")
        self.var_t02_doppler  = tk.BooleanVar(value=True)
        self.var_t02_snr      = tk.BooleanVar(value=True)
        try:
            from . import lab_tools as _lt
            _default_runpkr = str(_lt.resolve_tool("runpkr00"))
        except Exception:
            _default_runpkr = ""
        self.var_t02_tool     = tk.StringVar(value=_default_runpkr)
        self.var_t02_teqc     = tk.StringVar(value=str(t02_stage.DEFAULT_TEQC))
        self.var_t02_week     = tk.StringVar(value="")
        self._t02_last_obs:    list[Path] = []
        self._t02_last_nav:    list[Path] = []

        ttk.Label(
            f,
            text=(
                "Convert a GNSS binary log to RINEX OBS + NAV. The OBS "
                "output becomes a perfect base or rover for the PPK tab.\n"
                "  •  Trimble  (.t02 / .t01 / .t00 / .r00)  →  "
                "runpkr00 + teqc\n"
                "  •  Javad    (.jps / .tps)                →  "
                "jps2rin   (RTKLIB convbin -r javad as fallback)\n"
                "The converter is auto-picked from the input extension; "
                "you can override if you know better."
            ),
            foreground=self._fg_dim, wraplength=920, style="Dim.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 4))

        # ── Group 1: Files ──────────────────────────────────────────────────
        g_in = ttk.LabelFrame(f, text="  Input / Output  ")
        g_in.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        g_in.columnconfigure(1, weight=1)

        in_label = "Binary log  (.t02 / .t01 / .jps / .tps)"
        if self._dnd_files is not None:
            in_label = "Binary log  (drop file here)"
        in_entry = self._row_path(
            g_in, 0, label=in_label,
            var=self.var_t02_input, kind="file_open",
            file_types=[
                ("Trimble", "*.t02 *.T02 *.t01 *.T01 *.t00 *.T00 *.r00 *.R00"),
                ("Javad",   "*.jps *.JPS *.tps *.TPS *.tpd *.TPD"),
                ("All",     "*.*"),
            ],
            on_change=self._on_t02_input_changed,
        )
        self._register_dnd_video(in_entry, self.var_t02_input,
                                 self._on_t02_input_changed)

        self._row_path(
            g_in, 1, label="Output directory",
            var=self.var_t02_out, kind="dir", on_change=None,
        )

        # ── Group 2: Options ────────────────────────────────────────────────
        g_op = ttk.LabelFrame(f, text="  Conversion Options  ")
        g_op.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        g_op.columnconfigure(1, weight=1)
        g_op.columnconfigure(3, weight=1)

        ttk.Label(g_op, text="converter").grid(
            row=0, column=0, sticky="e", padx=(10, 4), pady=6)
        cb_conv = ttk.Combobox(
            g_op, textvariable=self.var_t02_conv,
            values=list(t02_stage.VALID_CONVERTERS),
            state="readonly", width=12,
        )
        cb_conv.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=6)
        cb_conv.bind("<<ComboboxSelected>>",
                     lambda _e: self._on_t02_converter_changed())
        _Tooltip(cb_conv,
                 "trimble — runpkr00 unpacks .T02/.T01/.T00 → .tgd, then "
                 "teqc converts .tgd to RINEX OBS + NAV.\n"
                 "jps2rin — Javad's official .jps converter.\n"
                 "convbin — RTKLIB convbin -r javad (.jps fallback).")

        ttk.Label(g_op, text="RINEX version").grid(
            row=0, column=2, sticky="e", padx=(0, 4))
        cb_ver = ttk.Combobox(g_op, textvariable=self.var_t02_version,
                              values=list(t02_stage.SUPPORTED_RINEX_VERSIONS),
                              state="readonly", width=8)
        cb_ver.grid(row=0, column=3, sticky="w", padx=(0, 10))
        _Tooltip(cb_ver,
                 "RINEX version of the OBS + NAV output.\n"
                 "3.05 is the modern default; 2.11 for legacy tools.")

        ttk.Label(g_op, text="extras (convbin only)",
                  style="Dim.TLabel").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=(0, 8))
        extras = ttk.Frame(g_op)
        extras.grid(row=1, column=1, columnspan=3, sticky="w", padx=(0, 10), pady=(0, 8))
        self._t02_cb_doppler = ttk.Checkbutton(
            extras, text="Doppler  (-od)", variable=self.var_t02_doppler)
        self._t02_cb_doppler.grid(row=0, column=0, padx=(0, 14))
        self._t02_cb_snr = ttk.Checkbutton(
            extras, text="SNR  (-os)", variable=self.var_t02_snr)
        self._t02_cb_snr.grid(row=0, column=1)
        # NOTE: initial _on_t02_converter_changed() call happens at the end of
        # _build_t02_tab once the engine-paths widgets exist.

        # ── Group 3: Engine paths ──────────────────────────────────────────
        g_eng = ttk.LabelFrame(f, text="  Converter Executables  ")
        g_eng.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        g_eng.columnconfigure(1, weight=1)

        # Primary tool (jps2rin / convbin / runpkr00 depending on converter).
        self._t02_tool_lbl = ttk.Label(g_eng, text="runpkr00")
        self._t02_tool_lbl.grid(row=0, column=0, sticky="e", padx=8, pady=6)
        ttk.Entry(g_eng, textvariable=self.var_t02_tool).grid(
            row=0, column=1, sticky="ew", padx=8)

        def _browse_tool() -> None:
            p = filedialog.askopenfilename(
                filetypes=[("Executables", "*.exe"), ("All", "*.*")],
                initialdir=str(Path(self.var_t02_tool.get()).parent
                               if self.var_t02_tool.get() else "/"),
            )
            if p:
                self.var_t02_tool.set(p)
        ttk.Button(g_eng, text="Browse…", command=_browse_tool).grid(
            row=0, column=2, sticky="w", padx=(0, 8))

        # teqc (Trimble pipeline only — hidden otherwise).
        self._t02_teqc_lbl = ttk.Label(g_eng, text="teqc")
        self._t02_teqc_lbl.grid(row=1, column=0, sticky="e", padx=8, pady=6)
        self._t02_teqc_entry = ttk.Entry(g_eng, textvariable=self.var_t02_teqc)
        self._t02_teqc_entry.grid(row=1, column=1, sticky="ew", padx=8)

        def _browse_teqc() -> None:
            p = filedialog.askopenfilename(
                filetypes=[("Executables", "*.exe"), ("All", "*.*")],
                initialdir=str(Path(self.var_t02_teqc.get()).parent
                               if self.var_t02_teqc.get() else "/"),
            )
            if p:
                self.var_t02_teqc.set(p)
        self._t02_teqc_btn = ttk.Button(g_eng, text="Browse…", command=_browse_teqc)
        self._t02_teqc_btn.grid(row=1, column=2, sticky="w", padx=(0, 8))

        # Optional Reference week override (Trimble only — teqc occasionally needs it
        # when the embedded week wraps; teqc prints a clear "(try -week N)" hint).
        self._t02_week_lbl = ttk.Label(g_eng, text="GPS week (optional)")
        self._t02_week_lbl.grid(row=2, column=0, sticky="e", padx=8, pady=6)
        self._t02_week_entry = ttk.Entry(g_eng, textvariable=self.var_t02_week, width=10)
        self._t02_week_entry.grid(row=2, column=1, sticky="w", padx=8)
        _Tooltip(self._t02_week_entry,
                 "Optional -week N override forwarded to teqc.\n"
                 "Only needed when teqc warns 'translation may have started "
                 "with GPS week NNNN rather than MMMM'.")

        _Tooltip(g_eng,
                 "Trimble  : runpkr00 (top) + teqc (middle).  Set GPS week "
                 "if teqc complains about week rollover.\n"
                 "jps2rin / convbin: only the top path is used.\n"
                 "Env vars JPS2RIN / CONVBIN / RUNPKR00 / TEQC also honoured.")

        # ── Group 4: Last result ────────────────────────────────────────────
        g_out = ttk.LabelFrame(f, text="  Latest Outputs  ")
        g_out.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=10, pady=4)
        g_out.columnconfigure(0, weight=1)
        g_out.rowconfigure(0, weight=1)
        self._t02_result_text = tk.Text(
            g_out, height=6, bg=self._bg_inset, fg=self._fg,
            font=("Consolas", 9), wrap="none", relief="flat",
            padx=8, pady=4, insertbackground=self._accent,
        )
        self._t02_result_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._t02_result_text.insert("1.0", "(no run yet)")
        self._t02_result_text.configure(state="disabled")

        # ── Buttons ─────────────────────────────────────────────────────────
        btn_row = ttk.Frame(f)
        btn_row.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(6, 10))

        btn_run = ttk.Button(btn_row, text="Convert", style="Accent.TButton",
                             command=self._run_t02)
        btn_run.grid(row=0, column=0, padx=(0, 6))
        btn_open = ttk.Button(btn_row, text="Open output ↗",
                              command=self._t02_open_output)
        btn_open.grid(row=0, column=1, padx=6)
        btn_use_ppk = ttk.Button(btn_row, text="Send OBS to PPK rover",
                                 command=lambda: self._t02_send_to_ppk("rover"))
        btn_use_ppk.grid(row=0, column=2, padx=6)
        btn_use_base = ttk.Button(btn_row, text="Send OBS to PPK base",
                                  command=lambda: self._t02_send_to_ppk("base"))
        btn_use_base.grid(row=0, column=3, padx=6)
        btn_use_nav = ttk.Button(btn_row, text="Send NAV to PPK",
                                 command=self._t02_send_nav_to_ppk)
        btn_use_nav.grid(row=0, column=4, padx=6)
        self._buttons.append(btn_run)

        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(4, weight=1)

        # Now that every engine-paths widget exists, sync extras + teqc state.
        self._on_t02_converter_changed()

    # ------------------------------------------------------------------
    # T02 helpers
    # ------------------------------------------------------------------

    def _on_t02_input_changed(self) -> None:
        s = self.var_t02_input.get().strip()
        if not s:
            return
        p = Path(s)
        if not p.is_file():
            return
        # Default output dir to the input's parent so user gets sensible behaviour.
        if not self.var_t02_out.get().strip():
            self.var_t02_out.set(str(p.parent))
        # Auto-pick converter from extension (user can still override).
        picked = t02_stage.auto_pick_converter(p)
        if self.var_t02_conv.get() != picked:
            self.var_t02_conv.set(picked)
        self._on_t02_converter_changed()

    def _on_t02_converter_changed(self) -> None:
        # Resolve tool paths via the lab_tools chain (env var → vendor → PATH).
        # Fall back to the tool name itself (string) when the resolver can't
        # find it — the user then picks the binary via the GUI file dialog.
        from . import lab_tools as _lt
        def _resolve(name: str) -> str:
            try:
                return str(_lt.resolve_tool(name))
            except Exception:
                return name  # e.g. "runpkr00"

        conv = self.var_t02_conv.get()
        all_defaults = {_resolve("jps2rin"), _resolve("convbin"), _resolve("runpkr00")}
        current = self.var_t02_tool.get().strip()
        if conv == "trimble":
            new_default = _resolve("runpkr00")
            tool_label = "runpkr00"
            show_teqc = True
            extras_enabled = False
        elif conv == "jps2rin":
            new_default = _resolve("jps2rin")
            tool_label = "jps2rin"
            show_teqc = False
            extras_enabled = False
        else:  # convbin
            new_default = _resolve("convbin")
            tool_label = "convbin"
            show_teqc = False
            extras_enabled = True

        # Only overwrite the tool path if user hasn't customised it.
        if not current or current in all_defaults:
            self.var_t02_tool.set(new_default)
        self._t02_tool_lbl.configure(text=tool_label)

        # Rate-signal / SNR are convbin-only.
        try:
            target_state = "!disabled" if extras_enabled else "disabled"
            self._t02_cb_doppler.state([target_state])
            self._t02_cb_snr.state([target_state])
        except tk.TclError:
            pass

        # teqc row + Reference week only relevant for Trimble pipeline.
        teqc_state = "normal" if show_teqc else "disabled"
        for w in (self._t02_teqc_entry, self._t02_teqc_btn, self._t02_week_entry):
            try:
                w.configure(state=teqc_state)
            except tk.TclError:
                pass
        teqc_fg = self._fg if show_teqc else self._fg_dim
        try:
            self._t02_teqc_lbl.configure(foreground=teqc_fg)
            self._t02_week_lbl.configure(foreground=teqc_fg)
        except tk.TclError:
            pass

    def _t02_open_output(self) -> None:
        s = self.var_t02_out.get().strip()
        if not s:
            messagebox.showinfo("No output", "Pick an output directory first.")
            return
        p = Path(s)
        if not p.is_dir():
            messagebox.showerror("Missing", f"Not a folder:\n{p}")
            return
        self._open_folder(p)

    def _t02_render_result(self, res: "t02_stage.T02ConvertResult") -> None:
        lines = [
            f"converter : {res.converter}",
            f"engine    : {res.tool_exe}",
            f"input     : {res.input_file}",
            f"output dir: {res.output_dir}",
            f"RINEX ver : {res.rinex_version}",
            "",
            f"OBS files ({len(res.obs_files)}):",
        ]
        for p in res.obs_files:
            lines.append(f"  • {p}   ({p.stat().st_size:,} B)")
        lines.append(f"NAV files ({len(res.nav_files)}):")
        for p in res.nav_files:
            lines.append(f"  • {p}   ({p.stat().st_size:,} B)")
        self._t02_result_text.configure(state="normal")
        self._t02_result_text.delete("1.0", "end")
        self._t02_result_text.insert("1.0", "\n".join(lines))
        self._t02_result_text.configure(state="disabled")

    def _t02_send_to_ppk(self, role: str) -> None:
        if not self._t02_last_obs:
            messagebox.showinfo("Nothing to send", "Run the converter first.")
            return
        obs = self._t02_last_obs[0]
        if role == "rover":
            self.var_ppk_rover.set(str(obs))
            self._on_ppk_rover_changed()
            self._log(f"[t02→ppk] rover obs set to {obs}")
        elif role == "base":
            self.var_ppk_base.set(str(obs))
            self._log(f"[t02→ppk] base obs set to {obs}")

    def _t02_send_nav_to_ppk(self) -> None:
        if not self._t02_last_nav:
            messagebox.showinfo("Nothing to send", "Run the converter first.")
            return
        existing = self._ppk_get_nav_files()
        seen = {p.resolve() for p in existing}
        for p in self._t02_last_nav:
            if p.resolve() not in seen:
                existing.append(p)
                seen.add(p.resolve())
        self._ppk_set_nav_files(existing)
        self._log(f"[t02→ppk] added {len(self._t02_last_nav)} nav file(s) "
                  f"to PPK panel")

    def _run_t02(self) -> None:
        s_in = self.var_t02_input.get().strip()
        s_out = self.var_t02_out.get().strip()
        if not s_in:
            messagebox.showerror("Missing input", "Pick a .t02 / .jps file first.")
            return
        if not s_out:
            messagebox.showerror("Missing output", "Pick an output directory.")
            return
        in_path = Path(s_in)
        out_dir = Path(s_out)
        version = self.var_t02_version.get()
        converter = self.var_t02_conv.get()
        tool_raw = self.var_t02_tool.get().strip()
        tool_path: Optional[Path] = Path(tool_raw) if tool_raw else None
        teqc_raw = self.var_t02_teqc.get().strip()
        teqc_path: Optional[Path] = Path(teqc_raw) if teqc_raw else None
        inc_dop = self.var_t02_doppler.get()
        inc_snr = self.var_t02_snr.get()
        week_raw = self.var_t02_week.get().strip()
        gps_week: Optional[int]
        if week_raw:
            try:
                gps_week = int(week_raw)
            except ValueError:
                messagebox.showerror(
                    "Invalid GPS week",
                    f"GPS week must be an integer, got {week_raw!r}.",
                )
                return
        else:
            gps_week = None

        def go() -> None:
            res = t02_stage.run(
                input_file=in_path,
                output_dir=out_dir,
                rinex_version=version,
                converter=converter,
                tool_exe=tool_path,
                teqc_exe=teqc_path,
                gps_week=gps_week,
                include_doppler=inc_dop,
                include_snr=inc_snr,
                log=self._log,
            )
            # Marshal worker-thread state mutations + UI render onto the
            # main thread so any read of self._t02_last_* from a callback
            # (e.g. _t02_send_to_ppk) sees a consistent value.
            self.root.after(0, lambda r=res: (
                setattr(self, "_t02_last_obs", list(r.obs_files)),
                setattr(self, "_t02_last_nav", list(r.nav_files)),
                self._t02_render_result(r),
            ))

        self._run_async(go, f"Binary → RINEX ({converter})")

    def _build_video_only_tab(self, nb: ttk.Notebook) -> None:
        """Standalone media → samples panel — no Post-processing, no RAW folder required.

        Names every sample after its exact source PTS in seconds (e.g.
        ``0.034550.png``) by passing ``name_prefix=''`` to the samples stage,
        which preserves the source presentation timestamp via the
        ``select='not(mod(n,K))'`` filter (no PTS resampling).
        """
        f = self._make_scrollable_tab(nb, "Video Only")

        self.var_vo_video    = tk.StringVar()
        self.var_vo_out      = tk.StringVar()
        self.var_vo_fps      = tk.DoubleVar(value=6.0)
        self.var_vo_rotation = tk.StringVar(value="0")
        self.var_vo_format   = tk.StringVar(value="png")
        self.var_vo_decimals = tk.IntVar(value=6)
        self._vo_video_path: Optional[Path] = None

        ttk.Label(
            f,
            text=(
                "Standalone frame extraction: drop in any video (MP4, MPEG-TS, MOV …) "
                "and get frames named by their exact PTS seconds-from-start "
                "(e.g. 0.034550.png). No PPK, RAW folder, or CSV required.\n"
                "Filenames carry the original presentation timestamp — no PTS "
                "resampling — so timing is exact to the source frame clock."
            ),
            foreground="#888", wraplength=920,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        # ── Inputs ──────────────────────────────────────────────────────────
        g_in = ttk.LabelFrame(f, text="Inputs")
        g_in.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        g_in.columnconfigure(1, weight=1)

        vid_label = "Video file  (.mp4 / .mpg / .ts / .mov)"
        if self._dnd_files is not None:
            vid_label = "Video file  (drop file here)"
        vid_entry = self._row_path(
            g_in, 0, label=vid_label,
            var=self.var_vo_video, kind="file_open",
            file_types=[
                ("Video", "*.mp4 *.mpg *.mpeg *.ts *.m2ts *.mov *.mkv *.avi"),
                ("All", "*.*"),
            ],
            on_change=self._on_vo_video_changed,
        )
        self._register_dnd_video(vid_entry, self.var_vo_video,
                                 self._on_vo_video_changed)
        self._row_path(
            g_in, 1, label="Output folder",
            var=self.var_vo_out, kind="dir",
            on_change=self._on_vo_out_changed,
        )

        # ── Extraction options ──────────────────────────────────────────────
        g_op = ttk.LabelFrame(f, text="Extraction Options")
        g_op.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        g_op.columnconfigure(1, weight=1)
        g_op.columnconfigure(3, weight=1)

        ttk.Label(g_op, text="fps").grid(row=0, column=0, sticky="e", padx=(10, 4), pady=6)
        sb_fps = ttk.Spinbox(g_op, textvariable=self.var_vo_fps,
                             from_=0.1, to=120.0, increment=0.5, width=8)
        sb_fps.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=6)
        _Tooltip(sb_fps, "Target extraction frame rate (Hz).\n"
                 "Actual decimation is K = round(source_fps / target); PTS is "
                 "preserved per kept frame (no resampling).")

        ttk.Label(g_op, text="filename decimals").grid(row=0, column=2, sticky="e", padx=(0, 4))
        sb_dec = ttk.Spinbox(g_op, textvariable=self.var_vo_decimals,
                             from_=3, to=12, increment=1, width=6)
        sb_dec.grid(row=0, column=3, sticky="w", padx=(0, 10))
        _Tooltip(sb_dec, "Decimal places of seconds in each filename.\n"
                 "6 → microseconds (0.034550.png).  "
                 "9 → nanoseconds — typical max useful precision from H.264 PTS.")

        ttk.Label(g_op, text="image format").grid(row=1, column=0, sticky="e",
                                                  padx=(10, 4), pady=(0, 8))
        cb_fmt = ttk.Combobox(g_op, textvariable=self.var_vo_format,
                              values=["png", "tiff", "jpeg1"],
                              state="readonly", width=10)
        cb_fmt.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(0, 8))
        _Tooltip(cb_fmt, "png/tiff lossless; jpeg1 = ffmpeg -q:v 1 (near-lossless, ~10× smaller).")

        ttk.Label(g_op, text="rotation").grid(row=1, column=2, sticky="e",
                                              padx=(0, 4), pady=(0, 8))
        cb_rot = ttk.Combobox(g_op, textvariable=self.var_vo_rotation,
                              values=["0", "90", "180", "270"],
                              state="readonly", width=8)
        cb_rot.grid(row=1, column=3, sticky="w", padx=(0, 10), pady=(0, 8))
        _Tooltip(cb_rot, "Clockwise rotation applied to every extracted frame.\n"
                 "Auto-updates the preview window on change.")
        self.var_vo_rotation.trace_add(
            "write", lambda *_: self._update_video_only_preview()
        )

        # ── Buttons ─────────────────────────────────────────────────────────
        btn_row = ttk.Frame(f)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 8))

        btn_run = ttk.Button(btn_row, text="Extract frames",
                             style="Accent.TButton",
                             command=self._run_video_only)
        btn_run.grid(row=0, column=0, padx=(0, 6))
        btn_open = ttk.Button(btn_row, text="Open output ↗",
                              command=self._open_vo_output)
        btn_open.grid(row=0, column=1, padx=6)
        btn_preview = ttk.Button(btn_row, text="Preview now",
                                 command=self._update_video_only_preview)
        btn_preview.grid(row=0, column=2, padx=6)
        _Tooltip(btn_preview, "Re-extract the rotation preview frame (at t = 5 s).")

        self._buttons.append(btn_run)

        ttk.Label(
            f,
            text=(
                "Filenames are exactly the seconds-from-start PTS — for example "
                "0.034550.png (no 'frame_' prefix). An extracted_frame_times.csv "
                "with full-precision (12 dp) timestamps is written next to the "
                "frames folder for any downstream tool that needs them."
            ),
            foreground="#4cc9f0", font=("Segoe UI", 8), wraplength=900,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

    def _register_dnd_video(
        self, widget: tk.Widget, var: tk.StringVar,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a media-file drop target (folder drops resolve to first file)."""
        if self._dnd_files is None:
            return

        def _handle(event: "tk.Event") -> None:  # type: ignore[type-arg]
            data = event.data or ""
            paths = self.root.tk.splitlist(data)  # type: ignore[attr-defined]
            if not paths:
                return
            p = Path(paths[0])
            if p.is_dir():
                exts = {".mp4", ".mpg", ".mpeg", ".ts", ".m2ts", ".mov", ".mkv", ".avi"}
                for cand in sorted(p.iterdir()):
                    if cand.suffix.lower() in exts:
                        p = cand
                        break
            var.set(str(p))
            if on_change:
                on_change()

        try:
            widget.drop_target_register(self._dnd_files)  # type: ignore[attr-defined]
            widget.dnd_bind("<<Drop>>", _handle)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _on_vo_video_changed(self) -> None:
        s = self.var_vo_video.get().strip()
        if not s:
            self._vo_video_path = None
            return
        p = Path(s)
        if not p.is_file():
            self._vo_video_path = None
            return
        self._vo_video_path = p
        # Default output folder to <video_parent>/<stem>_frames if not yet set.
        if not self.var_vo_out.get().strip():
            self.var_vo_out.set(str(p.parent / f"{p.stem}_frames"))
        self.root.after(150, self._update_video_only_preview)

    def _on_vo_out_changed(self) -> None:
        s = self.var_vo_out.get().strip()
        if s:
            Path(s).mkdir(parents=True, exist_ok=True)

    def _update_video_only_preview(self) -> None:
        if self._vo_video_path is None or not self._vo_video_path.is_file():
            return
        rotation = int(self.var_vo_rotation.get() or "0")
        self._refresh_preview(self._vo_video_path, rotation)

    def _open_vo_output(self) -> None:
        s = self.var_vo_out.get().strip()
        if not s:
            messagebox.showinfo("No output folder",
                                "Pick an output folder first.")
            return
        out = Path(s)
        out.mkdir(parents=True, exist_ok=True)
        self._open_folder(out)

    def _run_video_only(self) -> None:
        if self._vo_video_path is None or not self._vo_video_path.is_file():
            messagebox.showerror("Missing video", "Pick a video file first.")
            return
        out_s = self.var_vo_out.get().strip()
        if not out_s:
            messagebox.showerror("Missing output", "Pick an output folder first.")
            return
        video    = self._vo_video_path
        out      = Path(out_s)
        out.mkdir(parents=True, exist_ok=True)
        fps      = float(self.var_vo_fps.get())
        fmt      = self.var_vo_format.get()
        pts_dec  = int(self.var_vo_decimals.get())
        rotation = int(self.var_vo_rotation.get() or "0")

        def _prog(n: int, total: "Optional[int]") -> None:
            self.root.after(0, lambda n=n, t=total: self._set_frame_progress(n, t))

        def go() -> None:
            frames_stage.run(
                video=video,
                out_dir=out,
                fps=fps,
                fmt=fmt,                  # type: ignore[arg-type]
                pts_name_decimals=pts_dec,
                rotation=rotation,        # type: ignore[arg-type]
                name_prefix="",           # filenames = "<pts>.ext", no prefix
                progress_cb=_prog,
                log=self._log,
            )

        self._show_progress_bar()
        self._run_async(go, "Video-only frame extraction")

    def _build_smoothers_tab(self, nb: ttk.Notebook) -> None:
        """All 10+ smoothers in one place — checkboxes + Run All.

        UX:
        * Inputs persisted across sessions via :class:`AppState`.
        * Tooltips on every smoother row carry the full description.
        * Drag-and-drop is wired on each file picker.
        * Run button stays disabled until .pos + output dir are set.
        * Results table is sortable; rows are colour-coded
          (gold=best, lightgreen=top-3, salmon=error).
        * Per-smoother live progress streams to the log.
        """
        from .smoothers import describe, list_smoothers

        f = self._make_scrollable_tab(nb, "Smoothers")

        # ── Header card ────────────────────────────────────────────
        header = ttk.Frame(f)
        header.pack(fill="x", padx=14, pady=(14, 6))
        title_row = ttk.Frame(header)
        title_row.pack(fill="x")
        ttk.Label(title_row, text="Smoother comparison",
                  style="Title.TLabel").pack(side="left")
        from ._ui_helpers import StatusPill
        self._sm_status = StatusPill(title_row, initial="idle")
        self._sm_status.pack(side="left", padx=(12, 0))
        ttk.Label(
            header, justify="left", wraplength=900,
            style="Sub.TLabel",
            text=(
                "Run any subset of the 10 shipped smoothers against the "
                "same PPK input. Drop a reference .pos for hRMSE / P95 "
                "vs truth. Best row is highlighted gold."
            ),
        ).pack(anchor="w", pady=(4, 0))

        state = getattr(self, "_state", None)
        if not hasattr(self, "var_sm_pos"):
            self.var_sm_pos = tk.StringVar(
                value=getattr(state, "last_pos_path", "") or "")
            self.var_sm_sensors = tk.StringVar()
            self.var_sm_gt = tk.StringVar()
            self.var_sm_out = tk.StringVar(
                value=getattr(state, "last_out_dir", "") or "")

        def _persist_paths(*_args) -> None:
            st = getattr(self, "_state", None)
            if st is None:
                return
            try:
                st.last_pos_path = self.var_sm_pos.get().strip()
                st.last_out_dir = self.var_sm_out.get().strip()
                st.save()
            except Exception:
                pass

        for v in (self.var_sm_pos, self.var_sm_out):
            v.trace_add("write", _persist_paths)

        def _pick(var, title, *, dirsel=False):
            if dirsel:
                p = filedialog.askdirectory(title=title)
            else:
                p = filedialog.askopenfilename(title=title)
            if p:
                var.set(p)

        inputs_box = ttk.LabelFrame(f, text="Inputs")
        inputs_box.pack(fill="x", padx=10, pady=(8, 4))
        for label, var, picker_title, dirsel, required in [
            (".pos file",                     self.var_sm_pos,     "Pick the PPK .pos file", False, True),
            ("sensors_*.txt (IMU)",           self.var_sm_sensors, "Pick sensors_*.txt",     False, False),
            ("reference .pos",             self.var_sm_gt,      "Pick a reference .pos",  False, False),
            ("output folder",                 self.var_sm_out,     "Pick the output folder", True,  True),
        ]:
            row = ttk.Frame(inputs_box)
            row.pack(fill="x", padx=8, pady=2)
            lbl_text = label + (" *" if required else "  (optional)")
            ttk.Label(row, text=lbl_text, width=30, anchor="w").pack(side="left")
            ent = ttk.Entry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True)
            self._dnd_bind_path(ent, var) if hasattr(self, "_dnd_bind_path") else None
            ttk.Button(row, text="Browse…",
                       command=lambda v=var, t=picker_title, d=dirsel:
                           _pick(v, t, dirsel=d)).pack(side="left", padx=(4, 0))

        # ── Smoother selection grid ────────────────────────────────
        cb_frame = ttk.LabelFrame(f, text="Smoothers")
        cb_frame.pack(fill="x", padx=10, pady=(8, 4))

        self.var_sm_enable = {}
        sorted_names = list_smoothers()
        for i, name in enumerate(sorted_names):
            info = describe(name)
            v = tk.BooleanVar(value=name != "fgo")
            self.var_sm_enable[name] = v
            label_parts = [info.name]
            tags = []
            if info.requires_imu:
                tags.append("IMU")
            if info.optional_dep:
                tags.append(f"opt:{info.optional_dep}")
            if tags:
                label_parts.append(f"[{', '.join(tags)}]")
            label = "  ".join(label_parts)
            cb = ttk.Checkbutton(cb_frame, text=label, variable=v)
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=8, pady=2)
            # Tooltip with full description + benchmark notes.
            try:
                from ._ui_helpers import Tooltip
                tip = info.description
                if info.requires_imu:
                    tip += "\n\nNeeds sensors_*.txt — IMU rows required."
                if info.optional_dep:
                    tip += f"\n\nOptional dep: pip install {info.optional_dep}"
                Tooltip(cb, tip)
            except Exception:
                pass

        # Configure grid columns to share space.
        cb_frame.grid_columnconfigure(0, weight=1, uniform="sm")
        cb_frame.grid_columnconfigure(1, weight=1, uniform="sm")

        # ── Action row ─────────────────────────────────────────────
        act = ttk.Frame(f)
        act.pack(fill="x", padx=10, pady=(8, 4))
        self._sm_run_btn = ttk.Button(
            act, text="▶  Run selected smoothers",
            style="Primary.TButton",
            command=self._sm_run_selected,
        )
        self._sm_run_btn.pack(side="left")
        ttk.Button(act, text="Select all",
                   command=lambda: [v.set(True) for v in self.var_sm_enable.values()]
                   ).pack(side="left", padx=(6, 0))
        ttk.Button(act, text="Clear all",
                   command=lambda: [v.set(False) for v in self.var_sm_enable.values()]
                   ).pack(side="left", padx=(6, 0))
        ttk.Button(act, text="Open output folder",
                   command=self._sm_open_output).pack(side="left", padx=(6, 0))

        # Disable Run button until the two required fields are set.
        def _sync_run_btn(*_args) -> None:
            ok = bool(self.var_sm_pos.get().strip()
                      and self.var_sm_out.get().strip())
            self._sm_run_btn.configure(state=("normal" if ok else "disabled"))
        self.var_sm_pos.trace_add("write", _sync_run_btn)
        self.var_sm_out.trace_add("write", _sync_run_btn)
        _sync_run_btn()

        # ── Results table ──────────────────────────────────────────
        tbl_box = ttk.LabelFrame(f, text="Results")
        tbl_box.pack(fill="both", expand=True, padx=10, pady=(8, 10))
        tbl = ttk.Treeview(
            tbl_box,
            columns=("smoother", "status", "hRMSE", "P95", "runtime", "out"),
            show="headings", height=12,
        )
        # Sortable headers — click toggles ascending/descending.
        self._sm_sort_state: dict[str, bool] = {}

        def _sort_by(col: str) -> None:
            reverse = self._sm_sort_state.get(col, False)
            def _key(row):
                v = tbl.set(row, col)
                # Pull a comparable number out of "2.416 m" / "0.13 s".
                try:
                    return float(v.split()[0])
                except (ValueError, IndexError):
                    return float("inf")
            items = list(tbl.get_children(""))
            items.sort(key=_key, reverse=reverse)
            for i, it in enumerate(items):
                tbl.move(it, "", i)
            self._sm_sort_state[col] = not reverse

        for col, w in [("smoother", 220), ("status", 80),
                       ("hRMSE", 100), ("P95", 100),
                       ("runtime", 90), ("out", 320)]:
            tbl.heading(col, text=col, command=lambda c=col: _sort_by(c))
            tbl.column(col, width=w, anchor="w")
        tbl.tag_configure("best", background="#fff3cd", foreground="#1a1a1a")
        tbl.tag_configure("top3", background="#d4edda", foreground="#1a1a1a")
        tbl.tag_configure("err",  background="#f8d7da", foreground="#a00")
        tbl.pack(fill="both", expand=True, padx=4, pady=4)
        self._sm_tbl = tbl

    def _sm_open_output(self) -> None:
        """Reveal the smoothers/ output folder in the OS file browser."""
        out = self.var_sm_out.get().strip()
        if not out:
            messagebox.showinfo("No output folder", "Pick an output folder first.")
            return
        target = Path(out) / "smoothers"
        if not target.is_dir():
            messagebox.showinfo(
                "Not yet created",
                f"{target}\nwill be created when you run the smoothers.",
            )
            return
        import os as _os, subprocess as _sp, sys as _sys
        try:
            if _sys.platform.startswith("win"):
                _os.startfile(str(target))   # type: ignore[attr-defined]
            elif _sys.platform == "darwin":
                _sp.run(["open", str(target)], check=False)
            else:
                _sp.run(["xdg-open", str(target)], check=False)
        except Exception as e:
            messagebox.showerror("Open folder", f"Could not open:\n{target}\n\n{e}")

    def _sm_run_selected(self) -> None:
        """Worker for the Smoothers tab Run button."""
        from .smoothers import run_smoother

        pos_path = self.var_sm_pos.get().strip()
        out_path = self.var_sm_out.get().strip()
        if not pos_path:
            messagebox.showerror("Missing input", "Pick a .pos file first.")
            return
        if not out_path:
            messagebox.showerror("Missing output", "Pick an output folder.")
            return
        sensors_path = self.var_sm_sensors.get().strip()
        gt_path = self.var_sm_gt.get().strip()
        names = [n for n, v in self.var_sm_enable.items() if v.get()]
        if not names:
            messagebox.showerror("Nothing selected",
                                 "Tick at least one smoother.")
            return

        for it in self._sm_tbl.get_children():
            self._sm_tbl.delete(it)
        if hasattr(self, "_sm_status"):
            self._sm_status.set_state("running")

        # Client-export options (coord systems + Z smoothing + time bases) —
        # read the Tk vars on the main thread; the worker below must not
        # touch widgets. audio_start_utc_s is resolved here too (file I/O,
        # but cheap) so the worker gets plain values only.
        (exp_coords, exp_smooth_z, exp_z_sigma,
         exp_time_bases, exp_audio_utc) = self._build_export_options()
        # Export-source + final-velocity controls (same tab group). All
        # neutral by default -> the client export stays byte-identical.
        (exp_source, exp_emit_fv,
         exp_vel_thr) = self._build_export_source_options()

        def go() -> None:
            from .parsers import parse_imu, parse_rtkpos
            self._log(f"[smoothers] loading PPK {pos_path}...")
            pos_rows = sorted(parse_rtkpos(Path(pos_path)),
                              key=lambda r: r.utc_s)
            self._log(f"[smoothers]   {len(pos_rows)} PPK rows")
            imu_rows = None
            if sensors_path:
                self._log(f"[smoothers] loading IMU {sensors_path}...")
                imu_rows = parse_imu(Path(sensors_path))
                self._log(f"[smoothers]   {len(imu_rows)} IMU rows")
            gt_rows = None
            if gt_path:
                self._log(f"[smoothers] loading GT {gt_path}...")
                gt_rows = sorted(parse_rtkpos(Path(gt_path)),
                                 key=lambda r: r.utc_s)
                self._log(f"[smoothers]   {len(gt_rows)} GT rows")
            out_root = Path(out_path) / "smoothers"
            out_root.mkdir(parents=True, exist_ok=True)

            stat_p = Path(pos_path).with_suffix(
                Path(pos_path).suffix + ".stat")
            stat_p = stat_p if stat_p.is_file() else None

            # Export-source override (user_export.resolve_export_rows):
            # resolved ONCE for the whole run — the same rows feed every
            # smoother's client export. Fail-soft: a failed override logs
            # loudly and falls back to the per-smoother rows rather than
            # aborting the comparison.
            export_rows_override = None
            if exp_source is not None:
                from .stages.user_export import resolve_export_rows
                try:
                    self._log(f"[smoothers] client-export source: "
                              f"{exp_source} (resolve_export_rows)")
                    export_rows_override = resolve_export_rows(
                        pos_rows, source=exp_source, imu_rows=imu_rows,
                        stat_path=stat_p, log=self._log)
                    self._log(f"[smoothers]   export source {exp_source}: "
                              f"{len(export_rows_override)} rows")
                except Exception as _se:
                    self._log(f"[smoothers]   export source {exp_source} "
                              f"FAILED ({type(_se).__name__}: {_se}); "
                              f"falling back to per-smoother rows.")
                    export_rows_override = None

            results = []
            n_total = len(names)
            for idx, name in enumerate(names, start=1):
                self._log(f"[smoothers] ({idx}/{n_total}) running {name}...")
                res = run_smoother(name, pos_rows, imu_rows=imu_rows,
                                   gt_rows=gt_rows, log=self._log,
                                   stat_path=stat_p)
                status = "OK" if res.ok else (res.error_code or "ERR")
                h_str = (f"{res.hrmse_m:.3f} m"
                         if res.hrmse_m is not None else "—")
                p_str = (f"{res.h_p95_m:.3f} m"
                         if res.h_p95_m is not None else "—")
                rt_str = f"{res.runtime_s:.2f} s"
                out_file = ""
                if res.ok:
                    out_file = str(out_root / f"{name}.csv")
                    _write_smoothed_csv(Path(out_file), res.fused)
                    # Client-ready export (stages.user_export): full column
                    # set + chosen coordinate systems + Z smoothing. Written
                    # ALONGSIDE the legacy {name}.csv above (never replaces
                    # it). Failures are logged per-smoother, not fatal.
                    try:
                        from .stages.user_export import (
                            export_kml, export_trajectory,
                        )
                        client_csv = out_root / f"{name}.client.csv"
                        # Export-source override: when set, the client
                        # CSV/KML carry the resolved rows (raw or a chosen
                        # smoother), not this smoother's own output.
                        exp_rows = (list(export_rows_override)
                                    if export_rows_override is not None
                                    else list(res.fused))
                        exp_tag = (exp_source
                                   if export_rows_override is not None
                                   else name)
                        export_trajectory(
                            exp_rows, client_csv,
                            source_tag=exp_tag,
                            coord_systems=exp_coords,
                            smooth_z=exp_smooth_z,
                            z_sigma_s=exp_z_sigma,
                            time_bases=exp_time_bases,
                            audio_start_utc_s=exp_audio_utc,
                            emit_final_velocity=exp_emit_fv,
                            vel_disagree_threshold_mps=exp_vel_thr,
                        )
                        export_kml(
                            exp_rows,
                            out_root / f"{name}.client.kml",
                            name=name,
                            smooth_z=exp_smooth_z,
                            z_sigma_s=exp_z_sigma,
                        )
                        self._log(f"[smoothers]   client export → {client_csv}")
                    except Exception as _xe:
                        self._log(f"[smoothers]   {name}: client export "
                                  f"failed: {_xe}")
                results.append(res)
                tag = "err" if not res.ok else ""
                self.root.after(0, lambda r=res, st=status, h=h_str, p=p_str,
                                rt=rt_str, of=out_file, tg=tag:
                                self._sm_tbl.insert("", "end", values=(
                                    r.name, st, h, p, rt, of),
                                    tags=(tg,) if tg else ()))
                if not res.ok:
                    self._log(f"[smoothers]   {name}: {res.error_message} "
                              f"(code {res.error_code})")
            # Re-tag best + top-3 after all rows are in. Recolour by
            # ascending hRMSE; ties stay in insertion order.
            ranked = [r for r in results if r.ok and r.hrmse_m is not None]
            ranked.sort(key=lambda r: r.hrmse_m)
            def _retag():
                for it in self._sm_tbl.get_children():
                    vals = self._sm_tbl.item(it, "values")
                    name = vals[0]
                    if ranked and name == ranked[0].name:
                        self._sm_tbl.item(it, tags=("best",))
                    elif any(name == r.name for r in ranked[1:3]):
                        self._sm_tbl.item(it, tags=("top3",))
                    elif vals[1] != "OK":
                        self._sm_tbl.item(it, tags=("err",))
            self.root.after(0, _retag)
            if ranked:
                self._log(f"[smoothers] BEST: {ranked[0].name} @ "
                          f"{ranked[0].hrmse_m:.3f} m hRMSE")
            # Flip the status pill — any error = error state, else done.
            final_state = "error" if any(not r.ok for r in results) else "done"
            if hasattr(self, "_sm_status"):
                self.root.after(0, lambda s=final_state:
                                self._sm_status.set_state(s))

        self._run_async(go, "Smoothers comparison")

    # ==================================================================
    # Motion sensor Calibration tab (JOB D)
    # ==================================================================
    def _calib_dir(self) -> "Path":
        """Directory where Motion sensor calibrations are stored (next to app configs)."""
        from pathlib import Path as _P
        try:
            base = _P(self.paths.out_dir) if getattr(self.paths, "out_dir", None) else _P.home()
        except Exception:
            base = _P.home()
        d = base / "imu_calibrations"
        return d

    def _build_imu_calib_tab(self, nb: ttk.Notebook) -> None:
        """Allan-variance Motion sensor calibration + before/after fusion view.

        Sections:
        * Run Allan on a chosen sensors_*.txt -> sigma(tau) plot + params -> Export JSON.
        * Re-upload a saved calibration (enter/confirm device label).
        * "Use calibration in fusion" toggle + BEFORE/AFTER view (default Motion sensor
          noise vs calibration; path + 2-sigma accuracy numbers).

        Thread-safe: all heavy work runs via :meth:`_run_async`; Tk widgets are
        only touched from the main thread via ``self.root.after(0, ...)`` and
        logging goes through the queue (:meth:`_log`).
        """
        f = self._make_scrollable_tab(nb, "IMU Calib")

        # State holder for the most recent calibration object (worker -> main).
        self._calib_current = None  # type: ignore[assignment]

        if not hasattr(self, "var_calib_sensors"):
            self.var_calib_sensors = tk.StringVar()
            self.var_calib_label = tk.StringVar()
            self.var_calib_drive_pos = tk.StringVar()
            self.var_calib_loaded = tk.StringVar()
            self.var_calib_use_in_fusion = tk.BooleanVar(value=False)
            self.var_calib_ba_pos = tk.StringVar()

        # ── Header ─────────────────────────────────────────────────
        header = ttk.Frame(f)
        header.pack(fill="x", padx=14, pady=(14, 6))
        ttk.Label(header, text="IMU Allan-variance calibration",
                  style="Title.TLabel").pack(side="left")
        ttk.Label(
            f, justify="left", wraplength=900, style="Sub.TLabel",
            text=("Compute gyro ARW / accel VRW / bias-instability from a "
                  "static sensors_*.txt (preferred) or by mining stationary "
                  "ZUPT segments from a drive. Keyed by a device label you "
                  "type (the capture app hardcodes the model, so device id is "
                  "unreliable). Export to JSON and feed it into IMU-GNSS "
                  "fusion for a BEFORE/AFTER comparison."),
        ).pack(anchor="w", padx=14, pady=(2, 6))

        def _pick(var, title, *, dirsel=False):
            p = (filedialog.askdirectory(title=title) if dirsel
                 else filedialog.askopenfilename(title=title))
            if p:
                var.set(p)

        # ── 1. Compute calibration ─────────────────────────────────
        box1 = ttk.LabelFrame(f, text="1 · Compute calibration")
        box1.pack(fill="x", padx=10, pady=(6, 4))

        def _file_row(parent, label, var, title, dirsel=False):
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=8, pady=2)
            ttk.Label(row, text=label, width=30, anchor="w").pack(side="left")
            ent = ttk.Entry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True)
            if hasattr(self, "_dnd_bind_path"):
                self._dnd_bind_path(ent, var)
            ttk.Button(row, text="Browse…",
                       command=lambda: _pick(var, title, dirsel=dirsel)
                       ).pack(side="left", padx=(4, 0))

        lbl_row = ttk.Frame(box1)
        lbl_row.pack(fill="x", padx=8, pady=2)
        ttk.Label(lbl_row, text="Device label *", width=30, anchor="w").pack(side="left")
        ttk.Entry(lbl_row, textvariable=self.var_calib_label).pack(
            side="left", fill="x", expand=True)

        _file_row(box1, "Static sensors_*.txt (preferred)",
                  self.var_calib_sensors, "Pick a static sensors_*.txt")
        _file_row(box1, "…or drive .pos (mine ZUPT)",
                  self.var_calib_drive_pos, "Pick a drive .pos with velocity")
        ttk.Label(
            box1, style="Hint.TLabel", wraplength=860, justify="left",
            text=("If you give a drive .pos, also point the static field at the "
                  "drive's sensors_*.txt — stationary segments are mined from "
                  "the .pos velocity and the matching IMU rows are sliced out."),
        ).pack(anchor="w", padx=10, pady=(0, 4))

        btn_row1 = ttk.Frame(box1)
        btn_row1.pack(fill="x", padx=8, pady=(4, 6))
        btn_run = ttk.Button(btn_row1, text="Run Allan analysis",
                             command=self._run_allan_calibration)
        btn_run.pack(side="left")
        self._buttons.append(btn_run)
        btn_plot = ttk.Button(btn_row1, text="Show σ(τ) plot",
                              command=self._show_allan_plot)
        btn_plot.pack(side="left", padx=(6, 0))
        self._buttons.append(btn_plot)
        btn_export = ttk.Button(btn_row1, text="Export JSON…",
                                command=self._export_calibration)
        btn_export.pack(side="left", padx=(6, 0))
        self._buttons.append(btn_export)

        # Params readout.
        self._calib_params_txt = tk.Text(box1, height=9, wrap="none",
                                         background=self._bg_inset,
                                         foreground=self._fg, relief="flat")
        self._calib_params_txt.pack(fill="x", padx=8, pady=(2, 8))
        self._calib_params_txt.insert("end", "No calibration computed yet.\n")
        self._calib_params_txt.configure(state="disabled")

        # ── 2. Re-upload a saved calibration ───────────────────────
        box2 = ttk.LabelFrame(f, text="2 · Re-upload a saved calibration")
        box2.pack(fill="x", padx=10, pady=(6, 4))
        _file_row(box2, "Calibration JSON", self.var_calib_loaded,
                  "Pick a saved imu_calib_*.json")
        btn_row2 = ttk.Frame(box2)
        btn_row2.pack(fill="x", padx=8, pady=(4, 6))
        btn_load = ttk.Button(btn_row2, text="Load & confirm label",
                              command=self._load_calibration_file)
        btn_load.pack(side="left")
        self._buttons.append(btn_load)
        btn_find = ttk.Button(btn_row2, text="Find by device label",
                              command=self._find_calibration_by_label)
        btn_find.pack(side="left", padx=(6, 0))
        self._buttons.append(btn_find)

        # ── 3. Before/After fusion ─────────────────────────────────
        box3 = ttk.LabelFrame(f, text="3 · Use calibration in fusion — BEFORE / AFTER")
        box3.pack(fill="x", padx=10, pady=(6, 10))
        _file_row(box3, "PPK .pos for fusion", self.var_calib_ba_pos,
                  "Pick the PPK .pos to fuse")
        ttk.Checkbutton(box3, text="Use calibration in fusion",
                        variable=self.var_calib_use_in_fusion).pack(
            anchor="w", padx=10, pady=(2, 2))
        btn_row3 = ttk.Frame(box3)
        btn_row3.pack(fill="x", padx=8, pady=(4, 6))
        btn_ba = ttk.Button(btn_row3, text="Run BEFORE/AFTER comparison",
                            command=self._run_before_after_fusion)
        btn_ba.pack(side="left")
        self._buttons.append(btn_ba)
        self._calib_ba_txt = tk.Text(box3, height=10, wrap="none",
                                     background=self._bg_inset,
                                     foreground=self._fg, relief="flat")
        self._calib_ba_txt.pack(fill="x", padx=8, pady=(2, 8))
        self._calib_ba_txt.insert("end", "Run a comparison to see the numbers.\n")
        self._calib_ba_txt.configure(state="disabled")

    def _set_calib_text(self, widget: "tk.Text", content: str) -> None:
        """Replace the contents of a read-only Text widget (main thread only)."""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", content)
        widget.configure(state="disabled")

    def _format_calibration(self, cal) -> str:
        lines = [
            f"device_label : {cal.device_label}",
            f"date        : {cal.date}",
            f"source      : {cal.source}"
            + (f"  ({cal.n_static_segments} ZUPT segments)"
               if cal.source == "mined_zupt" else ""),
            f"sample rate : {cal.sample_rate_hz:.2f} Hz   "
            f"duration: {cal.duration_s:.1f} s   n={cal.n_samples}",
            "",
            f"{'axis':<5}{'random walk':>16}{'bias instab.':>16}{'rate RW':>14}",
        ]
        for axn in ("gx", "gy", "gz", "ax", "ay", "az"):
            if axn in cal.axes:
                a = cal.axes[axn]
                lines.append(
                    f"{axn:<5}{a.random_walk:>16.6g}{a.bias_instability:>16.6g}"
                    f"{a.rate_random_walk:>14.6g}")
        lines.append("")
        lines.append(f"mean gyro ARW : {cal.mean_gyro_arw():.6g} rad/s/sqrt(Hz)")
        lines.append(f"mean accel VRW: {cal.mean_accel_vrw():.6g} (m/s^2)/sqrt(Hz)")
        for w in cal.warnings:
            lines.append(f"! {w}")
        return "\n".join(lines) + "\n"

    def _run_allan_calibration(self) -> None:
        label = self.var_calib_label.get().strip()
        if not label:
            messagebox.showerror("Device label required",
                                 "Enter a device label first (e.g. \"Eli's S23 Ultra\").")
            return
        sensors = self.var_calib_sensors.get().strip()
        drive_pos = self.var_calib_drive_pos.get().strip()
        if not sensors:
            messagebox.showerror("Input required",
                                 "Pick a static sensors_*.txt (or a drive sensors "
                                 "file together with a drive .pos).")
            return

        from pathlib import Path as _P

        def go() -> None:
            from .imu_calibration import compute_calibration
            from .parsers import parse_rtkpos
            self._log(f"[calib] Running Allan analysis on {sensors}")
            if drive_pos:
                self._log(f"[calib] Mining ZUPT segments using {drive_pos}")
                pos_rows = parse_rtkpos(_P(drive_pos))
                cal = compute_calibration(
                    label, drive_imu_path=_P(sensors), drive_pos_rows=pos_rows)
            else:
                cal = compute_calibration(label, static_imu_path=_P(sensors))
            self._calib_current = cal
            txt = self._format_calibration(cal)
            self.root.after(0, lambda: self._set_calib_text(self._calib_params_txt, txt))
            self._log(f"[calib] Done — source={cal.source}, "
                      f"mean accel VRW={cal.mean_accel_vrw():.4g}")

        self._run_async(go, "IMU Allan calibration")

    def _show_allan_plot(self) -> None:
        sensors = self.var_calib_sensors.get().strip()
        if not sensors:
            messagebox.showerror("Input required", "Pick a sensors_*.txt first.")
            return
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            messagebox.showerror("matplotlib missing",
                                 "Install matplotlib to use this feature.")
            return
        from pathlib import Path as _P
        from .parsers import parse_imu
        from .allan import compute_allan

        rows = parse_imu(_P(sensors))
        if not rows:
            messagebox.showerror("Empty file", "No IMU rows parsed.")
            return
        res = compute_allan(rows)

        BG = "#0c0f14"
        fig = Figure(figsize=(10, 6), facecolor=BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        colors = {"gx": "#38b6ff", "gy": "#5cffb1", "gz": "#b69cff",
                  "ax": "#ff5c5c", "ay": "#ffb15c", "az": "#ff5cf0"}
        for axn, a in res.axes.items():
            if a.tau_s.size:
                ax.loglog(a.tau_s, a.sigma, lw=1.1, label=axn,
                          color=colors.get(axn, "#cccccc"))
        ax.set_xlabel("Averaging time τ (s)", color="#cccccc")
        ax.set_ylabel("Allan deviation σ(τ)", color="#cccccc")
        ax.set_title(f"Overlapping Allan deviation — {res.sample_rate_hz:.1f} Hz, "
                     f"{res.duration_s:.0f} s", color="#ffffff", fontsize=10)
        ax.tick_params(colors="#888888")
        ax.grid(True, which="both", color="#1a1a1a", lw=0.6)
        ax.legend(fontsize=8.5, framealpha=0.3, facecolor="#0d1220",
                  edgecolor="#2a3a55", labelcolor="#dddddd", ncol=2)
        fig.tight_layout()

        win = tk.Toplevel(self.root)
        win.title("Allan deviation σ(τ)")
        win.geometry("900x600")
        win.configure(bg=BG)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        win.protocol("WM_DELETE_WINDOW", lambda: (fig.clf(), win.destroy()))

    def _export_calibration(self) -> None:
        cal = getattr(self, "_calib_current", None)
        if cal is None:
            messagebox.showerror("Nothing to export",
                                 "Run an Allan analysis first.")
            return
        from .imu_calibration import save_calibration, _safe_filename
        default_name = f"imu_calib_{_safe_filename(cal.device_label)}.json"
        p = filedialog.asksaveasfilename(
            title="Export calibration JSON", defaultextension=".json",
            initialfile=default_name, filetypes=[("JSON", "*.json")])
        if not p:
            return
        from pathlib import Path as _P
        save_calibration(cal, _P(p))
        self._log(f"[calib] Exported calibration to {p}")
        messagebox.showinfo("Exported", f"Calibration saved to:\n{p}")

    def _load_calibration_file(self) -> None:
        p = self.var_calib_loaded.get().strip()
        if not p:
            p = filedialog.askopenfilename(
                title="Pick a saved calibration JSON",
                filetypes=[("JSON", "*.json")])
            if not p:
                return
            self.var_calib_loaded.set(p)
        from pathlib import Path as _P
        from .imu_calibration import load_calibration
        try:
            cal = load_calibration(_P(p))
        except (FileNotFoundError, ValueError, KeyError) as e:
            messagebox.showerror("Load failed", str(e))
            return
        # Confirm / propagate the device label.
        self.var_calib_label.set(cal.device_label)
        self._calib_current = cal
        self._set_calib_text(self._calib_params_txt, self._format_calibration(cal))
        self._log(f"[calib] Loaded calibration for device label "
                  f"\"{cal.device_label}\" from {p}")
        messagebox.showinfo("Loaded",
                            f"Loaded calibration for:\n{cal.device_label}\n"
                            f"(source: {cal.source})")

    def _find_calibration_by_label(self) -> None:
        label = self.var_calib_label.get().strip()
        if not label:
            messagebox.showerror("Device label required",
                                 "Enter the device label to search for.")
            return
        from .imu_calibration import find_calibration_by_label
        cal = find_calibration_by_label(self._calib_dir(), label)
        if cal is None:
            messagebox.showinfo("Not found",
                                f"No saved calibration matches label "
                                f"\"{label}\" in:\n{self._calib_dir()}")
            return
        self._calib_current = cal
        self._set_calib_text(self._calib_params_txt, self._format_calibration(cal))
        self._log(f"[calib] Found calibration for \"{label}\" "
                  f"(date {cal.date}, source {cal.source}).")

    def _run_before_after_fusion(self) -> None:
        pos_path = self.var_calib_ba_pos.get().strip()
        if not pos_path:
            messagebox.showerror("Input required",
                                 "Pick the PPK .pos to fuse.")
            return
        cal = getattr(self, "_calib_current", None)
        if self.var_calib_use_in_fusion.get() and cal is None:
            messagebox.showerror("No calibration",
                                 "Toggle is on but no calibration is loaded. "
                                 "Run an Allan analysis or load a JSON first.")
            return

        from pathlib import Path as _P

        def go() -> None:
            from .parsers import parse_rtkpos
            from .smoothers import run_smoother
            pos_rows = parse_rtkpos(_P(pos_path))
            if len(pos_rows) < 2:
                self._log("[calib] .pos too short for fusion.")
                return
            self._log("[calib] BEFORE — fusion with default IMU process noise")
            before = run_smoother("epoch_weight_v2", pos_rows, imu_rows=None)
            self._log("[calib] AFTER  — fusion with calibration applied")
            after = run_smoother("epoch_weight_v2", pos_rows, imu_rows=None,
                                 calibration=cal)
            report = self._before_after_report(before, after, cal)
            self.root.after(0, lambda: self._set_calib_text(self._calib_ba_txt, report))
            self._log("[calib] BEFORE/AFTER comparison complete.")

        self._run_async(go, "IMU calibration BEFORE/AFTER")

    @staticmethod
    def _traj_2sigma(fused) -> tuple:
        """Return (mean_2sigma_h_m, p95_2sigma_h_m) from a fused row list.

        Uses each row's reported horizontal sd (sd_n/sd_e) when present,
        scaled to 2-sigma; falls back to NaN when no covariance is available.
        """
        import math as _m
        vals = []
        for r in fused:
            sdn = getattr(r, "sd_n", float("nan"))
            sde = getattr(r, "sd_e", float("nan"))
            if _m.isfinite(sdn) and _m.isfinite(sde):
                vals.append(2.0 * _m.hypot(sdn, sde))
        if not vals:
            return (float("nan"), float("nan"))
        vals.sort()
        mean = sum(vals) / len(vals)
        p95 = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
        return (mean, p95)

    def _before_after_report(self, before, after, cal) -> str:
        import math as _m
        # Horizontal path divergence between the two fused tracks.
        max_div = 0.0
        rms_div = 0.0
        nb = min(len(before.fused), len(after.fused))
        if nb:
            from .geo import llh_to_ecef, ecef_to_enu
            ref = (before.fused[0].lat_deg, before.fused[0].lon_deg,
                   before.fused[0].h_m)
            acc = 0.0
            for i in range(nb):
                bx, by, bz = llh_to_ecef(before.fused[i].lat_deg,
                                         before.fused[i].lon_deg,
                                         before.fused[i].h_m)
                be, bn, _ = ecef_to_enu(bx, by, bz, ref)
                ax_, ay_, az_ = llh_to_ecef(after.fused[i].lat_deg,
                                            after.fused[i].lon_deg,
                                            after.fused[i].h_m)
                ae, an, _ = ecef_to_enu(ax_, ay_, az_, ref)
                d = _m.hypot(ae - be, an - bn)
                max_div = max(max_div, d)
                acc += d * d
            rms_div = _m.sqrt(acc / nb)

        b_mean, b_p95 = self._traj_2sigma(before.fused)
        a_mean, a_p95 = self._traj_2sigma(after.fused)
        lines = [
            "BEFORE = default IMU process noise   AFTER = calibration applied",
            "",
            f"calibration : {(cal.device_label + ' / ' + cal.source) if cal else '(none — toggle off)'}",
            f"epochs      : before={before.n_output}  after={after.n_output}",
            "",
            f"{'metric':<28}{'BEFORE':>14}{'AFTER':>14}",
            f"{'mean 2σ horiz (m)':<28}{b_mean:>14.3f}{a_mean:>14.3f}",
            f"{'p95 2σ horiz (m)':<28}{b_p95:>14.3f}{a_p95:>14.3f}",
            "",
            f"trajectory divergence  RMS={rms_div:.3f} m   max={max_div:.3f} m",
        ]
        if before.h_p95_m is not None and after.h_p95_m is not None:
            lines += [
                "",
                f"{'hRMSE vs reference (m)':<28}"
                f"{(before.hrmse_m or float('nan')):>14.3f}"
                f"{(after.hrmse_m or float('nan')):>14.3f}",
                f"{'p95 vs reference (m)':<28}"
                f"{before.h_p95_m:>14.3f}{after.h_p95_m:>14.3f}",
            ]
        return "\n".join(lines) + "\n"

    def _build_analysis_tab(self, nb: ttk.Notebook) -> None:
        """One-file Post-processing "Analysis" report: pick a subject .pos (+ optional
        subject/base .obs and ground-truth .pos), build a self-contained
        HTML with quality / source / SNR / fine measurements / noise /
        predicted-accuracy panels, and open it in the browser."""
        f = self._make_scrollable_tab(nb, "Analysis")

        if not hasattr(self, "var_an_pos"):
            self.var_an_pos = tk.StringVar()
            self.var_an_rover_obs = tk.StringVar()
            self.var_an_base_obs = tk.StringVar()
            self.var_an_gt_pos = tk.StringVar()
            self.var_an_out = tk.StringVar()

        ttk.Label(
            f,
            text=(
                "Summarize a PPK solution and its raw data in one offline HTML "
                "report: solution quality (Q) distribution, satellites over time, "
                "SNR per constellation (rover vs base), carrier-phase check, "
                "noise/precision and a predicted-accuracy panel (validated "
                "against a ground-truth .pos when one is given)."
            ),
            foreground="#888", wraplength=920, justify="left",
        ).pack(anchor="w", padx=10, pady=(8, 4))

        def _pick_an(var, title, *, dirsel=False):
            p = (filedialog.askdirectory(title=title) if dirsel
                 else filedialog.askopenfilename(title=title))
            if p:
                var.set(p)

        def _an_file_row(parent, label, var, title, tip, dirsel=False):
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=8, pady=2)
            ttk.Label(row, text=label, width=28, anchor="w").pack(side="left")
            ent = ttk.Entry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True)
            if hasattr(self, "_dnd_bind_path"):
                self._dnd_bind_path(ent, var)
            ttk.Button(row, text="Browse…",
                       command=lambda: _pick_an(var, title, dirsel=dirsel)
                       ).pack(side="left", padx=(4, 0))
            _Tooltip(ent, tip)

        box = ttk.LabelFrame(f, text="Inputs")
        box.pack(fill="x", padx=10, pady=(6, 4))
        _an_file_row(box, "Rover .pos  (required)", self.var_an_pos,
                     "Pick the rover PPK .pos",
                     "The RTKLIB .pos solution to analyse.\n"
                     "Old (DAY12-era) and new headers both work.")
        _an_file_row(box, "Rover .obs  (optional)", self.var_an_rover_obs,
                     "Pick the rover RINEX .obs",
                     "Enables the SNR panel and the carrier-phase check\n"
                     "(detects duty-cycled captures where RTK is impossible).")
        _an_file_row(box, "Base .obs  (optional)", self.var_an_base_obs,
                     "Pick the base RINEX .obs",
                     "Adds a ROVER vs BASE SNR comparison.")
        _an_file_row(box, "Ground-truth .pos  (optional)", self.var_an_gt_pos,
                     "Pick a ground-truth .pos",
                     "A survey-grade reference track. Adds a measured\n"
                     "error CDF and an error-vs-satellites panel\n"
                     "(rover epochs matched to truth by nearest time).")
        _an_file_row(box, "Output folder  (optional)", self.var_an_out,
                     "Pick the output folder",
                     "Where analysis_report.html is written.\n"
                     "Defaults to the .pos folder.", dirsel=True)

        btn_row = ttk.Frame(f)
        btn_row.pack(fill="x", padx=10, pady=(6, 10))
        btn_an = ttk.Button(btn_row, text="Build analysis report",
                            command=self._run_analysis_report)
        btn_an.pack(side="left")
        self._buttons.append(btn_an)
        _Tooltip(btn_an,
                 "Builds a single self-contained analysis_report.html\n"
                 "(plotly inlined — opens offline) and opens it in the browser.")

        # ── Media + stream export (combine_av) ────────────────────────────
        g_av = ttk.LabelFrame(f, text="Video + audio export")
        g_av.pack(fill="x", padx=10, pady=(4, 10))
        ttk.Label(
            g_av,
            text=(
                "Mux the session's global audio onto the FULL recording or "
                "any trimmed/cropped chop slice, synced on device boottime "
                "(audio crystal drift corrected). Uses the RAW session "
                "loaded on the Inputs tab, or asks for a folder."
            ),
            foreground="#888", wraplength=920, justify="left",
        ).pack(anchor="w", padx=8, pady=(6, 2))
        btn_av = ttk.Button(
            g_av, text="Export video + audio (full or crop)",
            command=self._run_export_av,
        )
        btn_av.pack(anchor="w", padx=8, pady=(2, 8))
        self._buttons.append(btn_av)
        _Tooltip(btn_av,
                 "Discovers every combinable video in the session "
                 "(recording_*.mp4 + chop_*/ slices), lets you pick one when\n"
                 "there are several, then runs ffmpeg to write "
                 "combined_<clip>.mp4 with the global audio track attached\n"
                 "at the boottime-correct offset. Opens the result when done.")

        # ── Camera-model accuracy (photo_compare) ─────────────────────────
        g_cam = ttk.LabelFrame(f, text="Camera-model accuracy")
        g_cam.pack(fill="x", padx=10, pady=(4, 10))
        ttk.Label(
            g_cam,
            text=(
                "Compare a COLMAP camera-model reconstruction against the "
                "GPS trajectory and a ground-truth .pos: per-frame error "
                "tables, alignment stats and a verdict, written as a CSV + "
                "self-contained HTML report."
            ),
            foreground="#888", wraplength=920, justify="left",
        ).pack(anchor="w", padx=8, pady=(6, 2))
        btn_cam = ttk.Button(
            g_cam, text="Camera-model accuracy report",
            command=self._run_camera_report,
        )
        btn_cam.pack(anchor="w", padx=8, pady=(2, 8))
        self._buttons.append(btn_cam)
        _Tooltip(btn_cam,
                 "Asks for the COLMAP images.txt/.bin (or the reconstruction\n"
                 "folder) and a ground-truth .pos, uses the pipeline's\n"
                 "Georef.csv when one was built this session (asks\n"
                 "otherwise), then writes camera_vs_gps_vs_gt.csv + an\n"
                 "offline HTML report and opens it in the browser.")

    def _run_export_av(self) -> None:
        """'Export media + stream (full or crop)' — mux via combine_av.

        Clip discovery + the (modal) chooser run on the Tk main thread;
        the actual plan+the external converter mux runs on the worker via ``_run_async``
        so the UI stays live. Errors are logged / message-boxed by the
        ``_run_async`` machinery — never crash the UI.
        """
        from . import combine_av

        session = self.paths.raw_folder
        if session is None:
            s = filedialog.askdirectory(
                title="Pick the RAW session folder "
                      "(contains recording_*.mp4 + audio_*.wav)")
            if not s:
                return
            session = Path(s)

        try:
            clips = combine_av.discover_videos(session)
        except Exception as e:
            self._log(f"[combine-av] discover failed: {type(e).__name__}: {e}")
            messagebox.showerror(
                "Export video + audio",
                f"Could not scan {session}:\n{e}")
            return
        if not clips:
            messagebox.showerror(
                "Export video + audio",
                f"No combinable videos found in\n{session}\n"
                "(no recording_*.mp4 and no chop_*/ slices).")
            return

        if len(clips) == 1:
            chosen = clips[0]
        else:
            chosen = self._choose_av_clip(clips)
            if chosen is None:
                return  # user cancelled

        self._log(f"[combine-av] session: {session}")
        self._log(f"[combine-av] clip: [{chosen.kind}] {chosen.label}")

        def go() -> None:
            plan = combine_av.plan_mux(session, which=chosen)
            self._log(f"[combine-av] audio_seek_s = {plan.audio_seek_s:+.6f} s"
                      f"   drift = {plan.ppm:+.2f} ppm"
                      + (f"   atempo = {plan.atempo:.9f}" if plan.atempo else ""))
            self._log(f"[combine-av] out: {plan.out_path}")
            for w in plan.warnings:
                self._log(f"[combine-av] WARN: {w}")
            self._log("[combine-av] ffmpeg: "
                      + " ".join(str(a) for a in plan.ffmpeg_cmd))
            out = combine_av.run_mux(plan)
            self._log(f"[combine-av] wrote {out}")
            self.root.after(0, lambda p=out: self._open_path_in_default(p))

        self._run_async(go, "Export video + audio")

    def _choose_av_clip(self, clips):
        """Modal listbox chooser over ``combine_av.ClipInfo`` entries.

        Returns the chosen ClipInfo, or ``None`` when cancelled. Main
        thread only (creates a Toplevel and blocks in ``wait_window``).
        """
        win = tk.Toplevel(self.root)
        win.title("Choose video to export")
        win.transient(self.root)
        win.grab_set()
        ttk.Label(
            win,
            text=("This session has several combinable videos.\n"
                  "Pick the FULL recording or a trimmed chop slice:"),
            justify="left",
        ).pack(anchor="w", padx=10, pady=(10, 4))
        lb = tk.Listbox(win, height=min(12, max(4, len(clips))),
                        width=70, exportselection=False)
        for c in clips:
            lb.insert("end", f"[{c.kind.upper()}]  {c.label}")
        lb.selection_set(0)
        lb.pack(fill="both", expand=True, padx=10, pady=4)

        chosen: "list" = []

        def _ok(_evt=None):
            sel = lb.curselection()
            if sel:
                chosen.append(clips[int(sel[0])])
            win.destroy()

        row = ttk.Frame(win)
        row.pack(fill="x", padx=10, pady=(4, 10))
        ttk.Button(row, text="Export", style="Accent.TButton",
                   command=_ok).pack(side="left")
        ttk.Button(row, text="Cancel",
                   command=win.destroy).pack(side="left", padx=(6, 0))
        lb.bind("<Double-Button-1>", _ok)
        lb.focus_set()
        win.wait_window()
        return chosen[0] if chosen else None

    def _run_analysis_report(self) -> None:
        pos_s = self.var_an_pos.get().strip()
        if not pos_s or not Path(pos_s).is_file():
            messagebox.showerror(
                "Rover .pos required",
                "Pick the rover .pos file to analyse first.")
            return

        def _opt(var: "tk.StringVar") -> Optional[Path]:
            s = var.get().strip()
            if not s:
                return None
            p = Path(s)
            if not p.is_file():
                self._log(f"[analysis] warning: {p} not found — skipped.")
                return None
            return p

        pos = Path(pos_s)
        rover_obs = _opt(self.var_an_rover_obs)
        base_obs = _opt(self.var_an_base_obs)
        gt_pos = _opt(self.var_an_gt_pos)
        out_s = self.var_an_out.get().strip()
        out_dir = Path(out_s) if out_s else pos.parent
        out_html = out_dir / "analysis_report.html"

        def go() -> None:
            from .analysis_report import build_analysis_report
            build_analysis_report(
                pos, out_html,
                rover_obs=rover_obs, base_obs=base_obs, gt_pos=gt_pos,
                log=self._log)
            self.root.after(0, lambda: self._open_path_in_default(out_html))

        self._run_async(go, "Analysis report")

    def _run_camera_report(self) -> None:
        """'Camera-model accuracy report' — ``photo_compare.build_report``.

        All file dialogs run on the Tk main thread BEFORE the worker starts;
        the parse/align/report work runs on the worker via ``_run_async``
        (errors are logged + message-boxed there — never crash the UI).
        The finished HTML opens via ``_open_path_in_default``.
        """
        # 1) COLMAP input: images.txt/.bin, or (Cancel) a reconstruction dir.
        colmap = filedialog.askopenfilename(
            title=("Pick the COLMAP images.txt / images.bin "
                   "(Cancel to pick a reconstruction folder instead)"),
            filetypes=[("COLMAP images", "images.txt images.bin"),
                       ("All files", "*.*")],
        )
        if not colmap:
            colmap = filedialog.askdirectory(
                title=("Pick the COLMAP reconstruction folder "
                       "(contains images.txt or images.bin)"))
        if not colmap:
            self._log("[camrep] cancelled (no COLMAP input).")
            return

        # 2) Ground-truth .pos.
        gt = filedialog.askopenfilename(
            title="Pick the ground-truth .pos",
            filetypes=[(".pos files", "*.pos"), ("All files", "*.*")],
        )
        if not gt:
            self._log("[camrep] cancelled (no ground-truth .pos).")
            return

        # 3) GPS per-frame positions: prefer the Georef.csv the pipeline
        #    built this session; otherwise ask. Cancelling falls back to
        #    interpolating the loaded rover .pos at frame times recovered
        #    from the RAW session / extracted_frame_times.csv.
        georef: "Optional[Path]" = None
        if (self.paths.georef_csv is not None
                and Path(self.paths.georef_csv).is_file()):
            georef = Path(self.paths.georef_csv)
            self._log(f"[camrep] using session Georef.csv: {georef}")
        else:
            s = filedialog.askopenfilename(
                title=("Pick the Georef CSV (Cancel to recover frame "
                       "times from the loaded RAW session instead)"),
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            georef = Path(s) if s else None
        session_dir = self.paths.raw_folder if georef is None else None
        pos_path = self.paths.pos_path if georef is None else None
        frame_times = (self.paths.frame_times_csv
                       if georef is None else None)
        if georef is None and pos_path is None:
            messagebox.showerror(
                "Camera-model accuracy report",
                "Need a Georef.csv, or a loaded session with a rover .pos\n"
                "(run/load the PPK solution first) so per-frame GPS\n"
                "positions can be interpolated.")
            return

        # 4) Output dir: current out dir when known, else the Analysis
        #    tab's output folder, else next to the ground truth.
        out_s = self.var_an_out.get().strip()
        out_base = (self.paths.out_dir
                    or (Path(out_s) if out_s else Path(gt).parent))
        out_dir = Path(out_base) / "camera_report"
        colmap_p, gt_p = Path(colmap), Path(gt)

        def go() -> None:
            from .photo_compare import build_report
            out_dir.mkdir(parents=True, exist_ok=True)
            res = build_report(
                colmap_p, gt_p, out_dir,
                georef_csv=georef, pos=pos_path,
                frame_times=frame_times, session_dir=session_dir,
                log=self._log,
            )
            if getattr(res, "verdict", None):
                self._log(f"[camrep] {res.verdict}")
            target = res.html_path or res.csv_path
            if target is not None:
                self.root.after(
                    0, lambda p=target: self._open_path_in_default(p))

        self._run_async(go, "Camera-model accuracy report")

    def _build_viewers_tab(self, nb: ttk.Notebook) -> None:
        f = self._make_scrollable_tab(nb, "Viewers")

        ttk.Label(
            f,
            text=(
                "Self-contained HTML viewers (fully offline).  "
                "Plotly is vendored alongside each .html so the output folder is portable."
            ),
            foreground="#888", wraplength=900,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 4))

        grp = ttk.LabelFrame(f, text="Build Viewers")
        grp.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
        grp.columnconfigure(0, weight=1)

        btn1 = ttk.Button(grp, text="Build  trajectory_viewer.html",
                          command=self._run_traj_viewer)
        btn1.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 3))
        _Tooltip(btn1, "Interactive 3-D trajectory map (Plotly).\n"
                 "Shows PPK path coloured by quality (fix/float/single).\n"
                 "Requires: georef.csv.")

        btn2 = ttk.Button(grp, text="Build  orientation_panel.html",
                          command=self._run_orient_panel)
        btn2.grid(row=1, column=0, sticky="w", padx=10, pady=3)
        _Tooltip(btn2, "Time-series panel showing yaw, pitch, roll and their raw vs smoothed versions.\n"
                 "Requires: RAW folder (.pos file + data log).")

        btn3 = ttk.Button(grp, text="Build  comparison_viewer.html  "
                          "(toggle smoothing profiles)",
                          command=self._run_compare_viewer)
        btn3.grid(row=2, column=0, sticky="w", padx=10, pady=3)
        _Tooltip(btn3, "Side-by-side trajectory viewer: toggle between smoothing profiles.\n"
                 "Requires: RAW folder, .pos file, frame times CSV.")

        btn4 = ttk.Button(grp, text="Build  sync_player.html  "
                          "(video + trajectory)",
                          command=self._run_sync_player)
        btn4.grid(row=3, column=0, sticky="w", padx=10, pady=3)
        _Tooltip(btn4, "Synchronized video + trajectory player.\n"
                 "Scrub the video; the map pans to the matching frame position.\n"
                 "Includes velocity HUD overlay (Vel HUD button in player).\n"
                 "Requires: RAW folder, .pos file, frame times CSV.")

        btn5 = ttk.Button(grp, text="Build  velocity_viewer.html  "
                          "(Doppler speed timeline)",
                          command=self._run_vel_viewer)
        btn5.grid(row=4, column=0, sticky="w", padx=10, pady=(3, 4))
        _Tooltip(btn5, "Offline Plotly viewer: Doppler speed vs coordinate-derived speed + azimuth.\n"
                 "Disagreement = |Doppler − Coords| — large values predict position outliers.\n"
                 "Requires: .pos file.")

        btn6 = ttk.Button(grp, text="Build  geo_viewer.html  "
                          "(2D satellite + 3D terrain)",
                          command=self._run_geo_viewer)
        btn6.grid(row=5, column=0, sticky="w", padx=10, pady=(4, 4))
        _Tooltip(btn6, "Two-tab offline viewer:\n"
                 "  • 2D tab: PPK trajectory over satellite basemap (toggle on/off).\n"
                 "  • 3D tab: DSM terrain surface + trajectory in 3D.\n"
                 "Requires: .pos file.  Basemap/DSM GeoTIFFs optional (needs rasterio).")

        btn7 = ttk.Button(grp, text="Build  speed_azimuth_vs_gt.html  "
                          "(device vs reference)",
                          command=self._run_speed_vs_gt_viewer)
        btn7.grid(row=6, column=0, sticky="w", padx=10, pady=(4, 8))
        _Tooltip(btn7, "Compare device Doppler / Coords-Δ speed and azimuth against a "
                 "reference .pos file (e.g. surveyed reference rover).\n"
                 "Statistics: bias / RMSE / MAE / std / σ1 / σ2 / σ3 / P50 / P95 / max.\n"
                 "Azimuth stats split into static vs moving bands.\n"
                 "Requires: device .pos (in main pipeline state)  +  GT .pos (file picker).")

        btn8 = ttk.Button(grp, text="Export  KML batch  "
                          "(smoothing comparison set)",
                          command=self._run_kml_batch)
        btn8.grid(row=7, column=0, sticky="w", padx=10, pady=(4, 2))
        btn_all_kml = ttk.Button(grp, text="Export  ALL smoother KMLs  "
                                 "(every registered smoother)",
                                 command=self._run_all_smoother_kmls)
        btn_all_kml.grid(row=8, column=0, sticky="w", padx=10, pady=(2, 8))
        _Tooltip(btn_all_kml,
                 "Run every registered smoother (raw_ppk, gaussian_car, cv_rts_pv, "
                 "epoch_weight, ekf_smoothed, fgo, …) and export one KML per smoother.\n"
                 "Each smoother gets a distinct line colour.\n"
                 "IMU-based smoothers (ekf, fgo) need sensors_*.txt.\n"
                 "Output: out_dir/kml_smoothers/*.kml")
        _Tooltip(btn8, "Drop a folder of up to 7 .kml files into the output dir:\n"
                 "  • ppk_raw.kml             — PPK epochs, line segmented and coloured by RTKLIB Q\n"
                 "  • ppk_gauss_gentle.kml    — Gaussian smoothing (0.5s xy, 2s z)\n"
                 "  • ppk_gauss_car.kml       — Gaussian smoothing (2s xy, 10s z)\n"
                 "  • ppk_gauss_aggressive.kml — Gaussian smoothing (5s xy, 20s z)\n"
                 "  • data_gnss_raw.kml      — Android raw GPS provider track\n"
                 "  • data_flp.kml           — Fused-Location Provider (IMU-blended)\n"
                 "  • fused_bent.kml          — FLP shape warped onto PPK anchors (fused_bend.py)\n"
                 "Layers whose source data is missing are silently skipped.\n"
                 "Drop the folder into Google Earth; toggle layers in the sidebar.\n"
                 "Requires: .pos file. Measurements file enables the device-derived layers.")

        btn_capdiag = ttk.Button(grp, text="Build  capture_diag.html  "
                                 "(A/V<->GNSS sync, trim, video stats)",
                                 command=self._run_capture_diag)
        btn_capdiag.grid(row=9, column=0, sticky="w", padx=10, pady=(2, 8))
        _Tooltip(btn_capdiag,
                 "Capture diagnostics: audio->GNSS and video->GNSS sync offset/drift,\n"
                 "video trim (head/tail/total + % kept), resolution, fps, MB/min, focal length.\n"
                 "Requires: RAW folder. .pos + frame times optional (improves trim).")

        # ---- Client diagnostic viewers (post-Post-processing, lightweight) ----
        diag = ttk.LabelFrame(f, text="Client diagnostic viewers (post-PPK)")
        diag.grid(row=9, column=0, columnspan=3, sticky="ew",
                  padx=8, pady=(4, 4))
        diag.columnconfigure(0, weight=1)
        cb1 = ttk.Button(
            diag, text="Smoother comparison  (all routes overlaid)",
            command=self._run_client_compare)
        cb1.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 3))
        _Tooltip(cb1, "Overlays raw / Gauss-car / K_smart / ADAPTIVE / "
                 "cv_rts_pv / epoch_weighted on one plot. Toggle traces "
                 "via legend. Requires: .pos.")
        cb2 = ttk.Button(
            diag, text="Quality panel  (ns / speed / sigma / Q)",
            command=self._run_client_quality)
        cb2.grid(row=1, column=0, sticky="w", padx=10, pady=3)
        _Tooltip(cb2, "Stacked timeline: ns_used, Doppler speed, "
                 "sigma_h / sigma_v, RTKLIB Q flag.")
        cb3 = ttk.Button(
            diag, text="Raw vs Kalman diff  (where the smoother worked)",
            command=self._run_client_diff)
        cb3.grid(row=2, column=0, sticky="w", padx=10, pady=3)
        _Tooltip(cb3, "Per-epoch horizontal + vertical delta between raw "
                 ".pos and cleaned (Kalman) .pos. Needs both files.")
        cb_trust = ttk.Button(
            diag, text="Trust pane  (position + velocity confidence)",
            command=self._run_trust_pane)
        cb_trust.grid(row=3, column=0, sticky="w", padx=10, pady=3)
        _Tooltip(cb_trust, "Color-coded trajectory showing trusted vs untrusted regions.\n"
                 "Position trust: eff_sig < 0.85 -> h <= 10m (hard ceiling)\n"
                 "Velocity trust: |raw-v2| < 4.0 -> v <= 1.2 m/s @ 2sigma\n"
                 "Validated across 18 sessions (6 days, 9 devices).\n"
                 "Requires: .pos file.")
        cb_trust_v2 = ttk.Button(
            diag, text="Trust v2  (disagree-based accuracy labels)",
            command=self._run_trust_v2_pane)
        cb_trust_v2.grid(row=4, column=0, sticky="w", padx=10, pady=3)
        _Tooltip(cb_trust_v2,
                 "Disagree-based trust labels validated on 10 sessions:\n"
                 "GREEN: pos <= 3.1m + vel <= 1.2 m/s @ 2sigma (~50%)\n"
                 "ORANGE: pos <= 6.3m @ 2sigma (~49%)\n"
                 "RED: no guarantee (~1%)\n"
                 "Requires: .pos file.")
        cb_vio = ttk.Button(
            diag, text="VIO trajectory overlay  (SLOW: ~3-5 min)",
            command=self._run_client_vio)
        cb_vio.grid(row=5, column=0, sticky="w", padx=10, pady=3)
        _Tooltip(cb_vio, "Monocular VIO on recording_*.mp4; integrated "
                 "VIO trajectory overlaid on PPK. SLOW. Requires: RAW "
                 "folder + .pos.")
        cb4 = ttk.Button(
            diag, text="Open in RTKPlot  (RTKLIB native viewer)",
            command=self._run_client_rtkplot)
        cb4.grid(row=5, column=0, sticky="w", padx=10, pady=(3, 8))
        _Tooltip(cb4, "Launches bundled rtkplot.exe with rover .obs, "
                 "base .obs, .pos, .pos.stat preloaded.")

        basemap_grp = ttk.LabelFrame(f, text="Optional Basemap  (sync_player)")
        basemap_grp.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
        basemap_grp.columnconfigure(1, weight=1)

        self.var_sync_basemap = tk.StringVar()
        ttk.Label(basemap_grp, text="GeoTIFF basemap", foreground="#888").grid(
            row=0, column=0, sticky="e", padx=8, pady=6)
        ttk.Entry(basemap_grp, textvariable=self.var_sync_basemap, width=50).grid(
            row=0, column=1, sticky="ew", padx=8, pady=6)

        def _browse_basemap() -> None:
            p = filedialog.askopenfilename(
                filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
            if p:
                self.var_sync_basemap.set(p)

        ttk.Button(basemap_grp, text="Browse...", command=_browse_basemap).grid(
            row=0, column=2, sticky="w", padx=(0, 8))
        _Tooltip(basemap_grp, "Optional GeoTIFF background map for the sync player.\n"
                 "Requires rasterio at build time; rendered as a static PNG overlay.\n"
                 "If omitted the player uses an OpenStreetMap tile fallback (online).")

        self.var_video_bias_ms = tk.StringVar(value="0")
        ttk.Label(basemap_grp, text="Video-GNSS bias (ms)", foreground="#888").grid(
            row=1, column=0, sticky="e", padx=8, pady=4)
        ttk.Entry(basemap_grp, textvariable=self.var_video_bias_ms, width=10).grid(
            row=1, column=1, sticky="w", padx=8, pady=4)
        _Tooltip(basemap_grp,
                 "Time offset between video and GNSS in milliseconds.\n"
                 "Positive = video ahead of GNSS, negative = GNSS ahead.\n"
                 "Adjustable live in the sync player UI after building.")

        ttk.Label(
            f,
            foreground="#777", wraplength=900, font=("Segoe UI", 8),
            text=(
                "sync_player needs the original .mp4 at a relative path.  "
                "For best results keep the .html in the same folder as the .mp4.  "
                "If opened from a different location it falls back to a file:// URL "
                "which some browsers block — copy the .mp4 next to the .html in that case."
            ),
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=10, pady=(6, 2))

        geo_grp = ttk.LabelFrame(f, text="Geo viewer  (optional GeoTIFFs)")
        geo_grp.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
        geo_grp.columnconfigure(1, weight=1)

        self.var_geo_basemap = tk.StringVar()
        self.var_geo_dsm     = tk.StringVar()

        ttk.Label(geo_grp, text="Basemap GeoTIFF", foreground="#888").grid(
            row=0, column=0, sticky="e", padx=8, pady=5)
        ttk.Entry(geo_grp, textvariable=self.var_geo_basemap, width=50).grid(
            row=0, column=1, sticky="ew", padx=8, pady=5)

        def _browse_geo_basemap() -> None:
            p = filedialog.askopenfilename(
                filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
            if p:
                self.var_geo_basemap.set(p)

        ttk.Button(geo_grp, text="Browse...", command=_browse_geo_basemap).grid(
            row=0, column=2, sticky="w", padx=(0, 8))

        ttk.Label(geo_grp, text="DSM GeoTIFF", foreground="#888").grid(
            row=1, column=0, sticky="e", padx=8, pady=5)
        ttk.Entry(geo_grp, textvariable=self.var_geo_dsm, width=50).grid(
            row=1, column=1, sticky="ew", padx=8, pady=5)

        def _browse_geo_dsm() -> None:
            p = filedialog.askopenfilename(
                filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
            if p:
                self.var_geo_dsm.set(p)

        ttk.Button(geo_grp, text="Browse...", command=_browse_geo_dsm).grid(
            row=1, column=2, sticky="w", padx=(0, 8))

        _Tooltip(geo_grp, "Both fields optional.\n"
                 "Basemap: satellite/aerial GeoTIFF (WGS84 or any CRS rasterio can reproject).\n"
                 "DSM: single-band elevation GeoTIFF (e.g. Copernicus DEM 30m).\n"
                 "Requires rasterio; omit to build trajectory-only viewer.")

        self._buttons.extend([btn1, btn2, btn3, btn4, btn5, btn6, btn7])
        f.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_path(
        self,
        parent: tk.Widget,
        row: int,
        *,
        label: str,
        var: tk.StringVar,
        kind: str,
        on_change: Optional[Callable[[], None]] = None,
        file_types: Optional[list[tuple[str, str]]] = None,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=8, pady=4)
        e = ttk.Entry(parent, textvariable=var)
        e.grid(row=row, column=1, sticky="ew", padx=8)

        def browse() -> None:
            if kind == "dir":
                p = filedialog.askdirectory()
            elif kind == "file_open":
                p = filedialog.askopenfilename(
                    filetypes=file_types or [("All", "*.*")])
            else:
                p = filedialog.asksaveasfilename(
                    filetypes=file_types or [("All", "*.*")])
            if p:
                var.set(p)
                if on_change:
                    on_change()

        ttk.Button(parent, text="Browse...", command=browse).grid(
            row=row, column=2, sticky="w", padx=(0, 8))
        if on_change:
            var.trace_add("write", lambda *_: on_change())
        return e

    def _labelled_entry(
        self, parent: tk.Widget, row: int, label: str, var: tk.StringVar
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=8, pady=2)
        ttk.Entry(parent, textvariable=var, width=24).grid(
            row=row, column=1, sticky="w", padx=8)

    # ------------------------------------------------------------------
    # Drag-and-drop, recent projects, open-folder helpers
    # ------------------------------------------------------------------

    def _register_dnd_folder(
        self, widget: tk.Widget, var: tk.StringVar,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a folder drop target on ``widget`` if tkinterdnd2 is present.

        Accepts both folder drops and (gracefully) file drops, normalising to
        the parent directory in the latter case.
        """
        if self._dnd_files is None:
            return

        def _handle(event: "tk.Event") -> None:  # type: ignore[type-arg]
            data = event.data or ""
            paths = self.root.tk.splitlist(data)  # type: ignore[attr-defined]
            if not paths:
                return
            p = Path(paths[0])
            if p.is_file():
                p = p.parent
            var.set(str(p))
            if on_change:
                on_change()

        try:
            widget.drop_target_register(self._dnd_files)  # type: ignore[attr-defined]
            widget.dnd_bind("<<Drop>>", _handle)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _load_recent(self) -> list[str]:
        try:
            data = json.loads(_RECENT_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [str(x) for x in data if isinstance(x, str)][:_RECENT_MAX]

    def _save_recent(self) -> None:
        try:
            _RECENT_FILE.write_text(
                json.dumps(self._recent, indent=2), encoding="utf-8",
            )
        except OSError:
            pass

    def _push_recent(self, folder: str) -> None:
        if not folder:
            return
        try:
            norm = str(Path(folder).resolve())
        except OSError:
            norm = folder
        existing = [p for p in self._recent if Path(p) != Path(norm)]
        self._recent = [norm] + existing
        self._recent = self._recent[:_RECENT_MAX]
        self._save_recent()
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        m = self._recent_menu
        if m is None:
            return
        m.delete(0, "end")
        if not self._recent:
            m.add_command(label="(empty)", state="disabled")
            return
        for folder in self._recent:
            m.add_command(
                label=folder,
                command=lambda f=folder: self._pick_recent(f),
            )
        m.add_separator()
        m.add_command(label="Clear list", command=self._clear_recent)

    def _pick_recent(self, folder: str) -> None:
        self.var_raw.set(folder)
        self._on_raw_changed()

    def _clear_recent(self) -> None:
        self._recent = []
        self._save_recent()
        self._rebuild_recent_menu()

    def _open_folder(self, folder: Path) -> None:
        if not folder.is_dir():
            messagebox.showerror("Folder missing", f"Not a folder:\n{folder}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as e:
            messagebox.showerror("Open folder", f"Could not open:\n{folder}\n\n{e}")

    def _open_last_output(self) -> None:
        if self.paths.out_dir is None:
            messagebox.showinfo(
                "No output folder",
                "Pick an Output folder on the Inputs tab first.",
            )
            return
        self._open_folder(self.paths.out_dir)

    def _pick_obs(self) -> None:
        if not self.paths.out_dir:
            messagebox.showinfo(
                "Pick output folder first",
                "Please choose an output folder on the Inputs tab first.",
            )
            return
        suggested = str(self.paths.out_dir / "measurements.obs")
        p = filedialog.asksaveasfilename(
            initialfile="measurements.obs",
            defaultextension=".obs",
            filetypes=[("RINEX OBS", "*.obs"), ("All", "*.*")],
            initialdir=str(self.paths.out_dir),
        )
        self.var_obs_path.set(p or suggested)

    # ------------------------------------------------------------------
    # State syncing
    # ------------------------------------------------------------------

    def _on_raw_changed(self) -> None:
        s = self.var_raw.get().strip()
        if not s:
            return
        try:
            raw = RawInputs.from_folder(Path(s))
        except Exception as e:
            self.detected.configure(
                text=f"RAW folder error: {e}", foreground="#ef4444")
            self.paths.raw = None
            self.paths.raw_folder = None
            return
        self.paths.raw = raw
        self.paths.raw_folder = Path(s)
        self.detected.configure(
            text=(
                "Detected:  "
                f"meas = {raw.measurements_txt.name}   "
                f"rec = {raw.recording_txt.name}   "
                f"mp4 = {raw.recording_mp4.name}   "
                f"sens = {raw.sensors_txt.name}"
            ),
            foreground="#22c55e",
        )
        self._push_recent(str(self.paths.raw_folder))
        self.root.after(200, self._update_rotation_preview)
        self._propagate_paths()

    def _on_arnx_changed(self) -> None:
        s = self.var_arnx.get().strip()
        self.paths.android_rinex_src = Path(s) if s else None

    def _on_out_changed(self) -> None:
        s = self.var_out.get().strip()
        if not s:
            return
        out = Path(s)
        out.mkdir(parents=True, exist_ok=True)
        self.paths.out_dir = out
        if not self.var_obs_path.get():
            self.var_obs_path.set(str(out / "measurements.obs"))
        self._propagate_paths()

    def _on_pos_changed(self) -> None:
        s = self.var_pos_path.get().strip()
        if s:
            self.paths.pos_path = Path(s)
        self._propagate_paths()

    def _propagate_paths(self) -> None:
        """Auto-fill downstream tabs from ``self.paths`` — no double upload.

        Mirrors the existing "wire .pos into the CSV tab" pattern (see the
        Post-processing tab docstring): whenever ``pos_path`` / ``raw`` / ``out_dir``
        become known, push them into the Smoothers tab's own StringVars
        (``var_sm_pos`` / ``var_sm_sensors`` / ``var_sm_out``) so the user
        never has to browse for a file that's already loaded elsewhere.
        Only fills fields that are currently *empty* — never stomps a value
        the user deliberately typed or browsed to (GT stays untouched; it's
        optional and user-supplied).

        Safe to call from any callback that touches ``self.paths`` — it is
        idempotent and cheap (a handful of ``StringVar.get``/``.set`` calls).
        """
        pos = self.paths.pos_path
        raw = self.paths.raw
        out_dir = self.paths.out_dir

        # -- Smoothers tab ------------------------------------------------
        if hasattr(self, "var_sm_pos"):
            if pos is not None and not self.var_sm_pos.get().strip():
                self.var_sm_pos.set(str(pos))
            if raw is not None and not self.var_sm_sensors.get().strip():
                sens = getattr(raw, "sensors_txt", None)
                if sens is not None:
                    self.var_sm_sensors.set(str(sens))
            if out_dir is not None and not self.var_sm_out.get().strip():
                self.var_sm_out.set(str(out_dir))
            # var_sm_gt intentionally left alone -- optional/user-supplied.

        # -- Viewers tab (sync_player background layer etc. stay user-driven; the
        # viewer run handlers already read straight from self.paths / raw,
        # so there is nothing to auto-fill there beyond what's already
        # shared state -- see _run_sync_player / _run_traj_viewer / etc.)

    # ------------------------------------------------------------------
    # Background runner
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self._log_q.put(msg)

    @staticmethod
    def _log_tag_for(msg: str) -> str:
        """Severity tag for one log line (kept Tk-free so it is unit-testable)."""
        s = msg.lstrip()
        if s.startswith("==="):
            return "t_done" if "done" in s.lower() else "t_stage"
        if s.startswith("!!!"):
            return "t_error"
        if s.startswith("Traceback") or "Error:" in s or "Exception" in s:
            return "t_error"
        if s.startswith("[") or s.startswith("cmd="):
            return "t_step"
        if "warning" in s.lower() or "warn" in s.lower():
            return "t_warn"
        return "t_normal"

    def _drain_log_queue(self) -> None:
        # Cap the batch so a flooded queue can never freeze the mainloop:
        # anything beyond MAX_PER_TICK waits for the next scheduled tick.
        inserted = False
        try:
            for _ in range(self.MAX_PER_TICK):
                msg = self._log_q.get_nowait()
                self.log_text.insert("end", msg + "\n", self._log_tag_for(msg))
                inserted = True
        except queue.Empty:
            pass
        if inserted:
            try:
                # Trim from the top so the widget (and Tk's text B-tree
                # memory) stays bounded during very long sessions.
                n_lines = int(self.log_text.index("end-1c").split(".")[0])
                if n_lines > self.MAX_LOG_LINES:
                    self.log_text.delete(
                        "1.0", f"{n_lines - self.MAX_LOG_LINES + 1}.0")
            except (tk.TclError, ValueError):
                pass
            # Autoscroll once per batch, not once per line.
            self.log_text.see("end")
        # Reschedule: drain a backlog quickly (short delay) but idle at the
        # normal poll cadence.
        delay = 10 if not self._log_q.empty() else self.POLL_MS
        self.root.after(delay, self._drain_log_queue)

    def _set_busy(self, busy: bool, stage: str = "") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for b in self._buttons:
            try:
                b.configure(state=state)
            except tk.TclError:
                pass
        self._status_lbl.configure(
            text=f"Running: {stage} …" if busy else "Ready",
            foreground=self._warn_amber if busy else self._good_green,
        )
        self._dot_canvas.itemconfigure(
            self._dot_item, fill=self._warn_amber if busy else self._good_green,
        )

    def _set_frame_progress(self, n: int, total: Optional[int]) -> None:
        if self._progress_running:
            try:
                self._progress_bar.stop()
            except tk.TclError:
                pass
            self._progress_running = False
        if total and total > 0:
            self._progress_bar.configure(mode="determinate", value=min(100, n * 100 // total))
        else:
            self._progress_bar.configure(mode="indeterminate")
            self._progress_bar.step(4)

    def _show_progress_bar(self, indeterminate: bool = False) -> None:
        if indeterminate:
            self._progress_bar.configure(mode="indeterminate", value=0)
            self._progress_bar.pack(side="left", padx=(10, 0))
            try:
                self._progress_bar.start(80)
                self._progress_running = True
            except tk.TclError:
                pass
        else:
            self._progress_bar.configure(mode="determinate", value=0)
            self._progress_bar.pack(side="left", padx=(10, 0))

    def _hide_progress_bar(self) -> None:
        if self._progress_running:
            try:
                self._progress_bar.stop()
            except tk.TclError:
                pass
            self._progress_running = False
        self._progress_bar.pack_forget()
        self._progress_bar.configure(mode="determinate", value=0)

    def _run_async(self, fn: Callable[[], None], stage: str) -> None:
        if self._busy:
            messagebox.showinfo(
                "Pipeline busy", "Another stage is still running. Please wait.")
            return
        self._set_busy(True, stage)
        self._log(f"=== {stage} starting ===")

        # Show an indeterminate progress bar unless the caller already showed
        # a determinate one (sample extraction does its own progress tracking).
        if not self._progress_bar.winfo_ismapped():
            self._show_progress_bar(indeterminate=True)

        self._last_ok = True

        def runner() -> None:
            ok = True
            try:
                fn()
                self._log(f"=== {stage} done ===")
            except Exception as e:
                ok = False
                # Persist a structured error report so support can map
                # the failure code straight to the call site instead of
                # guessing from a screenshotted traceback.
                from .errors import (
                    PipelineError, report_user_message, save_error_report,
                )
                report_path = save_error_report(e, stage=stage)
                if isinstance(e, PipelineError):
                    self._log(f"!!! {stage} failed: {e.format()}")
                else:
                    self._log(f"!!! {stage} failed: {type(e).__name__}: {e}")
                self._log(traceback.format_exc())
                self._log(f"[error report] {report_path}")
                # Bubble a one-screen messagebox via the Tk main thread.
                msg = report_user_message(e)
                self.root.after(
                    0, lambda m=msg: messagebox.showerror(f"{stage} failed", m)
                )
            finally:
                self.root.after(0, lambda _ok=ok: self._finish_async(_ok))

        threading.Thread(target=runner, daemon=True).start()

    def _finish_async(self, ok: bool) -> None:
        self._set_busy(False)
        self._hide_progress_bar()
        if not ok:
            self._dot_canvas.itemconfigure(self._dot_item, fill=self._err_red)
            self._status_lbl.configure(foreground=self._err_red, text="Failed")
        # Drive the auto-build queue (if any) one step at a time. Each
        # `_run_async` stage refuses to start while `self._busy` is True, so
        # we cannot fire the next auto-build step until the previous one has
        # fully finished (success or failure) -- this is that hand-off point.
        self._auto_build_step()

    # ------------------------------------------------------------------
    # Auto-build: chain Smoothers + Viewers after Post-processing completes.
    # ------------------------------------------------------------------

    def _auto_build_step(self) -> None:
        """Pop and run the next queued auto-build stage, if any.

        No-op when the queue is empty/absent. Called from `_finish_async`
        so stages run strictly one-after-another (mirrors the single-worker
        discipline `_run_async` already enforces via `self._busy`).
        """
        queue_ = getattr(self, "_auto_build_queue", None)
        if not queue_:
            return
        step_fn, step_name = queue_.pop(0)
        try:
            step_fn()
        except Exception as e:
            # A queued step failed to even *start* (e.g. raised before
            # reaching _run_async). Log it, then keep the chain moving --
            # one broken stage must never block the rest or the GUI.
            self._log(f"!!! [auto-build] {step_name} failed to start: "
                      f"{type(e).__name__}: {e}")
            self._auto_build_step()

    def _auto_build_all(self) -> None:
        """Queue Smoothers + Velocity/Geo viewer builds after Post-processing.

        Opt-out via the "Auto-build" checkbox on the Post-processing tab
        (`self.var_auto_build`, default ON). Reuses the existing manual
        handlers (`_sm_run_selected`, `_run_vel_viewer`, `_run_geo_viewer`)
        so behaviour matches clicking the buttons by hand -- the only
        difference is inputs are already auto-filled by `_propagate_paths`
        and prerequisite checks are silent (log-only) instead of popping a
        messagebox, since there's no user standing in front of a click here.

        Runs on the existing worker-thread machinery (`_run_async`); never
        blocks the Tk mainloop. Each stage is independent -- a failure in
        one is logged and the chain continues with the next.
        """
        if not getattr(self, "var_auto_build", None) or not self.var_auto_build.get():
            return

        steps: list[tuple[Callable[[], None], str]] = []

        def _maybe_smoothers() -> None:
            if not (self.var_sm_pos.get().strip() and self.var_sm_out.get().strip()):
                self._log("[auto-build] skipping Smoothers "
                          "(.pos / output folder not ready)")
                self._auto_build_step()
                return
            names = [n for n, v in self.var_sm_enable.items() if v.get()]
            if not names:
                self._log("[auto-build] skipping Smoothers (none selected)")
                self._auto_build_step()
                return
            self._sm_run_selected()

        def _maybe_vel_viewer() -> None:
            if not (self.paths.out_dir and self.paths.pos_path):
                self._log("[auto-build] skipping Velocity viewer "
                          "(.pos / output folder not ready)")
                self._auto_build_step()
                return
            self._run_vel_viewer()

        def _maybe_geo_viewer() -> None:
            if not (self.paths.out_dir and self.paths.pos_path):
                self._log("[auto-build] skipping Geo viewer "
                          "(.pos / output folder not ready)")
                self._auto_build_step()
                return
            self._run_geo_viewer()

        steps.append((_maybe_smoothers, "Smoothers"))
        steps.append((_maybe_vel_viewer, "Velocity viewer"))
        steps.append((_maybe_geo_viewer, "Geo viewer"))

        self._auto_build_queue = steps
        self._log("[auto-build] queued: " + ", ".join(n for _, n in steps))
        # If nothing is currently running, kick the chain off immediately.
        # Otherwise leave the queue populated -- `_finish_async` (called when
        # the in-flight stage, e.g. Post-processing itself, wraps up) drains it, since
        # `_run_async` refuses to start a new stage while `self._busy`.
        if not self._busy:
            self._auto_build_step()

    # ------------------------------------------------------------------
    # Stage actions
    # ------------------------------------------------------------------

    def _require(self, *fields: tuple[str, object]) -> bool:
        missing = [name for name, val in fields if val in (None, "", Path(""))]
        if missing:
            messagebox.showerror(
                "Missing inputs", "Please fill: " + ", ".join(missing))
            return False
        return True

    def _run_rinex(self) -> None:
        if not self._require(
            ("RAW folder", self.paths.raw),
            ("Output folder", self.paths.out_dir),
            (".obs output path", self.var_obs_path.get()),
        ):
            return
        opts = rinex_stage.RinexOptions(
            skip_edit=self.var_skip_edit.get(),
            fix_bias=self.var_fix_bias.get(),
            marker_name=self.var_marker.get(),
            observer=self.var_observer.get(),
            agency=self.var_agency.get(),
            filter_mode=self.var_filter_mode.get(),
            keep_level=self.var_keep_level.get(),
        )
        meas = self.paths.raw.measurements_txt  # type: ignore[union-attr]
        arnx = self.paths.android_rinex_src
        obs  = Path(self.var_obs_path.get())

        def go() -> None:
            obs_out = rinex_stage.run(
                measurements_txt=meas,
                output_obs=obs,
                android_rinex_src=arnx,
                options=opts,
                log=self._log,
            )
            # Mutate state + sync UI on the main thread so Post-processing tab pickup
            # and any read of paths.obs_path stay race-free with Tk.
            self.root.after(0, lambda p=obs_out: (
                setattr(self.paths, "obs_path", p),
                self._sync_ppk_rover_from_obs(),
            ))

        self._run_async(go, "RINEX conversion")

    def _build_csv_options(self) -> georef_stage.CsvOptions:
        _nan = float("nan")
        _use_prior = self.var_use_pitch_prior.get()
        try:
            _pitch_p = float(self.var_pitch_prior.get()) if _use_prior else _nan
        except ValueError:
            _pitch_p = _nan
        try:
            _roll_p = float(self.var_roll_prior.get()) if _use_prior else _nan
        except ValueError:
            _roll_p = _nan
        _z_override = self.var_z_sigma_override.get()
        _smooth_alt = bool(self.var_smooth_alt.get())
        _alt_sigma = self.var_alt_smooth_sigma.get()
        return georef_stage.CsvOptions(
            smoothing=self.var_smoothing.get(),  # type: ignore[arg-type]
            custom_xy_sigma_s=self.var_xy_sigma.get(),
            custom_z_sigma_s=self.var_z_sigma.get(),
            include_altitude=self.var_include_alt.get(),
            z_sigma_override_s=_z_override if _z_override > 0 else None,
            # Explicit altitude-smoothing opt-in. When the box is unchecked we
            # pass None (legacy behaviour) so existing outputs are unchanged;
            # only an explicit check turns Z smoothing fully on with its sigma.
            smooth_altitude=(True if _smooth_alt else None),
            altitude_smooth_sigma_s=(_alt_sigma if _smooth_alt and _alt_sigma > 0 else None),
            add_ypr=self.var_add_ypr.get(),
            use_gravity_orientation=self.var_gravity_orient.get(),
            use_imu_fusion=self.var_imu_fusion.get(),
            pitch_prior_deg=_pitch_p,
            roll_prior_deg=_roll_p,
            accuracy_x_m=self.var_acc_xy.get(),
            accuracy_y_m=self.var_acc_xy.get(),
            accuracy_z_m=self.var_acc_z.get(),
            max_interp_gap_s=self.var_max_gap.get(),
        )

    def _build_export_options(
        self,
    ) -> "tuple[Optional[list[str]], bool, float, tuple, Optional[float]]":
        """Read the client-export controls (Group 5, Samples+CSV tab).

        Returns ``(coord_systems, smooth_z, z_sigma_s, time_bases,
        audio_start_utc_s)`` ready to pass to
        ``stages.user_export.export_trajectory`` / ``export_kml``.
        ``coord_systems`` is ``None`` when the selection equals the backend
        default (``geodetic`` + ``ecef``) or when nothing is ticked (legacy
        behaviour); otherwise an ordered subset of SUPPORTED_COORD_SYSTEMS.

        ``time_bases`` is an ordered tuple from the time-basis checkboxes
        (``gpst`` / ``utc`` / ``audio`` / ``iso``); ``("gpst",)`` when only the
        default (or nothing) is ticked — the backend default, byte-identical
        output. When ``'audio'`` is ticked, ``audio_start_utc_s`` (UTC of
        stream sample 0) is resolved from the currently loaded RAW session via
        ``audio_frame_export.resolve_session_anchors``; on failure a warning
        is logged and ``'audio'`` is dropped. ``audio_start_utc_s`` is ``None``
        unless ``'audio'`` survives.

        Must run on the Tk main thread (reads widget variables).
        """
        from .stages.user_export import DEFAULT_COORD_SYSTEMS
        selected = [
            name for name, var in (
                ("geodetic", self.var_exp_coord_geodetic),
                ("ecef", self.var_exp_coord_ecef),
                ("utm", self.var_exp_coord_utm),
                ("enu", self.var_exp_coord_enu),
            ) if var.get()
        ]
        coord_systems: "Optional[list[str]]"
        if not selected or tuple(selected) == tuple(DEFAULT_COORD_SYSTEMS):
            coord_systems = None  # backend default (geodetic + ecef)
        else:
            coord_systems = selected
        smooth_z = bool(self.var_exp_smooth_z.get())
        try:
            z_sigma_s = float(self.var_exp_z_sigma_s.get())
        except (tk.TclError, ValueError):
            z_sigma_s = 3.0
        if z_sigma_s <= 0:
            z_sigma_s = 3.0

        # --- Time bases (stable order: reference time, utc, stream, iso) -------------
        tb_selected = [
            name for name, var in (
                ("gpst", self.var_tb_gpst),
                ("utc", self.var_tb_utc),
                ("audio", self.var_tb_audio),
                ("iso", self.var_tb_iso),
            ) if var.get()
        ]
        if not tb_selected:
            tb_selected = ["gpst"]  # backend default (legacy column)

        # 'stream' basis needs the UTC of stream sample 0 — resolve it from
        # the loaded session's anchors (mirrors run_pipeline_from_raw.py).
        audio_start_utc_s: "Optional[float]" = None
        if "audio" in tb_selected:
            try:
                from .audio_frame_export import resolve_session_anchors
                if self.paths.raw_folder is None:
                    raise ValueError(
                        "no RAW session loaded (pick one on the Inputs tab)")
                anchors = resolve_session_anchors(
                    self.paths.raw_folder, inputs=self.paths.raw,
                    need_utc=True, log=self._log,
                )
                if anchors.boot_anchor is None:
                    raise ValueError("no boot->UTC anchor for this session")
                audio_start_utc_s = float(
                    anchors.boot_anchor.boottime_to_utc_s(
                        anchors.audio_start_boot_ns))
                self._log(
                    f"[export] audio time basis: audio_start_utc_s="
                    f"{audio_start_utc_s:.6f} (boot->UTC source: "
                    f"{anchors.boot_anchor_source or 'n/a'})")
            except Exception as _e:
                self._log(
                    f"[export] WARN: 'audio' time basis requested but the "
                    f"session's audio anchor could not be resolved "
                    f"({type(_e).__name__}: {_e}); dropping 'audio'.")
                tb_selected = [b for b in tb_selected if b != "audio"]
                if not tb_selected:
                    tb_selected = ["gpst"]
        time_bases = tuple(tb_selected)

        return coord_systems, smooth_z, z_sigma_s, time_bases, audio_start_utc_s

    def _build_export_source_options(
        self,
    ) -> "tuple[Optional[str], bool, Optional[float]]":
        """Read the export-source + final-velocity controls (Group 5).

        Returns ``(source, emit_final_velocity, vel_disagree_threshold_mps)``:

        * ``source`` — ``None`` for the "(as run)" default (each smoother's
          client export carries its own rows — legacy behaviour); otherwise
          ``"raw"`` or a smoother name, ready for
          ``stages.user_export.resolve_export_rows``.
        * ``emit_final_velocity`` — bool, threaded into
          ``export_trajectory(emit_final_velocity=...)``.
        * ``vel_disagree_threshold_mps`` — parsed float from the entry, or
          ``None`` when the field is blank / unparsable / not > 0 (gate
          disabled — the backend default).

        Defaults are all neutral so the default export stays byte-identical.
        Must run on the Tk main thread (reads widget variables).
        """
        src = (self.var_exp_source.get() or "").strip()
        source: "Optional[str]" = (
            None if not src or src == self.EXPORT_SOURCE_AS_RUN else src)
        emit_fv = bool(self.var_emit_final_vel.get())
        thr_s = (self.var_vel_disagree.get() or "").strip()
        thr: "Optional[float]" = None
        if thr_s:
            try:
                thr = float(thr_s)
            except ValueError:
                self._log(f"[export] WARN: velocity-disagree threshold "
                          f"{thr_s!r} is not a number; gate disabled.")
                thr = None
            else:
                if not thr > 0:  # also rejects NaN
                    self._log(f"[export] WARN: velocity-disagree threshold "
                              f"must be > 0 (got {thr}); gate disabled.")
                    thr = None
        return source, emit_fv, thr

    @staticmethod
    def _chop_passthrough(raw) -> "tuple[Optional[Path], Optional[Path]]":
        """Segment-session args for ``stages.georef.run``.

        When the loaded ``RawInputs`` is a cut "segment" session
        (``raw.is_chop``), coordinate output needs the segment's own media anchor plus the
        cut Container file so sample times land on the segment timeline, not the
        parent's. Returns ``(chop_video_anchor, video_path)``; both ``None``
        for regular sessions (unchanged legacy behaviour).
        """
        if raw is not None and getattr(raw, "is_chop", False):
            return (getattr(raw, "chop_video_anchor", None),
                    getattr(raw, "recording_mp4", None))
        return None, None

    def _run_frames(self) -> None:
        if not self._require(
            ("RAW folder", self.paths.raw),
            ("Output folder", self.paths.out_dir),
        ):
            return
        out      = self.paths.out_dir
        raw      = self.paths.raw
        fps      = float(self.var_fps.get())
        fmt      = self.var_format.get()  # type: ignore[assignment]
        pts_dec  = int(self.var_pts_name_decimals.get())
        rotation = int(self.var_rotation.get() or "0")
        mode     = self.var_fps_mode.get()

        adaptive_pair = self._compute_adaptive_indices_if_requested(mode)
        if mode == "adaptive" and adaptive_pair is None:
            return  # error already surfaced
        adaptive_indices = adaptive_pair[0] if adaptive_pair else None
        adaptive_pts = adaptive_pair[1] if adaptive_pair else None

        def _prog(n: int, total: "Optional[int]") -> None:
            self.root.after(0, lambda n=n, t=total: self._set_frame_progress(n, t))

        def go() -> None:
            res = frames_stage.run(
                video=raw.recording_mp4,   # type: ignore[union-attr]
                out_dir=out,               # type: ignore[arg-type]
                fps=fps,
                fmt=fmt,                   # type: ignore[arg-type]
                pts_name_decimals=pts_dec,
                rotation=rotation,         # type: ignore[arg-type]
                select_indices=adaptive_indices,
                pts_for_indices=adaptive_pts,
                progress_cb=_prog,
                log=self._log,
            )
            self.root.after(0, lambda p=res.frame_times_csv:
                            setattr(self.paths, "frame_times_csv", p))

        self._show_progress_bar()
        label = "Frame extraction (adaptive)" if mode == "adaptive" else "Frame extraction"
        self._run_async(go, label)

    def _on_fps_mode_changed(self) -> None:
        """Placeholder hook — callers consult var_fps_mode at run time."""
        mode = self.var_fps_mode.get()
        self._log(f"[gui] extraction mode → {mode}")

    def _compute_adaptive_indices_if_requested(
        self, mode: str,
    ) -> "Optional[tuple[list[int], list[float]]]":
        """Compute the adaptive keep-list when mode='adaptive'.

        Returns ``None`` for fixed mode, or ``(indices, true_pts_seconds)``
        for adaptive mode. Surfaces a messagebox error and returns ``None``
        if the user has not supplied a .pos file (needed for the Rate-signal
        integration).
        """
        if mode != "adaptive":
            return None
        if self.paths.raw is None or self.paths.pos_path is None:
            messagebox.showerror(
                "Adaptive mode requirements",
                "Adaptive mode needs both:\n"
                "  • RAW folder (Inputs tab)\n"
                "  • .pos file  (PPK or RINEX tab)",
            )
            return None
        try:
            opts = adaptive_stage.AdaptiveOptions(
                spacing_m=float(self.var_adapt_spacing_m.get()),
                turn_overlap=float(self.var_adapt_turn_overlap.get()),
                yaw_rate_threshold_dps=float(self.var_adapt_yawrate.get()),
                min_interval_s=float(self.var_adapt_min_dt.get()),
                max_interval_s=float(self.var_adapt_max_dt.get()),
            )
        except ValueError as e:
            messagebox.showerror("Adaptive options", f"Bad option: {e}")
            return None
        self._log(
            f"[adaptive] running selector: spacing={opts.spacing_m:.2f}m, "
            f"turn_overlap={opts.turn_overlap:.2f}, "
            f"yaw_rate={opts.yaw_rate_threshold_dps:.1f}°/s"
        )
        raw = self.paths.raw
        try:
            res = adaptive_stage.compute_keep_list(
                video=raw.recording_mp4,
                pos_file=self.paths.pos_path,
                recording_map=raw.recording_txt,
                options=opts,
                capture_meta=raw.capture_meta_json,
                video_anchor=raw.video_anchor_txt,
                # Segment clip: its own anchor's min bootNs overrides the parent
                # capture_meta t0 (segment PTS are rebased to 0).
                chop_video_anchor=(raw.chop_video_anchor
                                   if getattr(raw, "is_chop", False) else None),
                log=self._log,
            )
        except Exception as e:
            messagebox.showerror("Adaptive selector failed", str(e))
            return None
        self._log(
            f"[adaptive] kept {res.n_kept}/{res.n_total} source frames "
            f"({100.0*res.n_kept/max(res.n_total,1):.1f} %)"
        )
        if not res.keep_indices:
            messagebox.showwarning(
                "No frames selected",
                "Adaptive selector returned an empty keep-list. Loosen the "
                "spacing or turn-overlap thresholds.",
            )
            return None
        return res.keep_indices, res.keep_pts_s

    # Smoother names that need pre-smoothing of .pos before georef.run().
    _PRESMOOTH_NAMES = frozenset({
        "epoch_weighted", "epoch_weighted_v2",
        "cv_rts", "cv_rts_pv", "gate_then_cv",
        "ekf_smoothed", "kalman_simple_cv",
    })

    def _resolve_smoother_opts(
        self,
        opts: "georef_stage.CsvOptions",
        sm_choice: str,
    ) -> "tuple[georef_stage.CsvOptions, Optional[Path]]":
        """Translate a smoother dropdown choice into CsvOptions + pre-smooth path.

        Returns (opts, pre_smooth_pos). pre_smooth_pos is None when the
        smoother is handled inside georef.run() (Gaussian profiles,
        ns_adaptive flag, fgo flag, fused-bent). Non-None when the .pos
        must be pre-smoothed before georef.run().
        """
        pre_smooth_pos: "Optional[Path]" = None

        if sm_choice in ("ns_adaptive", "fgo"):
            opts = georef_stage.CsvOptions(
                smoothing="car",
                custom_xy_sigma_s=opts.custom_xy_sigma_s,
                custom_z_sigma_s=opts.custom_z_sigma_s,
                include_altitude=opts.include_altitude,
                z_sigma_override_s=opts.z_sigma_override_s,
                add_ypr=opts.add_ypr,
                use_gravity_orientation=opts.use_gravity_orientation,
                use_imu_fusion=opts.use_imu_fusion,
                pitch_prior_deg=opts.pitch_prior_deg,
                roll_prior_deg=opts.roll_prior_deg,
                accuracy_x_m=opts.accuracy_x_m,
                accuracy_y_m=opts.accuracy_y_m,
                accuracy_z_m=opts.accuracy_z_m,
                max_interp_gap_s=opts.max_interp_gap_s,
                use_ns_adaptive_smoothing=(sm_choice == "ns_adaptive"),
                use_fgo_smoothing=(sm_choice == "fgo"),
            )
        elif sm_choice in self._PRESMOOTH_NAMES:
            pre_smooth_pos = self.paths.out_dir / f"rover.{sm_choice}.pos"  # type: ignore[union-attr]
            opts = georef_stage.CsvOptions(
                smoothing="none",
                include_altitude=opts.include_altitude,
                add_ypr=opts.add_ypr,
                use_gravity_orientation=opts.use_gravity_orientation,
                use_imu_fusion=opts.use_imu_fusion,
                pitch_prior_deg=opts.pitch_prior_deg,
                roll_prior_deg=opts.roll_prior_deg,
                accuracy_x_m=opts.accuracy_x_m,
                accuracy_y_m=opts.accuracy_y_m,
                accuracy_z_m=opts.accuracy_z_m,
                max_interp_gap_s=opts.max_interp_gap_s,
            )

        return opts, pre_smooth_pos

    def _run_csv(self) -> None:
        if not self._require(
            ("RAW folder", self.paths.raw),
            ("Output folder", self.paths.out_dir),
            (".pos file", self.paths.pos_path),
        ):
            return
        ftc = self.paths.frame_times_csv or (
            self.paths.out_dir / "extracted_frame_times.csv"  # type: ignore[union-attr]
        )
        if not ftc.is_file():
            messagebox.showerror(
                "Missing frame times",
                f"Run 'Extract frames' first (expected {ftc}).",
            )
            return
        opts    = self._build_csv_options()
        out_csv = self.paths.out_dir / "georef.csv"  # type: ignore[union-attr]
        raw     = self.paths.raw
        pos     = self.paths.pos_path
        fps     = float(self.var_fps.get())
        sm_choice = self.var_smoothing.get()

        opts, pre_smooth_pos = self._resolve_smoother_opts(opts, sm_choice)
        chop_anchor, chop_video = self._chop_passthrough(raw)

        def go() -> None:
            pos_to_use = pos
            if pre_smooth_pos is not None:
                self._log(f"[csv] pre-smoothing .pos with {sm_choice} ...")
                pos_to_use = self._presmooth_pos(pos, pre_smooth_pos, sm_choice, raw)
            res = georef_stage.run(
                frame_times_csv=ftc,
                recording_map=raw.recording_txt,      # type: ignore[union-attr]
                pos_file=pos_to_use,
                data_log=raw.measurements_txt,        # type: ignore[union-attr]
                sensors_txt=raw.sensors_txt,           # type: ignore[union-attr]
                out_csv=out_csv,
                fps=fps,
                options=opts,
                capture_meta=raw.capture_meta_json,    # type: ignore[union-attr]
                video_anchor=raw.video_anchor_txt,     # type: ignore[union-attr]
                chop_video_anchor=chop_anchor,
                video_path=chop_video,
                log=self._log,
            )
            self.root.after(0, lambda p=res.csv_path:
                            setattr(self.paths, "georef_csv", p))

        self._run_async(go, f"Georef CSV build ({sm_choice})")

    def _presmooth_pos(self, pos_in: Path, pos_out: Path,
                        smoother: str, raw) -> Path:
        """Run any registered smoother over pos_in and write the smoothed
        result as a minimal The external solver-format .pos at pos_out.

        Supports epoch_weighted, epoch_weighted_v2 natively (Local-frame arrays),
        and all other smoothers via ``smoothers.run_smoother()`` (PosRow list).
        Returns pos_out.
        """
        from .parsers import parse_rtkpos, PosRow, parse_imu
        from .geo import _A, _E2, llh_to_ecef
        import math as _math

        rows = parse_rtkpos(pos_in)
        if not rows:
            raise RuntimeError(
                f"presmooth: {pos_in} has no PPK rows; re-run PPK first."
            )

        smoothed_rows: list[PosRow]

        if smoother in ("epoch_weighted", "epoch_weighted_v2"):
            stat_p = pos_in.with_suffix(pos_in.suffix + ".stat")
            stat_p = stat_p if stat_p.is_file() else None

            if smoother == "epoch_weighted_v2":
                from .epoch_weight_v2 import (
                    EpochWeightV2Options, smooth_epoch_weighted_v2,
                )
                imu_rows = None
                try:
                    if raw.sensors_txt is not None:
                        imu_rows = parse_imu(raw.sensors_txt)
                except (FileNotFoundError, OSError, ValueError) as e:
                    self._log(f"[csv] v2 IMU parse failed ({e}); no IMU.")
                ew_opts = EpochWeightV2Options(
                    stat_path=stat_p, zupt_enabled=True,
                    nhc_enabled=True, nhc_heading_source="doppler",
                    sigma_a_base=0.10,
                )
                v2 = smooth_epoch_weighted_v2(rows, imu_rows=imu_rows,
                                              options=ew_opts, log=self._log)
                Es, Ns, Us = v2.E_smooth, v2.N_smooth, v2.U_smooth

                gate_strategy = self.var_confidence_gate.get()
                if gate_strategy != "off":
                    from .epoch_confidence import EpochGateConfig, compute_epoch_gate
                    gate_cfg = EpochGateConfig(strategy=gate_strategy)
                    gate = compute_epoch_gate(rows, v2, config=gate_cfg)
                    self._log(
                        f"[csv] confidence gate '{gate_strategy}': "
                        f"session_passed={gate.session_passed} "
                        f"(smart_std={gate.session_smart_std:.2f}m), "
                        f"kept {gate.n_kept}/{gate.n_total} "
                        f"({100*gate.n_kept/max(gate.n_total,1):.1f}%), "
                        f"predicted_max={gate.predicted_max_m:.2f}m"
                    )
                    if not gate.session_passed:
                        self._log(
                            f"[csv] WARNING: session rejected by confidence gate "
                            f"(smart_std={gate.session_smart_std:.2f}m > "
                            f"{gate_cfg.session_smart_std_max}m). "
                            f"All epochs will be written but may exceed accuracy target."
                        )
                    elif gate.n_rejected > 0:
                        reject_mask = ~gate.keep_mask
                        Es = np.where(reject_mask, np.nan, Es)
                        Ns = np.where(reject_mask, np.nan, Ns)
                        Us = np.where(reject_mask, np.nan, Us)
            else:
                from .epoch_weight import smooth_epoch_weighted
                Es, Ns, Us = smooth_epoch_weighted(rows, stat_path=stat_p)

            ref = (rows[0].lat_deg, rows[0].lon_deg, rows[0].h_m)
            rx, ry, rz = llh_to_ecef(*ref)
            rlat = _math.radians(ref[0]); rlon = _math.radians(ref[1])
            sl, cl = _math.sin(rlat), _math.cos(rlat)
            so, co = _math.sin(rlon), _math.cos(rlon)
            smoothed_rows = []
            for i, r in enumerate(rows):
                e, n, u = float(Es[i]), float(Ns[i]), float(Us[i])
                if not (_math.isfinite(e) and _math.isfinite(n) and _math.isfinite(u)):
                    continue
                x = rx + (-so * e - sl * co * n + cl * co * u)
                y = ry + (co * e - sl * so * n + cl * so * u)
                z = rz + (cl * n + sl * u)
                p_xy = _math.hypot(x, y); lon_r = _math.atan2(y, x)
                lat_r = _math.atan2(z, p_xy * (1 - _E2))
                for _ in range(6):
                    sinl = _math.sin(lat_r)
                    Nrad = _A / _math.sqrt(1 - _E2 * sinl * sinl)
                    h_iter = p_xy / max(1e-12, _math.cos(lat_r)) - Nrad
                    lat_r = _math.atan2(z, p_xy * (1 - _E2 * Nrad / (Nrad + h_iter)))
                sinl = _math.sin(lat_r)
                Nrad = _A / _math.sqrt(1 - _E2 * sinl * sinl)
                h_m = p_xy / max(1e-12, _math.cos(lat_r)) - Nrad
                smoothed_rows.append(PosRow(
                    utc_s=r.utc_s,
                    lat_deg=_math.degrees(lat_r),
                    lon_deg=_math.degrees(lon_r),
                    h_m=h_m,
                    quality=r.quality, ns=r.ns,
                    vn=r.vn, ve=r.ve, vu=r.vu,
                    sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
                    sd_vn=r.sd_vn, sd_ve=r.sd_ve, sd_vu=r.sd_vu,
                    ratio=r.ratio, age_s=r.age_s,
                ))
        else:
            from .smoothers import run_smoother
            imu_rows = None
            try:
                if raw is not None and getattr(raw, "sensors_txt", None) is not None:
                    imu_rows = parse_imu(raw.sensors_txt)
                    self._log(f"[csv] loaded {len(imu_rows)} IMU rows for {smoother}")
            except Exception as e:
                self._log(f"[csv] IMU parse for {smoother}: {e}; continuing without IMU")
            res = run_smoother(smoother, rows, imu_rows=imu_rows, log=self._log)
            if not res.ok:
                raise RuntimeError(
                    f"Smoother '{smoother}' failed: {res.error_message} "
                    f"(code {res.error_code}). {res.error_hint or ''}"
                )
            smoothed_rows = res.fused
            self._log(f"[csv] {smoother}: {res.n_input} -> {res.n_output} rows "
                      f"in {res.runtime_s:.2f}s")

        if not smoothed_rows:
            raise RuntimeError(
                f"presmooth: {smoother} produced 0 rows from {len(rows)} input rows."
            )

        self._write_pos_file(smoothed_rows, pos_out, smoother)
        return pos_out

    def _write_pos_file(self, smoothed_rows: list, pos_out: Path,
                        smoother: str) -> None:
        """Write PosRow list as The external solver-format .pos (atomic via .tmp)."""
        import math as _math
        import os as _os
        import datetime as _dt
        from .time_sync import get_leap_seconds_for_epoch

        tmp = Path(str(pos_out) + ".tmp")
        pos_out.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as f:
            f.write(f"% program   : data_pipeline {smoother}\n")
            f.write("% (lat/lon/height=WGS84/ellipsoidal,Q=1:fix,2:float)\n")
            f.write("%  GPST                  latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio    vn(m/s)    ve(m/s)    vu(m/s)\n")
            for r in smoothed_rows:
                ls = get_leap_seconds_for_epoch(r.utc_s)
                t = _dt.datetime.fromtimestamp(r.utc_s + ls, tz=_dt.timezone.utc)
                date_s = t.strftime("%Y/%m/%d")
                time_s = t.strftime("%H:%M:%S.") + f"{t.microsecond // 1000:03d}"
                def _f(v, w=7, d=4):
                    return f"{v:>{w}.{d}f}" if (v is not None and _math.isfinite(v)) else " " * w
                f.write(
                    f"{date_s} {time_s}  {r.lat_deg:.9f}   {r.lon_deg:.9f}  {r.h_m:>10.4f}  "
                    f"{r.quality:>2d} {r.ns:>3d}  "
                    f"{_f(r.sd_n)} {_f(r.sd_e)} {_f(r.sd_u)} "
                    f"{'  0.0000':>8} {'  0.0000':>8} {'  0.0000':>8} "
                    f"{'   0.00':>7} {'   0.0':>7} "
                    f"{_f(r.vn, 10, 5)} {_f(r.ve, 10, 5)} {_f(r.vu, 10, 5)}\n"
                )
        _os.replace(tmp, pos_out)
        self._log(f"[csv] wrote smoothed .pos ({len(smoothed_rows)} rows) -> {pos_out}")

    def _run_frames_and_csv(self) -> None:
        if not self._require(
            ("RAW folder", self.paths.raw),
            ("Output folder", self.paths.out_dir),
            (".pos file", self.paths.pos_path),
        ):
            return
        opts     = self._build_csv_options()
        out      = self.paths.out_dir
        raw      = self.paths.raw
        pos      = self.paths.pos_path
        fps      = float(self.var_fps.get())
        fmt      = self.var_format.get()
        pts_dec  = int(self.var_pts_name_decimals.get())
        rotation = int(self.var_rotation.get() or "0")
        mode     = self.var_fps_mode.get()
        sm_choice = self.var_smoothing.get()
        adaptive_pair = self._compute_adaptive_indices_if_requested(mode)
        if mode == "adaptive" and adaptive_pair is None:
            return
        adaptive_indices = adaptive_pair[0] if adaptive_pair else None
        adaptive_pts = adaptive_pair[1] if adaptive_pair else None

        opts, pre_smooth_pos = self._resolve_smoother_opts(opts, sm_choice)
        chop_anchor, chop_video = self._chop_passthrough(raw)

        def _prog2(n: int, total: "Optional[int]") -> None:
            self.root.after(0, lambda n=n, t=total: self._set_frame_progress(n, t))

        def go() -> None:
            res = frames_stage.run(
                video=raw.recording_mp4,            # type: ignore[union-attr]
                out_dir=out,                        # type: ignore[arg-type]
                fps=fps,
                fmt=fmt,                            # type: ignore[arg-type]
                pts_name_decimals=pts_dec,
                rotation=rotation,                  # type: ignore[arg-type]
                select_indices=adaptive_indices,
                pts_for_indices=adaptive_pts,
                progress_cb=_prog2,
                log=self._log,
            )
            # Samples done: marshal state + UI flip onto main thread.
            self.root.after(0, lambda p=res.frame_times_csv: (
                setattr(self.paths, "frame_times_csv", p),
                self._hide_progress_bar(),
                self._show_progress_bar(indeterminate=True),
            ))
            pos_to_use = pos
            if pre_smooth_pos is not None:
                self._log(f"[csv] pre-smoothing .pos with {sm_choice} ...")
                pos_to_use = self._presmooth_pos(pos, pre_smooth_pos, sm_choice, raw)
            csv_res = georef_stage.run(
                frame_times_csv=res.frame_times_csv,
                recording_map=raw.recording_txt,    # type: ignore[union-attr]
                pos_file=pos_to_use,
                data_log=raw.measurements_txt,     # type: ignore[union-attr]
                sensors_txt=raw.sensors_txt,        # type: ignore[union-attr]
                out_csv=out / "georef.csv",  # type: ignore[operator]
                fps=fps,
                options=opts,
                capture_meta=raw.capture_meta_json,  # type: ignore[union-attr]
                chop_video_anchor=chop_anchor,
                video_path=chop_video,
                log=self._log,
            )
            self.root.after(0, lambda p=csv_res.csv_path:
                            setattr(self.paths, "georef_csv", p))

        self._show_progress_bar()
        self._run_async(go, f"Frames + Georef CSV ({sm_choice})")

    def _show_vel_plot(self) -> None:
        if not self._require((".pos file", self.paths.pos_path)):
            return
        import math as _m
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            messagebox.showerror("matplotlib missing",
                                 "Install matplotlib to use this feature.")
            return
        from .parsers import parse_rtkpos
        from .geo import llh_to_ecef, ecef_to_enu

        pos = parse_rtkpos(self.paths.pos_path)  # type: ignore[arg-type]
        if not pos:
            messagebox.showerror("Empty .pos", "No rows parsed.")
            return

        t0 = pos[0].utc_s
        dop_t, dop_v = [], []
        for r in pos:
            if _m.isfinite(r.vn) and _m.isfinite(r.ve):
                dop_t.append(r.utc_s - t0)
                dop_v.append(_m.sqrt(r.vn ** 2 + r.ve ** 2))

        ref_llh = (pos[0].lat_deg, pos[0].lon_deg, pos[0].h_m)
        crd_t, crd_v = [], []
        prev_e: Optional[float] = None
        prev_n: Optional[float] = None
        prev_t: Optional[float] = None
        for r in pos:
            x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
            e, n, _ = ecef_to_enu(x, y, z, ref_llh)
            if prev_e is not None and prev_t is not None:
                dt = r.utc_s - prev_t
                if 0 < dt <= 2.0:
                    de, dn = e - prev_e, n - (prev_n or 0.0)
                    crd_t.append(r.utc_s - t0)
                    crd_v.append(_m.sqrt(de * de + dn * dn) / dt)
            prev_e, prev_n, prev_t = e, n, r.utc_s

        BG = "#0c0f14"
        fig = Figure(figsize=(13, 5), facecolor=BG)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.plot(dop_t, dop_v, color="#38b6ff", lw=1.0, alpha=0.85,
                label="Doppler speed  — sqrt(ve² + vn²)  [PPK carrier-phase rate]")
        ax.plot(crd_t, crd_v, color="#ff5c5c", lw=0.8, alpha=0.7,
                label="Coords speed  — ENU position diff / dt  [1 Hz]")
        ax.set_xlabel("Session time  (s)", color="#cccccc")
        ax.set_ylabel("Horizontal speed  (m/s)", color="#cccccc")
        ax.set_title("Doppler speed vs coordinate-derived speed\n"
                     "Disagreement = |Doppler − Coords| — large values predict coordinate outliers",
                     color="#ffffff", fontsize=10)
        ax.tick_params(colors="#555555")
        ax.legend(fontsize=8.5, framealpha=0.3, facecolor="#0d1220",
                  edgecolor="#2a3a55", labelcolor="#dddddd", loc="upper left")
        ax.grid(color="#1a1a1a", lw=0.7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#222222")
        fig.tight_layout()

        win = tk.Toplevel(self.root)
        win.title("Velocity: Doppler vs Coords")
        win.geometry("1020x430")
        win.configure(bg=BG)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        win.protocol("WM_DELETE_WINDOW", lambda: (fig.clf(), win.destroy()))

    def _ensure_preview_win(self) -> None:
        if self._preview_win is not None and self._preview_win.winfo_exists():
            self._preview_win.deiconify()
            self._preview_win.lift()
            return
        win = tk.Toplevel(self.root)
        win.title("Rotation preview  (auto-updates on selection change)")
        win.configure(bg="#0a0a1a")
        win.resizable(True, True)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        self._preview_canvas = tk.Canvas(
            win, bg="#0a0a1a", highlightthickness=0, width=480, height=270,
        )
        self._preview_canvas.pack(padx=8, pady=(8, 2))
        self._preview_canvas.create_text(
            240, 135, text="Extracting frame…", fill="#888",
            font=("Segoe UI", 11), tags="msg",
        )
        self._preview_rot_lbl = ttk.Label(win, foreground="#888")
        self._preview_rot_lbl.pack(pady=(0, 8))
        self._preview_win = win

    def _update_rotation_preview(self) -> None:
        if not (self.paths.raw and self.paths.raw.recording_mp4.is_file()):
            return
        rotation = int(self.var_rotation.get() or "0")
        self._refresh_preview(self.paths.raw.recording_mp4, rotation)

    def _refresh_preview(self, video: Path, rotation: int) -> None:
        """Render the rotation-preview window for an arbitrary media + rotation."""
        import threading as _thr
        import tempfile as _tmp
        import subprocess as _sp
        import os as _os
        from .ffmpeg_paths import resolve_ffmpeg as _ffmpeg

        self._preview_gen += 1
        gen      = self._preview_gen
        rot_vf   = {0: None, 90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}
        vf       = rot_vf.get(rotation)

        self._ensure_preview_win()
        if self._preview_canvas and self._preview_canvas.winfo_exists():
            self._preview_canvas.delete("all")
            cw = self._preview_canvas.winfo_width() or 480
            ch = self._preview_canvas.winfo_height() or 270
            self._preview_canvas.create_text(
                cw // 2, ch // 2, text="Extracting…",
                fill="#888", font=("Segoe UI", 11),
            )

        def worker() -> None:
            tf = _tmp.NamedTemporaryFile(suffix=".png", delete=False)
            tf.close()
            path = tf.name
            cmd  = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", "5", "-i", str(video), "-vframes", "1"]
            if vf:
                cmd += ["-vf", vf]
            cmd.append(path)
            try:
                _sp.run(cmd, capture_output=True, check=True)
                self.root.after(0, lambda: self._show_preview_result(gen, path, rotation))
            except Exception:
                try: _os.unlink(path)
                except OSError: pass

        _thr.Thread(target=worker, daemon=True).start()

    def _show_preview_result(self, gen: int, path: str, rotation: int) -> None:
        import os as _os
        rot_lbl = {0: "0° — no rotation", 90: "90° clockwise",
                   180: "180° flip", 270: "270° counter-clockwise"}
        def cleanup() -> None:
            try: _os.unlink(path)
            except OSError: pass

        if gen != self._preview_gen:
            cleanup(); return
        if self._preview_win is None or not self._preview_win.winfo_exists():
            cleanup(); return
        if self._preview_canvas is None:
            cleanup(); return

        try:
            img = tk.PhotoImage(file=path)
        except Exception:
            cleanup(); return

        w, h = img.width(), img.height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        max_w = max(200, int(sw * 0.85))
        max_h = max(200, int(sh * 0.80))
        sub = max(1, max((w + max_w - 1) // max_w, (h + max_h - 1) // max_h))
        if sub > 1:
            img = img.subsample(sub, sub)
        dw, dh = img.width(), img.height()

        canvas = self._preview_canvas
        canvas.configure(width=dw, height=dh)
        canvas.delete("all")
        canvas.create_image(dw // 2, dh // 2, image=img, anchor="center")
        canvas.image = img  # type: ignore[attr-defined]

        self._preview_rot_lbl.configure(                   # type: ignore[union-attr]
            text=f"{rot_lbl.get(rotation, str(rotation) + '°')}  —  t ≈ 5 s")

        win = self._preview_win
        win.update_idletasks()
        win.geometry(f"{dw + 16}x{dh + 50}")

        if self._preview_last_path and self._preview_last_path != path:
            try: _os.unlink(self._preview_last_path)
            except OSError: pass
        self._preview_last_path = path

    def _run_vel_viewer(self) -> None:
        if not self._require(
            ("Output folder", self.paths.out_dir),
            (".pos file",     self.paths.pos_path),
        ):
            return
        out = self.paths.out_dir / "velocity_viewer.html"  # type: ignore[union-attr]
        pos = self.paths.pos_path
        data_log = self.paths.raw.measurements_txt if self.paths.raw is not None else None

        def go() -> None:
            viewers_stage.build_velocity_viewer(
                pos_file=pos,   # type: ignore[arg-type]
                out_html=out,
                data_log=data_log,
                log=self._log,
            )

        self._run_async(go, "Velocity viewer")

    def _run_speed_vs_gt_viewer(self) -> None:
        """Build the speed + azimuth device-vs-GT comparison HTML report.

        Reuses ``scripts/speed_vs_gt_html.py`` so the GUI and the CLI
        produce identical output. The device .pos comes from the
        pipeline state (set by Interchange-format or Post-processing tabs); the GT .pos is
        picked via a file dialog.
        """
        if not self._require(
            ("Output folder", self.paths.out_dir),
            ("device .pos file", self.paths.pos_path),
        ):
            return
        gt_path_str = filedialog.askopenfilename(
            title="Pick the reference .pos (e.g. surveyed reference rover)",
            filetypes=[("RTKLIB pos", "*.pos"), ("All", "*.*")],
        )
        if not gt_path_str:
            return
        data_pos = self.paths.pos_path
        gt_pos = Path(gt_path_str)
        out = self.paths.out_dir / "speed_azimuth_vs_gt.html"  # type: ignore[union-attr]

        # The speed/azimuth-vs-GT report builder is provided by an optional
        # ``speed_vs_gt_html`` module that lives under ``scripts/``. It is not
        # shipped with every build, so resolve it up front and fail with a
        # clear message instead of letting a bare ``import`` raise
        # ModuleNotFoundError mid-run (the old behaviour, which surfaced as an
        # opaque stack trace in the GUI).
        import sys as _sys
        import importlib.util as _ilu
        repo_root = Path(__file__).resolve().parent.parent
        scripts_dir = repo_root / "scripts"
        if str(scripts_dir) not in _sys.path:
            _sys.path.insert(0, str(scripts_dir))

        if _ilu.find_spec("speed_vs_gt_html") is None:
            msg = (
                "The Speed-vs-GT report builder is not available in this "
                "build.\n\nExpected module 'speed_vs_gt_html' (looked in "
                f"{scripts_dir}).\n\nThis optional viewer is not bundled; "
                "no report was generated."
            )
            self._log(f"[speed-vs-gt] unavailable: {msg}")
            messagebox.showinfo("Speed-vs-GT viewer unavailable", msg)
            return

        def go() -> None:
            # Re-implement the script's main inline so we can pipe self._log
            # and don't depend on argv. Same numbers, same chart code.
            import speed_vs_gt_html as svg_mod  # type: ignore[import-not-found]
            old_argv = _sys.argv
            _sys.argv = [
                "speed_vs_gt_html",
                "--phone-pos", str(data_pos),
                "--gt-pos",    str(gt_pos),
                "--out",       str(out),
            ]
            try:
                svg_mod.main()
            finally:
                _sys.argv = old_argv
            self._log(f"[speed-vs-gt] wrote {out}")

        self._run_async(go, "Speed-vs-GT viewer")

    def _run_geo_viewer(self) -> None:
        if not self._require(
            ("Output folder", self.paths.out_dir),
            (".pos file",     self.paths.pos_path),
        ):
            return
        out = self.paths.out_dir / "geo_viewer.html"  # type: ignore[union-attr]
        pos = self.paths.pos_path

        bm_raw = self.var_geo_basemap.get().strip()
        dsm_raw = self.var_geo_dsm.get().strip()
        basemap = Path(bm_raw)  if bm_raw  else None
        dsm     = Path(dsm_raw) if dsm_raw else None

        if basemap is not None and not basemap.is_file():
            messagebox.showerror("Basemap", f"GeoTIFF not found:\n{basemap}")
            return
        if dsm is not None and not dsm.is_file():
            messagebox.showerror("DSM", f"GeoTIFF not found:\n{dsm}")
            return

        # Use Coordinate output CSV if already built
        csv_candidate = self.paths.georef_csv or (
            (self.paths.out_dir / "georef.csv")  # type: ignore[union-attr]
            if self.paths.out_dir else None
        )
        mcsv = csv_candidate if (csv_candidate is not None and csv_candidate.is_file()) else None

        def go() -> None:
            viewers_stage.build_geo_viewer(
                pos_file=pos,         # type: ignore[arg-type]
                out_html=out,
                georef_csv=mcsv,
                basemap_tiff=basemap,
                dsm_tiff=dsm,
                log=self._log,
            )

        self._run_async(go, "Geo viewer")

    def _run_traj_viewer(self) -> None:
        if not self._require(
            ("RAW folder", self.paths.raw),
            ("Output folder", self.paths.out_dir),
        ):
            return
        csv_path = self.paths.georef_csv or (
            self.paths.out_dir / "georef.csv"  # type: ignore[union-attr]
        )
        if not csv_path.is_file():
            messagebox.showerror(
                "Missing CSV", f"Build the Georef CSV first ({csv_path}).")
            return
        out      = self.paths.out_dir / "trajectory_viewer.html"  # type: ignore[union-attr]
        data_log = self.paths.raw.measurements_txt               # type: ignore[union-attr]

        def go() -> None:
            viewers_stage.build_trajectory_viewer(
                data_log=data_log,
                georef_csv=csv_path,
                out_html=out,
                recording_map=self.paths.raw.recording_txt,  # type: ignore[union-attr]
                pos_file=self.paths.pos_path,
                log=self._log,
            )

        self._run_async(go, "Trajectory viewer")

    def _run_orient_panel(self) -> None:
        if not self._require(
            ("RAW folder", self.paths.raw),
            ("Output folder", self.paths.out_dir),
            (".pos file", self.paths.pos_path),
        ):
            return
        out       = self.paths.out_dir / "orientation_panel.html"  # type: ignore[union-attr]
        data_log = self.paths.raw.measurements_txt                # type: ignore[union-attr]
        pos       = self.paths.pos_path

        sensors = self.paths.raw.sensors_txt if self.paths.raw else None  # type: ignore[union-attr]

        def go() -> None:
            viewers_stage.build_orientation_panel(
                data_log=data_log,
                pos_file=pos,                  # type: ignore[arg-type]
                out_html=out,
                sensors_txt=sensors,
                log=self._log,
            )

        self._run_async(go, "Orientation panel")

    def _run_compare_viewer(self) -> None:
        if not self._require(
            ("RAW folder", self.paths.raw),
            ("Output folder", self.paths.out_dir),
            (".pos file", self.paths.pos_path),
            ("frame times CSV", self.paths.frame_times_csv),
        ):
            return
        out      = self.paths.out_dir / "comparison_viewer.html"  # type: ignore[union-attr]
        raw      = self.paths.raw                                  # type: ignore[assignment]
        pos      = self.paths.pos_path
        ftcsv    = self.paths.frame_times_csv
        rec      = raw.recording_txt                               # type: ignore[union-attr]
        fps      = float(self.var_fps.get())
        data_log = raw.measurements_txt                           # type: ignore[union-attr]

        def go() -> None:
            viewers_stage.build_comparison_viewer(
                data_log=data_log,
                pos_file=pos,                  # type: ignore[arg-type]
                frame_times_csv=ftcsv,         # type: ignore[arg-type]
                recording_map=rec,
                out_html=out,
                fps=fps,
                capture_meta=raw.capture_meta_json,    # type: ignore[union-attr]
                video_anchor=raw.video_anchor_txt,     # type: ignore[union-attr]
                # Segment clip: its own anchor's min bootNs overrides the parent
                # capture_meta t0 (segment PTS are rebased to 0).
                chop_video_anchor=(raw.chop_video_anchor  # type: ignore[union-attr]
                                   if getattr(raw, "is_chop", False) else None),
                log=self._log,
            )

        self._run_async(go, "Comparison viewer")

    def _run_sync_player(self) -> None:
        if not self._require(
            ("RAW folder", self.paths.raw),
            ("Output folder", self.paths.out_dir),
            (".pos file", self.paths.pos_path),
            ("frame times CSV", self.paths.frame_times_csv),
        ):
            return
        out      = self.paths.out_dir / "sync_player.html"  # type: ignore[union-attr]
        raw      = self.paths.raw                            # type: ignore[assignment]
        pos      = self.paths.pos_path
        ftcsv    = self.paths.frame_times_csv
        rec      = raw.recording_txt                         # type: ignore[union-attr]
        video    = raw.recording_mp4                         # type: ignore[union-attr]
        rotation = int(self.var_rotation.get() or "0")

        basemap_raw = self.var_sync_basemap.get().strip()
        basemap_geotiff = Path(basemap_raw) if basemap_raw else None
        if basemap_geotiff is not None and not basemap_geotiff.is_file():
            messagebox.showerror(
                "Basemap", f"GeoTIFF basemap not found:\n{basemap_geotiff}")
            return

        try:
            bias_ms = float(self.var_video_bias_ms.get().strip() or "0")
        except ValueError:
            bias_ms = 0.0

        stat = pos.with_suffix(pos.suffix + ".stat") if pos else None  # type: ignore[union-attr]
        if stat is not None and not stat.is_file():
            stat = None

        def go() -> None:
            viewers_stage.build_sync_player(
                video=video,
                pos_file=pos,                  # type: ignore[arg-type]
                frame_times_csv=ftcsv,         # type: ignore[arg-type]
                recording_map=rec,
                out_html=out,
                rotation=rotation,
                sensors_txt=raw.sensors_txt,   # type: ignore[union-attr]
                data_log=raw.measurements_txt,  # type: ignore[union-attr]
                basemap_geotiff=basemap_geotiff,
                stat_file=stat,
                video_bias_ms=bias_ms,
                # Stream + feature map (when the session ships a wav). Old-format
                # sessions have none -> these stay None and the player omits stream.
                wav=getattr(raw, "audio_wav", None),
                audio_anchor=raw.audio_anchor_txt,        # type: ignore[union-attr]
                capture_meta=raw.capture_meta_json,       # type: ignore[union-attr]
                video_anchor=raw.video_anchor_txt,        # type: ignore[union-attr]
                # Segment clip: its own anchor's min bootNs overrides the parent
                # capture_meta t0 (segment PTS are rebased to 0).
                chop_video_anchor=(raw.chop_video_anchor  # type: ignore[union-attr]
                                   if getattr(raw, "is_chop", False) else None),
                show_spectrogram=True,
                log=self._log,
            )

        self._run_async(go, "Sync video + trajectory player")

    def _run_capture_diag(self) -> None:
        if not self._require(
            ("RAW folder", self.paths.raw),
            ("Output folder", self.paths.out_dir),
        ):
            return
        raw = self.paths.raw
        out = self.paths.out_dir / "capture_diag.html"   # type: ignore[union-attr]
        pos = self.paths.pos_path
        ftcsv = self.paths.frame_times_csv
        session_dir = raw.measurements_txt.parent        # type: ignore[union-attr]

        def go() -> None:
            from .stages import capture_diag_viewer as capdiag_stage
            res = capdiag_stage.build_capture_diag_viewer(
                session_dir=session_dir,
                out_html=out,
                pos_file=pos,
                frame_times=ftcsv,
                # Segment clip: diagnose against the segment's own anchor t0, not
                # the parent capture_meta video_t0_boottime_ns.
                chop_video_anchor=(raw.chop_video_anchor  # type: ignore[union-attr]
                                   if getattr(raw, "is_chop", False) else None),
                log=self._log,
            )
            self._log(f"[capture_diag] wrote {res.html_path}")

        self._run_async(go, "Capture diagnostics")

    def _run_kml_batch(self) -> None:
        if not self._require(
            ("Output folder", self.paths.out_dir),
            (".pos file", self.paths.pos_path),
        ):
            return
        out_dir = self.paths.out_dir / "kml_batch"   # type: ignore[union-attr]
        pos     = self.paths.pos_path
        raw     = self.paths.raw
        # Measurements is optional — device-derived layers skip gracefully if absent.
        measurements = raw.measurements_txt if raw is not None else None

        def go() -> None:
            res = kml_stage.export_all_kmls(
                pos_file=pos,                          # type: ignore[arg-type]
                measurements_txt=measurements,
                out_dir=out_dir,
                log=self._log,
            )
            self._log(f"[kml] {len(res.written)} files written to {out_dir}")
            for k, v in res.skipped.items():
                self._log(f"[kml]   skipped {k}: {v}")

        self._run_async(go, "KML batch export")

    def _run_all_smoother_kmls(self) -> None:
        if not self._require(
            ("Output folder", self.paths.out_dir),
            (".pos file", self.paths.pos_path),
        ):
            return
        out_dir = self.paths.out_dir / "kml_smoothers"   # type: ignore[union-attr]
        pos = self.paths.pos_path
        raw = self.paths.raw
        sensors = raw.sensors_txt if raw is not None else None
        stat = Path(str(pos) + ".stat") if pos else None
        if stat is not None and not stat.is_file():
            stat = None

        def go() -> None:
            res = kml_stage.export_smoother_kmls(
                pos_file=pos,               # type: ignore[arg-type]
                out_dir=out_dir,
                sensors_txt=sensors,
                stat_file=stat,
                log=self._log,
            )
            self._log(f"[kml-all] {len(res.written)} smoother KMLs → {out_dir}")
            for k, v in res.skipped.items():
                self._log(f"[kml-all]   skipped {k}: {v}")

        self._run_async(go, "All smoother KMLs")

    # ---- Client diagnostic viewers (post-Post-processing) ----
    def _client_viewers_out_dir(self) -> Optional[Path]:
        if self.paths.pos_path is None:
            return None
        base = self.paths.out_dir if self.paths.out_dir else self.paths.pos_path.parent
        return base / "viewers"

    def _open_path_in_default(self, path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except OSError as ex:
            messagebox.showerror("Open viewer", str(ex))

    def _run_client_compare(self) -> None:
        if not self._require((".pos file", self.paths.pos_path)):
            return
        from .client_viewers import make_smoother_comparison
        out_dir = self._client_viewers_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        pos = self.paths.pos_path                    # type: ignore[assignment]
        stat = pos.with_suffix(pos.suffix + ".stat")
        out = out_dir / "smoother_comparison.html"
        def go() -> None:
            try:
                make_smoother_comparison(
                    pos, out, stat_path=stat if stat.is_file() else None)
                self._log(f"[viewers] wrote {out}")
                self.root.after(0, lambda: self._open_path_in_default(out))
            except (RuntimeError, ValueError, OSError) as ex:
                self._log(f"[viewers] compare failed: {ex}")
        self._run_async(go, "Smoother comparison viewer")

    def _run_client_quality(self) -> None:
        if not self._require((".pos file", self.paths.pos_path)):
            return
        from .client_viewers import make_quality_panel
        out_dir = self._client_viewers_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        pos = self.paths.pos_path                    # type: ignore[assignment]
        out = out_dir / "quality_panel.html"
        def go() -> None:
            try:
                make_quality_panel(pos, out)
                self._log(f"[viewers] wrote {out}")
                self.root.after(0, lambda: self._open_path_in_default(out))
            except (RuntimeError, ValueError, OSError) as ex:
                self._log(f"[viewers] quality failed: {ex}")
        self._run_async(go, "Quality panel viewer")

    def _run_client_diff(self) -> None:
        if not self._require((".pos file", self.paths.pos_path)):
            return
        pos = self.paths.pos_path                    # type: ignore[assignment]
        clean = pos.with_name(pos.stem + "_clean.pos")
        if not clean.is_file():
            messagebox.showerror(
                "Raw vs Kalman diff",
                f"Need both raw and cleaned .pos.\n\n"
                f"Looked for cleaned at:\n  {clean}\n\n"
                "Run the full pipeline (Wizard or "
                "`data_pipeline-cli.exe pipeline ...`) to produce it.")
            return
        from .client_viewers import make_ppk_vs_kalman_diff
        out_dir = self._client_viewers_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "ppk_vs_kalman_diff.html"
        def go() -> None:
            try:
                make_ppk_vs_kalman_diff(pos, clean, out)
                self._log(f"[viewers] wrote {out}")
                self.root.after(0, lambda: self._open_path_in_default(out))
            except (RuntimeError, ValueError, OSError) as ex:
                self._log(f"[viewers] diff failed: {ex}")
        self._run_async(go, "Raw vs Kalman diff viewer")

    def _run_trust_pane(self) -> None:
        if not self._require((".pos file", self.paths.pos_path)):
            return
        pos = self.paths.pos_path
        raw = self.paths.raw
        sensors = raw.sensors_txt if raw else None
        out_dir = self._client_viewers_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "trust_pane.html"
        def go() -> None:
            try:
                from .stages.trust_pane import build_trust_pane
                build_trust_pane(pos, out, sensors_txt=sensors, log=self._log)
                self.root.after(0, lambda: self._open_path_in_default(out))
            except (RuntimeError, ValueError, OSError) as ex:
                self._log(f"[viewers] trust pane failed: {ex}")
        self._run_async(go, "Trust pane")

    def _run_trust_v2_pane(self) -> None:
        if not self._require((".pos file", self.paths.pos_path)):
            return
        pos = self.paths.pos_path
        raw = self.paths.raw
        sensors = raw.sensors_txt if raw else None
        out_dir = self._client_viewers_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "trust_v2.html"
        def go() -> None:
            try:
                from .stages.trust_pane_v2 import build_trust_pane_v2
                build_trust_pane_v2(pos, out, sensors_txt=sensors, log=self._log)
                self.root.after(0, lambda: self._open_path_in_default(out))
            except (RuntimeError, ValueError, OSError) as ex:
                self._log(f"[viewers] trust v2 pane failed: {ex}")
        self._run_async(go, "Trust v2 pane")

    def _run_client_vio(self) -> None:
        if not self._require((".pos file", self.paths.pos_path)):
            return
        raw = self.paths.raw
        if raw is None or not raw.recording_mp4.is_file():
            messagebox.showerror(
                "VIO viewer",
                "Need RAW folder with recording_*.mp4 + recording_*.txt.")
            return
        if not messagebox.askyesno(
            "VIO trajectory overlay",
            "VIO runs monocular essential-matrix on every frame. "
            "Typically 3-5 minutes for a 35-min video. Continue?"):
            return
        from .client_viewers import make_vio_overlay
        out_dir = self._client_viewers_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        pos = self.paths.pos_path                    # type: ignore[assignment]
        video = raw.recording_mp4
        rec_map = raw.recording_txt
        out = out_dir / "vio_overlay.html"
        def go() -> None:
            try:
                make_vio_overlay(
                    pos, video, rec_map, out,
                    capture_meta=raw.capture_meta_json,
                    video_anchor=raw.video_anchor_txt,
                    # Segment clip: its own anchor's min bootNs overrides the
                    # parent capture_meta t0 (segment PTS are rebased to 0).
                    chop_video_anchor=(raw.chop_video_anchor
                                       if getattr(raw, "is_chop", False)
                                       else None),
                    log=self._log,
                )
                self._log(f"[viewers] wrote {out}")
                self.root.after(0, lambda: self._open_path_in_default(out))
            except (FileNotFoundError, RuntimeError, ValueError, OSError) as ex:
                self._log(f"[viewers] VIO failed: {ex}")
        self._run_async(go, "VIO trajectory overlay")

    def _run_client_rtkplot(self) -> None:
        from .client_viewers import launch_rtkplot, RtkPlotArgs
        pos = self.paths.pos_path
        raw = self.paths.raw
        rover_obs = None
        if pos is not None:
            cand = pos.with_suffix(".obs")
            if cand.is_file():
                rover_obs = cand
        if rover_obs is None and raw is not None and raw.measurements_txt:
            cand = raw.measurements_txt.with_suffix(".obs")
            if cand.is_file():
                rover_obs = cand
        stat = None
        if pos is not None:
            cand = pos.with_suffix(pos.suffix + ".stat")
            if cand.is_file():
                stat = cand
        clean = None
        if pos is not None:
            cand = pos.with_name(pos.stem + "_clean.pos")
            if cand.is_file():
                clean = cand
        try:
            p = launch_rtkplot(RtkPlotArgs(
                rover_obs=rover_obs,
                pos_file=clean if clean else pos,
                stat_file=stat,
            ))
            self._log(f"[rtkplot] launched (pid={p.pid})")
        except (FileNotFoundError, OSError) as ex:
            messagebox.showerror("RTKPlot", str(ex))


def main() -> int:
    App().root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
