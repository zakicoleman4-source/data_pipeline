"""Reusable Tk widgets and persistence helpers for the GUI.

Keeping these out of ``gui.py`` (already big) makes the main App class
read as orchestration rather than widget plumbing. None of these
classes depend on anything in this package -- pure Tk wrappers.
"""
from __future__ import annotations

import json
import sys
import time
import tkinter as tk
import tkinter.ttk as ttk
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Scrollable sample
# ---------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    """A Tk sample whose contents scroll vertically.

    Usage::

        sf = ScrollableFrame(parent)
        sf.pack(fill="both", expand=True)
        # add children to sf.body, not sf:
        ttk.Label(sf.body, text="hello").pack()

    Mouse-wheel binding only fires when the pointer is over the canvas so it
    doesn't steal scroll events from other widgets.
    """

    def __init__(self, parent: tk.Misc, *, padding: int = 4) -> None:
        super().__init__(parent)
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self._vbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vbar.set)
        self._vbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(self._canvas, padding=padding)
        self._win = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body_config)
        self._canvas.bind("<Configure>", self._on_canvas_config)
        # Wheel binding: install / remove on pointer enter / leave so we
        # don't compete with other scrollables in the same window.
        self._canvas.bind("<Enter>", self._bind_wheel)
        self._canvas.bind("<Leave>", self._unbind_wheel)

    def _on_body_config(self, _e: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_config(self, e: tk.Event) -> None:
        self._canvas.itemconfigure(self._win, width=e.width)

    def _bind_wheel(self, _e: tk.Event) -> None:
        if sys.platform == "darwin":
            self._canvas.bind_all("<MouseWheel>", self._on_wheel_mac)
        else:
            self._canvas.bind_all("<MouseWheel>", self._on_wheel)
            self._canvas.bind_all("<Button-4>", self._on_wheel_linux_up)
            self._canvas.bind_all("<Button-5>", self._on_wheel_linux_down)

    def _unbind_wheel(self, _e: tk.Event) -> None:
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self._canvas.unbind_all(seq)
            except tk.TclError:
                pass

    def _on_wheel(self, e: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def _on_wheel_mac(self, e: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * e.delta), "units")

    def _on_wheel_linux_up(self, _e: tk.Event) -> None:
        self._canvas.yview_scroll(-3, "units")

    def _on_wheel_linux_down(self, _e: tk.Event) -> None:
        self._canvas.yview_scroll(3, "units")


# ---------------------------------------------------------------------------
# Tooltip
# ---------------------------------------------------------------------------

class Tooltip:
    """Lightweight hover tooltip. ``Tooltip(widget, "text")`` is enough."""

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: Optional[str] = None
        self._tw: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")

    def _schedule(self, _e: tk.Event) -> None:
        self._cancel(_e)
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self, _e: tk.Event) -> None:
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._tw is not None:
            try:
                self._tw.destroy()
            except tk.TclError:
                pass
            self._tw = None

    def _show(self) -> None:
        if self._tw is not None:
            return
        x = self.widget.winfo_rootx() + 14
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tw = tk.Toplevel(self.widget)
        self._tw.wm_overrideredirect(True)
        self._tw.wm_geometry(f"+{x}+{y}")
        try:
            self._tw.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        lbl = tk.Label(
            self._tw, text=self.text, justify="left",
            background="#1f2937", foreground="#f3f4f6",
            relief="solid", borderwidth=1, padx=8, pady=4,
            wraplength=380, font=("Segoe UI", 9),
        )
        lbl.pack()


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

class StatusBar(ttk.Frame):
    """Bottom strip showing stage state, elapsed time, and an indeterminate
    progress bar that animates while a stage is busy."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=(8, 2))
        self._state = tk.StringVar(value="ready")
        self._stage = tk.StringVar(value="")
        self._elapsed = tk.StringVar(value="")
        self._t0: Optional[float] = None

        # Progress indicator
        self._bar = ttk.Progressbar(self, mode="indeterminate", length=120)
        self._bar.pack(side="left", padx=(0, 8))

        ttk.Label(self, textvariable=self._state, foreground="#374151").pack(side="left")
        ttk.Label(self, text="    ").pack(side="left")
        ttk.Label(self, textvariable=self._stage, foreground="#6b7280",
                  font=("Segoe UI", 9, "italic")).pack(side="left")
        ttk.Label(self, textvariable=self._elapsed, foreground="#6b7280").pack(side="right")

        self._tick_after_id: Optional[str] = None

    def start(self, stage: str) -> None:
        self._t0 = time.monotonic()
        self._state.set("running")
        self._stage.set(stage)
        self._bar.start(80)
        self._tick()

    def stop(self, outcome: str = "ready") -> None:
        self._t0 = None
        self._bar.stop()
        self._state.set(outcome)
        if self._tick_after_id is not None:
            try:
                self.after_cancel(self._tick_after_id)
            except tk.TclError:
                pass
            self._tick_after_id = None

    def _tick(self) -> None:
        if self._t0 is None:
            return
        dt = time.monotonic() - self._t0
        m, s = divmod(int(dt), 60)
        self._elapsed.set(f"{m:d}:{s:02d}")
        self._tick_after_id = self.after(500, self._tick)


# ---------------------------------------------------------------------------
# App state (persistent settings)
# ---------------------------------------------------------------------------

def _default_state_path() -> Path:
    """Return the per-user state file location.

    Windows: %APPDATA%\\data_pipeline\\state.json
    Other:   ~/.config/data_pipeline/state.json
    """
    if sys.platform == "win32":
        base = Path.home() / "AppData" / "Roaming" / "data_pipeline"
    else:
        base = Path.home() / ".config" / "data_pipeline"
    return base / "state.json"


@dataclass
class AppState:
    """Persisted across GUI sessions (paths, window geometry, knob values)."""

    last_raw_folder: str = ""
    last_arnx: str = ""
    last_out_dir: str = ""
    last_pos_path: str = ""
    last_obs_path: str = ""
    last_sync_basemap: str = ""
    geometry: str = "1100x820"
    sash_position: int = 540

    # Knob values mirrored from CsvOptions so reopening the app
    # restores the last-used pipeline tuning.
    smoothing: str = "car"
    fps: float = 6.0
    pts_name_decimals: int = 6
    image_format: str = "png"
    xy_sigma_s: float = 2.0
    z_sigma_s: float = 10.0
    use_rts: bool = False
    add_ypr: bool = True
    include_alt: bool = False
    acc_xy: float = 0.10
    acc_z: float = 0.30
    acc_yaw: float = 10.0
    acc_pitch: float = 5.0
    acc_roll: float = 5.0
    ypr_sigma: float = 3.0
    decimate_orient_hz: float = 10.0
    max_gap: float = 2.0

    recent_raw_folders: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppState":
        p = path or _default_state_path()
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {f.name for f in cls.__dataclass_fields__.values()}
            return cls(**{k: v for k, v in data.items() if k in known})
        except Exception:
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        p = path or _default_state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(p)

    def remember_raw(self, folder: str, max_keep: int = 10) -> None:
        if not folder:
            return
        if folder in self.recent_raw_folders:
            self.recent_raw_folders.remove(folder)
        self.recent_raw_folders.insert(0, folder)
        del self.recent_raw_folders[max_keep:]


# ---------------------------------------------------------------------------
# Theme (lightweight ttk style sheet)
# ---------------------------------------------------------------------------

def apply_modern_theme(root: tk.Tk) -> None:
    """Install the modern ttk style sheet shared across all tabs.

    Design tokens (Tailwind-inspired, kept terse):
      bg=#f5f7fb   surface=#ffffff   border=#e2e8f0
      text=#0f172a faded=#475569     muted=#94a3b8
      primary=#2563eb  primary-hover=#1d4ed8
      success=#16a34a  warning=#d97706  danger=#dc2626
      gold=#fbbf24
    Sized for a comfortable 1440x900 laptop screen: 11 pt body, 13 pt
    section headings, 18 pt page titles. Falls back silently on
    platforms where a particular theme isn't available.
    """
    style = ttk.Style(root)
    for candidate in ("clam", "alt", "default"):
        try:
            style.theme_use(candidate)
            break
        except tk.TclError:
            continue

    # ── tokens ──
    bg = "#f5f7fb"
    surface = "#ffffff"
    border = "#e2e8f0"
    text = "#0f172a"
    faded = "#475569"
    muted = "#94a3b8"
    primary = "#2563eb"
    primary_hover = "#1d4ed8"
    success = "#16a34a"
    danger = "#dc2626"
    body = ("Segoe UI", 11)
    body_b = ("Segoe UI", 11, "bold")
    section = ("Segoe UI", 12, "bold")
    title = ("Segoe UI", 16, "bold")
    sub = ("Segoe UI", 10)

    root.configure(bg=bg)

    # ── base ──
    style.configure(".", background=bg, foreground=text, font=body)
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=text, font=body)
    style.configure("Title.TLabel", background=bg, foreground=text, font=title)
    style.configure("Section.TLabel", background=bg, foreground=text, font=section)
    style.configure("Sub.TLabel", background=bg, foreground=faded, font=sub)
    style.configure("Muted.TLabel", background=bg, foreground=muted, font=sub)

    # ── samples / cards ──
    style.configure("Card.TFrame", background=surface, relief="solid",
                    borderwidth=1)
    style.configure("TLabelframe", background=bg, foreground=text,
                    borderwidth=1, relief="solid", padding=(8, 6))
    style.configure("TLabelframe.Label", background=bg, foreground=faded,
                    font=section)

    # ── buttons: default / primary / success / danger ──
    style.configure("TButton", padding=(12, 6), font=body, relief="flat",
                    background="#e5e7eb", foreground=text, borderwidth=0)
    style.map("TButton",
              background=[("active", "#cbd5e1"), ("disabled", "#f1f5f9")],
              foreground=[("disabled", muted)])
    style.configure("Primary.TButton", padding=(14, 7), font=body_b,
                    foreground="white", background=primary, borderwidth=0)
    style.map("Primary.TButton",
              background=[("active", primary_hover), ("disabled", "#cbd5e1")],
              foreground=[("disabled", "#ffffff")])
    style.configure("Success.TButton", padding=(14, 7), font=body_b,
                    foreground="white", background=success, borderwidth=0)
    style.map("Success.TButton",
              background=[("active", "#15803d"), ("disabled", "#cbd5e1")])
    style.configure("Danger.TButton", padding=(14, 7), font=body_b,
                    foreground="white", background=danger, borderwidth=0)
    style.map("Danger.TButton",
              background=[("active", "#b91c1c"), ("disabled", "#cbd5e1")])
    # Back-compat alias used by older code.
    style.configure("Accent.TButton", padding=(14, 7), font=body_b,
                    foreground="white", background=primary, borderwidth=0)
    style.map("Accent.TButton",
              background=[("active", primary_hover), ("disabled", "#cbd5e1")])

    # ── notebook tabs ──
    style.configure("TNotebook", background=bg, borderwidth=0, tabmargins=(4, 6, 4, 0))
    style.configure("TNotebook.Tab", padding=(18, 9), font=body)
    style.map("TNotebook.Tab",
              background=[("selected", surface), ("!selected", "#e2e8f0")],
              foreground=[("selected", primary), ("!selected", faded)],
              font=[("selected", body_b)])

    # ── entries / combobox ──
    style.configure("TEntry", padding=4, fieldbackground=surface,
                    bordercolor=border, lightcolor=border, darkcolor=border)
    style.map("TEntry", bordercolor=[("focus", primary)])
    style.configure("TCombobox", padding=4, fieldbackground=surface,
                    bordercolor=border, arrowcolor=faded)

    # ── checkbutton / radiobutton ──
    style.configure("TCheckbutton", background=bg, foreground=text, font=body)
    style.configure("TRadiobutton", background=bg, foreground=text, font=body)

    # ── tree (used in result tables) ──
    style.configure("Treeview", background=surface, fieldbackground=surface,
                    foreground=text, rowheight=26, font=body,
                    bordercolor=border, lightcolor=border, darkcolor=border)
    style.configure("Treeview.Heading", font=body_b, background="#eef2f7",
                    foreground=faded, padding=(8, 6))
    style.map("Treeview.Heading", background=[("active", "#dbeafe")])
    style.map("Treeview", background=[("selected", "#dbeafe")],
              foreground=[("selected", text)])

    # ── progress bar ──
    style.configure("TProgressbar", troughcolor="#e2e8f0",
                    background=primary, thickness=10)


# ---------------------------------------------------------------------------
# Status pill — coloured chip used to show "Idle / Running / Done / Error"
# ---------------------------------------------------------------------------

class StatusPill(ttk.Label):
    """Coloured chip showing one of {idle, running, done, error}.

    Use as a tiny inline indicator next to a workflow step. The label
    text is short ("Idle", "Done", …) and the background flips through
    a fixed palette so the user can scan a long form for what's left.
    """

    _COLOURS = {
        "idle":    ("#e2e8f0", "#475569", "Idle"),
        "running": ("#fef3c7", "#92400e", "Running…"),
        "done":    ("#dcfce7", "#166534", "Done"),
        "error":   ("#fee2e2", "#991b1b", "Error"),
    }

    def __init__(self, master: tk.Widget, initial: str = "idle") -> None:
        bg, fg, text = self._COLOURS[initial]
        super().__init__(master, text=f"  {text}  ", background=bg,
                         foreground=fg, font=("Segoe UI", 9, "bold"),
                         padding=(2, 2), borderwidth=0)
        self._state = initial

    def set_state(self, state: str) -> None:
        """Switch the pill to one of: idle / running / done / error."""
        if state not in self._COLOURS:
            state = "idle"
        bg, fg, text = self._COLOURS[state]
        self.configure(text=f"  {text}  ", background=bg, foreground=fg)
        self._state = state

    def state_name(self) -> str:
        return self._state
