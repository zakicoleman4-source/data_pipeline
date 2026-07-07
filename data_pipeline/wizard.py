"""One-click wizard for non-technical clients.

Opens a standalone Tkinter window that walks the user through:

    1. Drop / pick a RAW folder (the source app session).
    2. Drop / pick a reference input .obs file.
    3. Pick the nav/auxiliary data folder (auto-suggested from base.obs sibling).
    4. Pick an output folder (auto-suggested next to RAW).
    5. Click "Run pipeline" -> :func:`data_pipeline.pipeline_full.run_full`
       in a worker thread; live log streams into the window; final accuracy
       report renders in a dialog.

Designed to be the only entry-point a client ever has to learn. The full
GUI (multi-tab, per-stage knobs, viewers) still ships for power users.

Usage:

    python -m data_pipeline.wizard               # standalone
    data_pipeline.exe wizard                     # via PyInstaller launcher
    data_pipeline.wizard.open_window(parent)     # embed in main GUI

Design notes:
* No worker-thread mutation of Tk widgets -- log lines flow through a
  :class:`queue.Queue` polled from the Tk main loop (POLL_MS).
* Auto-detect helpers fail soft: if a sibling folder doesn't exist, the
  field stays blank, the user fills it in, and the wizard validates on
  Run.
* ``pipeline_full.run_full`` already raises actionable errors for every
  missing input, so the wizard relies on those messages instead of
  duplicating validation here.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


POLL_MS = 80


@dataclass
class _Paths:
    raw_folder:  Optional[Path] = None
    base_obs:    Optional[Path] = None
    nav_dir:     Optional[Path] = None
    out_dir:     Optional[Path] = None


def _suggest_nav_dir(base_obs: Optional[Path]) -> Optional[Path]:
    """Most lab sessions keep nav files in the same folder as base.obs.
    Return that folder when at least one nav-extension file lives there.
    """
    if base_obs is None or not base_obs.is_file():
        return None
    folder = base_obs.parent
    nav_ext_patterns = ("*.[0-9][0-9][nglpNGLP]", "*.nav", "*.NAV",
                        "*.sp3", "*.SP3", "*.clk", "*.CLK")
    for pat in nav_ext_patterns:
        if any(folder.glob(pat)):
            return folder
    return None


def _suggest_out_dir(raw_folder: Optional[Path]) -> Optional[Path]:
    """Default output sits next to the RAW folder as ``<RAW>_out``."""
    if raw_folder is None or not raw_folder.is_dir():
        return None
    return raw_folder.parent / f"{raw_folder.name}_out"


class _WizardWindow:
    """One-shot wizard window. Owns its own paths, log queue, worker."""

    def __init__(self, parent: Optional[tk.Misc] = None) -> None:
        self.parent = parent
        self.root: tk.Misc
        if parent is None:
            self.root = tk.Tk()
            self._owns_mainloop = True
        else:
            self.root = tk.Toplevel(parent)
            self._owns_mainloop = False
        self.root.title("data_pipeline -- One-click PPK pipeline")
        try:
            self.root.geometry("780x600")
            self.root.minsize(660, 480)
        except tk.TclError:
            pass

        self.paths = _Paths()
        self._log_q: queue.Queue[str] = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._busy = False
        # Populated by the worker on success so post-run viewer buttons
        # know which files to read.
        self._last_raw_pos:   Optional[Path] = None
        self._last_clean_pos: Optional[Path] = None
        self._last_stat:      Optional[Path] = None

        self._build_ui()
        self.root.after(POLL_MS, self._drain_log_queue)

    # ------- UI construction -------
    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}
        head = ttk.Frame(self.root)
        head.pack(fill="x", **pad)
        ttk.Label(
            head,
            text="One-click PPK pipeline",
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left")
        ttk.Label(
            head,
            text="Drop your files, hit Run. Accuracy report at the end.",
            foreground="#666",
        ).pack(side="left", padx=12)

        # Input fields.
        form = ttk.LabelFrame(self.root, text="Inputs")
        form.pack(fill="x", **pad)
        form.columnconfigure(1, weight=1)

        self._raw_var  = tk.StringVar()
        self._base_var = tk.StringVar()
        self._nav_var  = tk.StringVar()
        self._out_var  = tk.StringVar()

        rows = [
            ("RAW folder",  self._raw_var,  self._pick_raw,
             "Folder with measurements_*.txt, recording_*.txt/.mp4, sensors_*.txt"),
            ("Base .obs",   self._base_var, self._pick_base,
             "Survey-grade base-station RINEX OBS"),
            ("Nav folder",  self._nav_var,  self._pick_nav,
             "Folder with .26N / .26G / .26L / .26C nav files (auto-filled from base.obs sibling)"),
            ("Output dir",  self._out_var,  self._pick_out,
             "Where cleaned .pos + features CSV land (auto-suggested next to RAW)"),
        ]
        for row, (label, var, picker, tip) in enumerate(rows):
            ttk.Label(form, text=label).grid(
                row=row, column=0, sticky="e", padx=6, pady=3)
            e = ttk.Entry(form, textvariable=var)
            e.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
            ttk.Button(form, text="Browse...", command=picker).grid(
                row=row, column=2, padx=6, pady=3)
            ttk.Label(form, text=tip, foreground="#888").grid(
                row=row, column=1, columnspan=2, sticky="w", padx=6)

        # Action row.
        action = ttk.Frame(self.root)
        action.pack(fill="x", **pad)
        self._run_btn = ttk.Button(
            action, text="Run pipeline", command=self._on_run,
        )
        self._run_btn.pack(side="left")
        ttk.Button(action, text="Open output folder",
                   command=self._open_out_folder).pack(side="left", padx=8)
        ttk.Button(action, text="Doctor (check tools)",
                   command=self._on_doctor).pack(side="left", padx=8)
        self._status_lbl = ttk.Label(action, text="Idle", foreground="#888")
        self._status_lbl.pack(side="right")

        # Viewer row (populated after a successful pipeline run).
        viewers = ttk.LabelFrame(self.root, text="Viewers")
        viewers.pack(fill="x", **pad)
        self._viewer_buttons = [
            ttk.Button(viewers, text="Smoother comparison",
                       command=self._view_compare, state="disabled"),
            ttk.Button(viewers, text="Quality panel (ns / speed / sigma / Q)",
                       command=self._view_quality, state="disabled"),
            ttk.Button(viewers, text="Raw vs Kalman diff",
                       command=self._view_diff, state="disabled"),
            ttk.Button(viewers, text="VIO trajectory overlay (SLOW: ~3-5 min)",
                       command=self._view_vio, state="disabled"),
            ttk.Button(viewers, text="Open in RTKPlot",
                       command=self._view_rtkplot, state="disabled"),
        ]
        for b in self._viewer_buttons:
            b.pack(side="left", padx=4, pady=4)

        # Progress bar.
        self._progress = ttk.Progressbar(
            self.root, mode="indeterminate", length=200,
        )
        self._progress.pack(fill="x", padx=8, pady=2)

        # Log box.
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self._log = tk.Text(log_frame, wrap="none", height=18,
                            bg="#0a1020", fg="#e2e8f0", insertbackground="#fff")
        ys = ttk.Scrollbar(log_frame, orient="vertical",
                           command=self._log.yview)
        xs = ttk.Scrollbar(log_frame, orient="horizontal",
                           command=self._log.xview)
        self._log.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self._log.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

    # ------- Pickers -------
    def _pick_raw(self) -> None:
        d = filedialog.askdirectory(title="Pick the RAW (the capture app) folder")
        if d:
            self.paths.raw_folder = Path(d)
            self._raw_var.set(d)
            self._auto_fill_out()

    def _pick_base(self) -> None:
        f = filedialog.askopenfilename(
            title="Pick the base-station .obs file",
            filetypes=[("RINEX obs", "*.obs *.[0-9][0-9]o *.[0-9][0-9]O"),
                       ("All files", "*.*")],
        )
        if f:
            self.paths.base_obs = Path(f)
            self._base_var.set(f)
            self._auto_fill_nav()

    def _pick_nav(self) -> None:
        d = filedialog.askdirectory(
            title="Pick the folder containing nav files (.26N/.26G/etc)")
        if d:
            self.paths.nav_dir = Path(d)
            self._nav_var.set(d)

    def _pick_out(self) -> None:
        d = filedialog.askdirectory(title="Pick output folder")
        if d:
            self.paths.out_dir = Path(d)
            self._out_var.set(d)

    def _auto_fill_nav(self) -> None:
        nav = _suggest_nav_dir(self.paths.base_obs)
        if nav and not self._nav_var.get():
            self.paths.nav_dir = nav
            self._nav_var.set(str(nav))
            self._log_q.put(f"[wizard] nav folder auto-detected: {nav}")

    def _auto_fill_out(self) -> None:
        out = _suggest_out_dir(self.paths.raw_folder)
        if out and not self._out_var.get():
            self.paths.out_dir = out
            self._out_var.set(str(out))
            self._log_q.put(f"[wizard] output folder suggested: {out}")

    def _open_out_folder(self) -> None:
        out = self._out_var.get().strip()
        if not out:
            messagebox.showinfo("Output folder",
                                "No output folder set yet.")
            return
        p = Path(out)
        if not p.is_dir():
            messagebox.showinfo(
                "Output folder",
                f"Folder does not exist yet:\n{p}\n\n"
                "Run the pipeline first.")
            return
        try:
            if os.name == "nt":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", str(p)], check=False)
            else:
                import subprocess
                subprocess.run(["xdg-open", str(p)], check=False)
        except OSError as ex:
            messagebox.showerror("Open folder", f"Could not open {p}\n{ex}")

    def _on_doctor(self) -> None:
        from .lab_tools import report as tool_report
        rep = tool_report()
        lines = [f"{name:12s} {path}" for name, path in rep.items()]
        messagebox.showinfo(
            "External tool resolution",
            "\n".join(lines) +
            "\n\nMISSING entries mean the tool isn't bundled and the "
            "wizard will fall back to env vars / PATH if available.",
        )

    # ------- Run -------
    def _on_run(self) -> None:
        if self._busy:
            messagebox.showinfo("Wizard", "Pipeline already running.")
            return
        # Validate inputs locally for fast feedback.
        raw = self._raw_var.get().strip()
        base = self._base_var.get().strip()
        nav = self._nav_var.get().strip()
        out = self._out_var.get().strip()
        if not raw or not base or not out:
            messagebox.showerror(
                "Wizard",
                "Please fill in RAW folder, Base .obs, and Output dir.\n"
                "Nav folder is auto-filled from base.obs sibling but you "
                "can override it.",
            )
            return
        if not nav:
            self._nav_var.set(str(_suggest_nav_dir(Path(base)) or ""))
            nav = self._nav_var.get().strip()
            if not nav:
                messagebox.showerror(
                    "Wizard",
                    "Couldn't auto-detect a nav folder. Pick the folder "
                    "containing your .26N/.26G/.26L/.26C nav files.")
                return
        self._busy = True
        self._set_status("Running...", "#e0a020")
        self._run_btn.config(state="disabled")
        self._progress.start(80)
        self._log.delete("1.0", "end")
        self._worker = threading.Thread(
            target=self._worker_main,
            args=(Path(raw), Path(base), Path(nav), Path(out)),
            daemon=True,
        )
        self._worker.start()

    def _worker_main(
        self, raw: Path, base: Path, nav: Path, out: Path,
    ) -> None:
        # Worker thread: never touch Tk widgets directly. Push to log queue.
        def log_cb(msg: str) -> None:
            self._log_q.put(msg)
        try:
            from .pipeline_full import run_full
            result = run_full(
                raw_folder=raw,
                base_obs=base,
                nav_dir=nav,
                out_dir=out,
                log=log_cb,
            )
            self._log_q.put(_SENTINEL_DONE + repr({
                "cleaned_pos":  str(result.cleaned_pos),
                "raw_pos":      str(result.raw_pos),
                "features_csv": str(result.features_csv) if result.features_csv else "",
                "base_source":  result.base_source,
                "ci95_h_m":     result.accuracy.ci95_h_m,
                "ci95_v_m":     result.accuracy.ci95_v_m,
                "fix_pct":      result.accuracy.fix_pct,
                "float_pct":    result.accuracy.float_pct,
                "single_pct":   result.accuracy.single_pct,
                "n_epochs":     result.n_epochs,
                "duration_min": result.accuracy.duration_min,
                "source_chain": result.accuracy.source_chain,
            }))
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as ex:
            self._log_q.put(_SENTINEL_FAIL + f"{type(ex).__name__}: {ex}")
        except Exception as ex:  # pragma: no cover -- worker crash escape
            tb = traceback.format_exc()
            self._log_q.put(_SENTINEL_FAIL + f"Unhandled {type(ex).__name__}: {ex}\n{tb}")

    # ------- Log pump -------
    def _drain_log_queue(self) -> None:
        try:
            while True:
                msg = self._log_q.get_nowait()
                if msg.startswith(_SENTINEL_DONE):
                    payload = msg[len(_SENTINEL_DONE):]
                    self._finish_ok(payload)
                elif msg.startswith(_SENTINEL_FAIL):
                    payload = msg[len(_SENTINEL_FAIL):]
                    self._finish_fail(payload)
                elif msg.startswith(_SENTINEL_VIEW_DONE):
                    out_path = msg[len(_SENTINEL_VIEW_DONE):]
                    self._progress.stop()
                    self._set_status("Done", "#10b981")
                    if out_path:
                        self._open_html(Path(out_path))
                elif msg.startswith(_SENTINEL_VIEW_FAIL):
                    self._progress.stop()
                    self._set_status("View failed", "#ef4444")
                    messagebox.showerror(
                        "Viewer failed",
                        msg[len(_SENTINEL_VIEW_FAIL):])
                else:
                    self._log.insert("end", msg + "\n")
                    self._log.see("end")
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._drain_log_queue)

    def _finish_ok(self, payload_repr: str) -> None:
        import ast
        try:
            d = ast.literal_eval(payload_repr)
        except (SyntaxError, ValueError):
            d = {}
        self._busy = False
        self._progress.stop()
        self._run_btn.config(state="normal")
        self._set_status("Done", "#10b981")
        # Stash artifact paths so the Viewer buttons know what to read.
        rp = d.get("raw_pos", "")
        cp = d.get("cleaned_pos", "")
        self._last_raw_pos = Path(rp) if rp else None
        self._last_clean_pos = Path(cp) if cp else None
        if self._last_raw_pos is not None:
            stat_cand = self._last_raw_pos.with_suffix(".pos.stat")
            self._last_stat = stat_cand if stat_cand.is_file() else None
        for b in self._viewer_buttons:
            b.config(state="normal")
        ci_h = d.get("ci95_h_m", float("nan"))
        ci_v = d.get("ci95_v_m", float("nan"))
        msg = (
            "Pipeline finished.\n\n"
            f"Epochs:         {d.get('n_epochs', '?')}\n"
            f"Duration:       {d.get('duration_min', 0.0):.1f} min\n"
            f"Quality:        Fix {d.get('fix_pct', 0):.1f}%  "
            f"Float {d.get('float_pct', 0):.1f}%  "
            f"Single {d.get('single_pct', 0):.1f}%\n\n"
            f"Assumed accuracy (95% CI):\n"
            f"   horizontal  +/- {ci_h:.2f} m\n"
            f"   vertical    +/- {ci_v:.2f} m\n\n"
            f"Source chain: {d.get('source_chain', '')}\n"
            f"Base coords:  {d.get('base_source', '')}\n\n"
            f"Cleaned .pos: {d.get('cleaned_pos', '')}\n"
            f"Features CSV: {d.get('features_csv', '(skipped)')}\n\n"
            "Note: device PPK floor is ~3 m horiz / 6 m vert (1-sigma)."
        )
        messagebox.showinfo("Pipeline finished", msg)

    def _finish_fail(self, err_msg: str) -> None:
        self._busy = False
        self._progress.stop()
        self._run_btn.config(state="normal")
        self._set_status("Failed", "#ef4444")
        messagebox.showerror("Pipeline failed", err_msg)

    # ------- Viewer launchers -------
    def _open_html(self, html_path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(html_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", str(html_path)], check=False)
            else:
                import subprocess
                subprocess.run(["xdg-open", str(html_path)], check=False)
        except OSError as ex:
            messagebox.showerror("Open viewer", str(ex))

    def _viewer_out_dir(self) -> Optional[Path]:
        if self._last_raw_pos is None:
            return None
        return self._last_raw_pos.parent / "viewers"

    def _view_compare(self) -> None:
        if self._last_raw_pos is None:
            return
        from .client_viewers import make_smoother_comparison
        out_dir = self._viewer_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "smoother_comparison.html"
        try:
            make_smoother_comparison(
                self._last_raw_pos, out, stat_path=self._last_stat,
            )
        except (RuntimeError, ValueError, OSError) as ex:
            messagebox.showerror("Smoother comparison", str(ex))
            return
        self._open_html(out)

    def _view_quality(self) -> None:
        if self._last_raw_pos is None:
            return
        from .client_viewers import make_quality_panel
        out_dir = self._viewer_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "quality_panel.html"
        try:
            make_quality_panel(self._last_raw_pos, out)
        except (RuntimeError, ValueError, OSError) as ex:
            messagebox.showerror("Quality panel", str(ex))
            return
        self._open_html(out)

    def _view_diff(self) -> None:
        if self._last_raw_pos is None or self._last_clean_pos is None:
            return
        from .client_viewers import make_ppk_vs_kalman_diff
        out_dir = self._viewer_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "ppk_vs_kalman_diff.html"
        try:
            make_ppk_vs_kalman_diff(
                self._last_raw_pos, self._last_clean_pos, out,
            )
        except (RuntimeError, ValueError, OSError) as ex:
            messagebox.showerror("Diff viewer", str(ex))
            return
        self._open_html(out)

    def _view_vio(self) -> None:
        if self._last_raw_pos is None:
            return
        raw_folder_str = self._raw_var.get().strip()
        if not raw_folder_str:
            messagebox.showerror("VIO viewer",
                                 "Need the RAW folder (video + recording_*.txt).")
            return
        raw_folder = Path(raw_folder_str)
        # Resolve via RawInputs so a cut ("segment") session picks the segment container file +
        # segment anchor (not the parent full session), and the segment t0 reaches the
        # Motion model sample->UTC mapping. Globbing recording_*.container file directly would pick the
        # parent media for a segment and drop the segment t0 -> samples mapped ~t0 early.
        try:
            from .pipeline import RawInputs
            raw = RawInputs.from_folder(raw_folder)
        except Exception as ex:
            messagebox.showerror(
                "VIO viewer",
                f"Could not resolve capture inputs in {raw_folder}: {ex}")
            return
        if raw.recording_mp4 is None or raw.recording_txt is None:
            messagebox.showerror(
                "VIO viewer",
                f"Need both a video and a recording_*.txt in {raw_folder}.")
            return
        if not messagebox.askyesno(
            "VIO trajectory overlay",
            "VIO computation runs monocular essential-matrix on every "
            "video frame -- typically 3-5 minutes for a 35-minute "
            "session. Continue?"):
            return
        from .client_viewers import make_vio_overlay
        out_dir = self._viewer_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "vio_overlay.html"
        pos_in = self._last_clean_pos or self._last_raw_pos
        video = raw.recording_mp4
        rec_map = raw.recording_txt
        _chop_anchor = raw.chop_video_anchor if getattr(raw, "is_chop", False) else None

        def worker() -> None:
            try:
                make_vio_overlay(
                    pos_in, video, rec_map, out,
                    capture_meta=raw.capture_meta_json,
                    video_anchor=raw.video_anchor_txt,
                    chop_video_anchor=_chop_anchor,
                    log=lambda m: self._log_q.put(m),
                )
                self._log_q.put(_SENTINEL_VIEW_DONE + str(out))
            except (FileNotFoundError, RuntimeError, ValueError, OSError) as ex:
                self._log_q.put(
                    _SENTINEL_VIEW_FAIL +
                    f"{type(ex).__name__}: {ex}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._set_status("VIO running...", "#e0a020")
        self._progress.start(80)

    def _view_rtkplot(self) -> None:
        from .client_viewers import launch_rtkplot, RtkPlotArgs
        if self._last_raw_pos is None:
            return
        rover_obs = self._last_raw_pos.with_suffix(".obs")
        base_obs = Path(self._base_var.get().strip()) if self._base_var.get() else None
        try:
            launch_rtkplot(RtkPlotArgs(
                rover_obs=rover_obs if rover_obs.is_file() else None,
                base_obs=base_obs,
                pos_file=self._last_clean_pos
                          if (self._last_clean_pos and self._last_clean_pos.is_file())
                          else self._last_raw_pos,
                stat_file=self._last_stat,
            ))
        except (FileNotFoundError, OSError) as ex:
            messagebox.showerror("RTKPlot", str(ex))

    def _set_status(self, text: str, color: str) -> None:
        try:
            self._status_lbl.config(text=text, foreground=color)
        except tk.TclError:
            pass

    # ------- Run loop -------
    def run(self) -> None:
        if self._owns_mainloop:
            self.root.mainloop()


_SENTINEL_DONE = "<<__WIZARD_DONE__>>"
_SENTINEL_FAIL = "<<__WIZARD_FAIL__>>"
_SENTINEL_VIEW_DONE = "<<__VIEW_DONE__>>"
_SENTINEL_VIEW_FAIL = "<<__VIEW_FAIL__>>"


def open_window(parent: Optional[tk.Misc] = None) -> _WizardWindow:
    """Open the wizard. Pass ``parent`` to embed; omit for standalone."""
    w = _WizardWindow(parent=parent)
    return w


def main() -> int:
    w = open_window(parent=None)
    w.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
