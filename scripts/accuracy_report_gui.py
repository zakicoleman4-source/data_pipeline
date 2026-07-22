"""Point-and-click front end for scripts/accuracy_report.py.

Pick the four shared inputs once, add one or more camera-model CSVs (e.g. two
project exports from the same capture), press Generate — the report is built
and opened in your browser. No command line needed.

    python scripts/accuracy_report_gui.py
"""
from __future__ import annotations
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
REPORT = HERE / "accuracy_report.py"


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Accuracy report")
        root.geometry("720x560")
        self.vars = {k: tk.StringVar() for k in ("gt", "track", "georef", "ftimes", "out")}
        self.vars["out"].set(str(Path.home() / "accuracy_report.html"))
        self._proj_by_iid: dict[str, tuple[str, str]] = {}  # tree iid -> (label, path)
        self._build()

    # ---- layout ----
    def _row(self, parent, r, label, key, kind, pattern):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=self.vars[key], width=64).grid(row=r, column=1, padx=6)
        ttk.Button(parent, text="Browse…",
                   command=lambda: self._pick(key, kind, pattern)).grid(row=r, column=2, padx=6)

    def _build(self):
        top = ttk.LabelFrame(self.root, text="Shared inputs (one per capture)")
        top.pack(fill="x", padx=10, pady=8)
        self._row(top, 0, "Ground-truth track (.pos)", "gt", "open", [("POS", "*.pos"), ("All", "*.*")])
        self._row(top, 1, "Device track (.pos)", "track", "open", [("POS", "*.pos"), ("All", "*.*")])
        self._row(top, 2, "Per-frame device coords (.csv)", "georef", "open", [("CSV", "*.csv"), ("All", "*.*")])
        self._row(top, 3, "Per-frame times (.csv)", "ftimes", "open", [("CSV", "*.csv"), ("All", "*.*")])

        mid = ttk.LabelFrame(self.root, text="Camera-model projects (add one CSV per project)")
        mid.pack(fill="both", expand=True, padx=10, pady=8)
        self.tree = ttk.Treeview(mid, columns=("label", "path"), show="headings", height=6)
        self.tree.heading("label", text="Label")
        self.tree.heading("path", text="cameras_est_computed.csv")
        self.tree.column("label", width=140)
        self.tree.column("path", width=520)
        self.tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        btns = ttk.Frame(mid)
        btns.pack(side="left", fill="y", padx=6, pady=6)
        ttk.Button(btns, text="Add project…", command=self._add_project).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove", command=self._remove_project).pack(fill="x", pady=2)

        bot = ttk.Frame(self.root)
        bot.pack(fill="x", padx=10, pady=8)
        ttk.Label(bot, text="Save report to").grid(row=0, column=0, sticky="w")
        ttk.Entry(bot, textvariable=self.vars["out"], width=58).grid(row=0, column=1, padx=6)
        ttk.Button(bot, text="…", width=3, command=self._pick_out).grid(row=0, column=2)

        self.run_btn = ttk.Button(self.root, text="Generate report", command=self._run)
        self.run_btn.pack(pady=4)
        self.status = tk.Text(self.root, height=7, wrap="word", state="disabled")
        self.status.pack(fill="both", expand=False, padx=10, pady=(0, 10))

    # ---- actions ----
    def _pick(self, key, kind, pattern):
        fn = filedialog.askopenfilename(filetypes=pattern) if kind == "open" else ""
        if fn:
            self.vars[key].set(fn)

    def _pick_out(self):
        fn = filedialog.asksaveasfilename(defaultextension=".html",
                                          filetypes=[("HTML", "*.html")])
        if fn:
            self.vars["out"].set(fn)

    def _add_project(self):
        fn = filedialog.askopenfilename(title="camera-model estimate CSV",
                                        filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not fn:
            return
        default = Path(fn).parent.name or Path(fn).stem
        label = _ask_label(self.root, default)
        if label is None:
            return
        iid = self.tree.insert("", "end", values=(label, fn))
        self._proj_by_iid[iid] = (label, fn)

    def _remove_project(self):
        # Key removal off the tree iid (not label/path equality) so two rows
        # with the same label+path (e.g. the same CSV added twice) don't both
        # get dropped when only one is selected.
        for iid in self.tree.selection():
            self._proj_by_iid.pop(iid, None)
            self.tree.delete(iid)

    @property
    def projects(self) -> list[tuple[str, str]]:
        """Current project list, in tree display order."""
        return [self._proj_by_iid[iid] for iid in self.tree.get_children()
                if iid in self._proj_by_iid]

    def _log(self, msg):
        self.status.configure(state="normal")
        self.status.insert("end", msg + "\n")
        self.status.see("end")
        self.status.configure(state="disabled")

    def _run(self):
        for key, name in (("gt", "ground-truth"), ("track", "device track"),
                          ("georef", "per-frame coords"), ("ftimes", "per-frame times")):
            if not self.vars[key].get().strip():
                messagebox.showerror("Missing input", f"Please choose the {name} file.")
                return
        if not self.projects:
            if not messagebox.askyesno("No projects",
                                       "No camera-model project added. Compare device GPS vs "
                                       "ground truth only?"):
                return
        cmd = [sys.executable, str(REPORT),
               "--gt", self.vars["gt"].get(), "--track", self.vars["track"].get(),
               "--georef", self.vars["georef"].get(), "--ftimes", self.vars["ftimes"].get(),
               "--out", self.vars["out"].get()]
        if self.projects:
            cmd.append("--meta")
            cmd += [f"{lbl}={path}" for lbl, path in self.projects]
        else:
            cmd.append("--no-meta")
        self.run_btn.configure(state="disabled")
        self._log("Running… " + " ".join(Path(c).name if os.sep in c else c for c in cmd))
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd):
        try:
            kwargs = {}
            if sys.platform == "win32":
                # Prevent a console window from flashing open behind the GUI.
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", **kwargs)
            out = (r.stdout or "") + (r.stderr or "")
            self.root.after(0, self._done, r.returncode, out)
        except Exception as e:  # pragma: no cover - GUI runtime guard
            self.root.after(0, self._done, 1, str(e))

    def _done(self, code, out):
        self.run_btn.configure(state="normal")
        for line in out.strip().splitlines()[-8:]:
            self._log(line)
        if code == 0:
            path = self.vars["out"].get()
            self._log(f"Done → {path}")
            try:
                webbrowser.open(Path(path).resolve().as_uri())
            except Exception:
                pass
        else:
            messagebox.showerror("Report failed", out.strip()[-1500:] or "unknown error")


def _ask_label(root, default):
    """Small modal to name the project; returns the label or None if cancelled."""
    dlg = tk.Toplevel(root)
    dlg.title("Project label")
    dlg.transient(root)
    dlg.grab_set()
    ttk.Label(dlg, text="Label for this project:").pack(padx=12, pady=(12, 4))
    var = tk.StringVar(value=default)
    ent = ttk.Entry(dlg, textvariable=var, width=30)
    ent.pack(padx=12)
    ent.focus_set()
    ent.select_range(0, "end")
    result = {"v": None}

    def ok():
        result["v"] = var.get().strip() or default
        dlg.destroy()

    def cancel():
        dlg.destroy()

    bar = ttk.Frame(dlg)
    bar.pack(pady=10)
    ttk.Button(bar, text="OK", command=ok).pack(side="left", padx=6)
    ttk.Button(bar, text="Cancel", command=cancel).pack(side="left", padx=6)
    dlg.bind("<Return>", lambda _e: ok())
    dlg.bind("<Escape>", lambda _e: cancel())
    root.wait_window(dlg)
    return result["v"]


def main():
    if not REPORT.is_file():
        print(f"accuracy_report.py not found next to this GUI: {REPORT}")
        sys.exit(1)
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
