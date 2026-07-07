"""PyInstaller entry script for ``data_pipeline.exe``.

Default behaviour: launch the Tkinter GUI (``data_pipeline.gui.main``).

Subcommands (so the same exe doubles as a CLI on machines without a
display, in scripts, or for headless end-to-end smoke tests):

    data_pipeline.exe                    -> GUI (default, full UI)
    data_pipeline.exe wizard             -> client one-click pipeline
    data_pipeline.exe pipeline ...       -> data_pipeline.pipeline_full.main
    data_pipeline.exe frames ...         -> data_pipeline.stages.frames.main
    data_pipeline.exe georef ...      -> data_pipeline.stages.georef.main
    data_pipeline.exe doctor             -> print resolved external tools
    data_pipeline.exe viewers compare    -> all-smoothers comparison HTML
    data_pipeline.exe viewers quality    -> ns + speed + sigma + Q panel
    data_pipeline.exe viewers diff       -> raw vs cleaned per-epoch diff
    data_pipeline.exe viewers rtkplot    -> launch bundled rtkplot.exe

The launcher lives at the repo root (next to ``data_pipeline.spec``)
because PyInstaller treats the spec's first ``scripts`` arg as a
standalone file with no ``__package__`` context.
"""

from __future__ import annotations

import sys


_SUBCOMMANDS = {"pipeline", "frames", "georef", "doctor",
                "wizard", "viewers", "--help", "-h"}


def _route() -> int:
    # PyInstaller bundle masquerades as ``python`` for any .py script
    # passed as argv[1] — needed so data_pipeline.stages.rinex can
    # ``subprocess.run([sys.executable, vendored_gnsslogger_to_rnx.py,
    # ...])`` and have the bundled python runtime execute it.
    if len(sys.argv) >= 2 and sys.argv[1].lower().endswith(".py"):
        import os
        import runpy
        script = sys.argv[1]
        sys.argv = [script] + sys.argv[2:]
        # Mirror plain-python behaviour: prepend the script's directory to
        # sys.path so its sibling modules (vendored android_rinex uses
        # ``import gnsslogger`` + ``import rinex3``) resolve. PyInstaller
        # otherwise ships a curated sys.path that excludes the cwd.
        script_dir = os.path.dirname(os.path.abspath(script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        runpy.run_path(script, run_name="__main__")
        return 0
    if len(sys.argv) >= 2 and sys.argv[1] in _SUBCOMMANDS:
        cmd = sys.argv[1]
        # Drop the subcommand from argv so the wrapped CLI sees a clean argv.
        sys.argv = [f"data_pipeline {cmd}"] + sys.argv[2:]
        if cmd in {"--help", "-h"}:
            print(__doc__)
            return 0
        if cmd == "doctor":
            from data_pipeline.lab_tools import report
            for name, path in report().items():
                print(f"{name:12s} {path}")
            return 0
        if cmd == "pipeline":
            from data_pipeline.pipeline_full import main as cli_main
            return cli_main()
        if cmd == "frames":
            from data_pipeline.stages.frames import main as cli_main
            return cli_main()
        if cmd == "georef":
            from data_pipeline.stages.georef import main as cli_main
            return cli_main()
        if cmd == "wizard":
            from data_pipeline.wizard import main as wiz_main
            return wiz_main()
        if cmd == "viewers":
            from data_pipeline.client_viewers import main as v_main
            return v_main()
    from data_pipeline.gui import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(_route())
