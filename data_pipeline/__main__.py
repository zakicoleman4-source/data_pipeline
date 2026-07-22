"""Module entrypoint: ``python -m data_pipeline`` launches the GUI."""

from __future__ import annotations

from .gui import main


if __name__ == "__main__":
    raise SystemExit(main())
