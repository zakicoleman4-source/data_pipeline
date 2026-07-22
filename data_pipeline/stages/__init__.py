"""Pipeline stages.

Each stage exposes:

* a ``run(...)`` function with explicit, dataclass-friendly parameters
  (used by both the GUI and the CLI), and
* a ``main()`` argparse entrypoint so the stage is also runnable as
  ``python -m data_pipeline.stages.<name>``.
"""

__all__ = ["rinex", "frames", "georef", "viewers", "ppk", "t02", "adaptive_frames"]
