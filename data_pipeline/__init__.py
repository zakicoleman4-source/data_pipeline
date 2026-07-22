"""data_to_georef - turn a the source app session into a Coordinate output dataset.

Submodules:
- :mod:`data_pipeline.gui`        Tkinter front-end orchestrating the four stages.
- :mod:`data_pipeline.pipeline`   Programmatic, threadable orchestration helpers.
- :mod:`data_pipeline.parsers`    Parsers for ``.pos``, the source app ``Fix`` and
                                   ``OrientationDeg`` lines, ``recording_*.txt``,
                                   and the ``extracted_frame_times.csv`` we emit.
- :mod:`data_pipeline.geo`        The standard datum / LLH / Cartesian XYZ / Local-frame conversions.
- :mod:`data_pipeline.smoothing`  Gaussian and circular Gaussian smoothing.
- :mod:`data_pipeline.time_sync`  Media-PTS <-> UTC and Reference time <-> UTC.
- :mod:`data_pipeline.stages`     The four pipeline stages (Interchange-format, samples,
                                   Coordinate output CSV, viewers).
"""

__version__ = "1.1.0"
__all__ = ["__version__"]
