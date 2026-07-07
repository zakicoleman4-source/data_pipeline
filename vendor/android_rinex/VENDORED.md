# Vendored copy of `android_rinex`

These three files (and the upstream LICENSE/README) are a verbatim copy of
an upstream open-source conversion project, vendored here so this repo is
fully self-contained for offline use.

* `src/gnsslogger.py`
* `src/gnsslogger_to_rnx.py`
* `src/rinex3.py`

License: BSD-2-Clause &mdash; see `LICENSE`. Copyright (c) 2017, Rokubun.

When `data_pipeline.stages.rinex` cannot find a user-supplied
`android_rinex/src` directory, it falls back to this vendored copy so a
fresh clone of this repository works end-to-end without any external
download.

To upgrade the vendored copy, replace the three files above with the latest
versions from upstream and update this notice with the source commit.
