# correct_relative_times.py — Standalone time-anchor corrector

Map a list of **relative-time floats** (e.g. stream-event pick times from a
`recording_*.wav`) into **UTC** and **Reference time** for a source-app session,
using all anchors in `recording_*.txt` to absorb writer jitter and clock
drift.

Outputs (every run emits **two variants**, see §6):
1. **`<times>.corrected.robust.csv`** + **`drift_report.robust.html`** —
   fit uses MAD-based outlier rejection. Recommended for production.
2. **`<times>.corrected.raw.csv`** + **`drift_report.raw.html`** — fit
   uses every anchor, no rejection. Use for "what would the answer be if
   I trusted every byte the device wrote." Diagnostic / forensic.

The script is fully self-contained (no `data_pipeline` import). It
inlines the OLS time-anchor fit, the Reference-UTC epoch offset table, and the
HTML report. Drop it on a machine with Python 3.10+, NumPy, and
optionally a copy of `plotly.min.js` for the report.

---

## 1. Install

```
pip install numpy
```

That's all the runtime needs. For the HTML report to render plots
**offline**, place a `plotly.min.js` either:
- next to `correct_relative_times.py`, or
- at `<repo>/data_pipeline/assets/plotly.min.js` (auto-detected when the
  script lives in `<repo>/scripts/`).

If neither path exists the HTML still writes; you just need an internet
connection (or copy plotly.min.js next to the HTML) to view it.

---

## 2. Inputs

### Session folder

A folder containing (e.g. `the reference session/`):

```
measurements_<ts>.txt
recording_<ts>.txt       # the anchor file — REQUIRED
recording_<ts>.mp4       # the media file — REQUIRED (for the basename in CSV)
sensors_<ts>.txt
```

`recording_*.txt` is the source app anchor file. Every line:
```
<video_ns>,<UTC_ISO>,<UTC_ISO>
```
Each row is the device's best-effort log of `(source clock, system clock)`
at the moment The platform flushed the writer. Typical session has 2-5k rows.

### Times file

The list of relative-time floats (seconds since session start).
Two accepted layouts:

**Plain text (one float per line, blank / `#`-comments skipped):**
```
# my pickings
0.0
1.5
10.25
30.0
```

**CSV with a recognised header:** the script will read whichever column
matches one of `relative_time_s`, `t_s`, `t_rel_s`, `t_video_s`,
`time_s`, `seconds`, or `pick_s`. Other columns are ignored.

```
label,relative_time_s
clap_start,0.0
mark_a,5.123
end,180.75
```

If no header is found, the first column is used.

---

## 3. Run

```
python correct_relative_times.py <session_dir> --times <floats_file>
```

### Optional flags

| Flag | Meaning |
|------|---------|
| `--out <csv>` | Override output CSV path. Default: `<times>.corrected.csv`. |
| `--html <html>` | Override HTML path. Default: `<session>/drift_report.html`. |
| `--offset-s <s>` | Set `t_video = t_relative + offset_s`. Use when wav and container file do **not** share start time (e.g. wav started 0.05 s earlier → pass `--offset-s -0.05`). Default 0. |
| `--roll-window-s <s>` | Window for the rolling-RMSE plot. Default 30 s. |
| `--no-html` | Skip HTML emission. |

### Quick examples

```
# Default paths (CSV next to times file, HTML next to session):
python correct_relative_times.py C:/eli_test/the reference session --times pickings.txt

# Custom output and wider rolling window:
python correct_relative_times.py C:/eli_test/the reference session ^
  --times pickings.csv ^
  --out  the reference session_corrected.csv ^
  --html the reference session_drift.html ^
  --roll-window-s 60
```

Every run emits both a **robust** and a **raw** variant — see §6. If
you passed `--out foo.csv` the files will be `foo.robust.csv` and
`foo.raw.csv`; if you passed `--html bar.html` they will be
`bar.robust.html` and `bar.raw.html`.

---

## 4. Output CSV — column reference

| Column | Meaning |
|--------|---------|
| `relative_time_s` | The pick time you provided. |
| `video_pts_s` | `relative_time_s + offset_s`. The query point on the media clock. |
| `video_start_utc_iso` | UTC at media PTS = 0 (from the OLS fit's intercept — the most-stable anchor we can produce for "when the session started"). |
| **`naive_utc_iso`** | **`video_start_utc + relative_time_s`.** What you'd guess if you trusted only the first anchor. NO drift correction. *This is the column your client asked for.* |
| `naive_utc_posix_s` | Same value as POSIX seconds (UTC). |
| **`anchored_utc_iso`** | **OLS-corrected UTC.** Uses every anchor in `recording_*.txt`. Drift absorbed. Use this for downstream work. |
| `anchored_utc_posix_s` | Same value as POSIX seconds (UTC). |
| `correction_s` | `anchored - naive`. How many seconds the regression shifted the pick. Sign tells you which way drift was leaning. |
| `anchor_uncertainty_s` | 1-sigma OLS uncertainty at the query time (closed-form prediction-of-mean variance). |
| `naive_gpst_iso` / `naive_gpst_posix_s` | Naive UTC + epoch offset. Matches The external solver `.pos` time axis without correction. |
| `anchored_gpst_iso` / `anchored_gpst_posix_s` | Anchored UTC + epoch offset. Use for `.pos` comparisons. |
| `leap_seconds` | Integer Reference-UTC offset at this epoch (18 s for 2017→present). |
| `mp4_basename` | Filename of the session's `.mp4`. |
| `session` | Session folder name. |

---

## 5. HTML report — plot reference

Six plots, two-up grid, all offline-renderable:

1. **Per-anchor residual vs media time.** Each blue dot = one
   recording_*.txt anchor: `utc_logged − utc_fit`. Orange diamonds = your
   pickings (placed at their fit uncertainty for scale). Structure in
   this plot (slope, steps, gaps) means the linear model didn't fully
   capture the source-vs-system relationship.

2. **Residual histogram.** Distribution shape of write-latency residuals.
   Roughly Gaussian = healthy; long right tail = device occasionally
   stalled the writer.

3. **Cumulative correction (anchored − naive).** Green line over full
   span shows how much the OLS shifts the naive mapping at every media
   time. Slope of this line = `drift_ppm × 1e-6`. Orange diamonds: your
   pickings — read directly how many ms each pick got corrected.

4. **Rolling RMSE.** Sliding-window RMSE of residuals. Flat = stationary
   jitter. Bumps = device under load (background sync, thermal throttle)
   when scheduling got worse.

5. **Residual Q-Q vs standard normal.** Straight line through the centre
   = Gaussian residuals. Tails curling away = heavy-tailed jitter
   (The platform scheduler outliers).

6. **Sorted |residual| vs percentile.** Direct reading of "95% of
   anchors are within X ms of the fit."

A summary table at the top of the HTML lists the headline stats:
n / n_rejected / RMSE / max / drift_ppm / sigma_hat / fit_uncertainty /
cubic improvement / span / accumulated drift / bias / P95.

---

## 6. Statistics — definitions

### TimeAnchor model

```
utc_s(video_ns) = intercept + slope * video_ns
                = ymean + slope * (video_ns - xmean)        (centred form)
```

`slope` ≈ 1e-9 s/ns. The deviation from exactly 1e-9 is the **drift**:

```
drift_ppm = (slope * 1e9 - 1) * 1e6
```

Positive ppm = source clock runs faster than system clock. Drift
accumulates linearly: over a 30-min session an 8 ppm drift =
30·60·8e-6 ≈ 14 ms of error at the end if you used only the first anchor.

### Residuals (write-latency proxy)

```
r_i      = utc_logged_i - utc_fit(video_ns_i)
RMSE     = sqrt( mean(r_i^2) )                              (biased, /n)
sigma_hat = RMSE * sqrt(n / (n - 2))                        (unbiased, /(n-2))
max_abs  = max(|r_i|)
```

RMSE captures how long The platform scheduling holds anchor writes before
they hit storage. Typical device: 15-25 ms RMSE, 50-100 ms max.

### Robust rejection — what the threshold is and why

Two fits are produced per run: **robust** and **raw**. The robust fit
removes anchors whose residual is too far from the bulk of the
distribution. Algorithm:

1. Run an initial OLS fit on every anchor.
2. Compute `MAD = median(|r_i|)` of the residuals. MAD is the
   median-absolute-deviation: a robust scale estimator that ignores
   tails (unlike standard deviation, one outlier can blow up `std` but
   barely moves `MAD`).
3. Drop any anchor with `|r_i| > 5 * MAD`. Five MADs is the threshold.
4. Re-fit on the survivors. Repeat steps 2-4 up to **3 times**, or stop
   early if the kept set didn't shrink, or would shrink below `n=2`.

**Why 5 × MAD specifically:** for a Gaussian distribution,
`sigma ≈ 1.4826 × MAD`, so `5 × MAD ≈ 3.37 × sigma`. In a clean
Gaussian write-latency distribution, fewer than 0.1 % of anchors would
ever exceed this — but The platform scheduling tails are heavier than
Gaussian, so the rejection rate in practice is 0.2 % to 8 %. The
threshold is intentionally **loose**: we want to kill only obvious
flush-stall outliers (e.g. an anchor written 300-400 ms late because
the OS paged in a new thread), not down-weight ordinary jitter. If you
need a tighter or looser threshold, edit the `mad_threshold=5.0`
default in `fit_time_anchor()`.

**Reading the difference between robust and raw:**
- `max_abs` will be much larger in raw (the worst outlier is included).
- `RMSE` will be slightly larger in raw because tails fatten the
  variance.
- `drift_ppm` should be **similar** between the two if outliers are
  symmetric; a meaningful gap (>1 ppm) suggests the outliers are
  biased toward one side and worth investigating.
- `fit_uncertainty_s` will be slightly larger in raw (sigma_hat goes up
  faster than `1/sqrt(n)` shrinks).

### Per-query uncertainty

```
var(yhat(x)) = sigma_hat^2 * ( 1/n + (x - xmean)^2 / Sxx )
              Sxx = sum( (xi - xmean)^2 )
1-sigma      = sqrt(var(yhat(x)))
```

This is `anchor_uncertainty_s` in the CSV. A U-curve: smallest at the
session midpoint (`xmean`), grows toward the edges. For a
well-distributed 5k-anchor session this stays sub-millisecond across the
whole session.

### Cubic check

A degree-3 polynomial is fit to the same anchors as a diagnostic. The
HTML reports `cubic_rmse - linear_rmse`. If this is small vs the linear
RMSE the linear model is adequate; if it's comparable, the source-vs-
system relationship has nonlinear drift you should investigate before
trusting sub-ms UTC.

### Edge-of-stream test (what your "test the stream at the edge" pickings
are doing)

If you have an external reference UTC for a clap/marker at the
beginning *and* end of the session:

```
error_anchored   = anchored_utc   - utc_gt
error_naive      = naive_utc      - utc_gt
```

Expected behaviour:
- `error_naive` should grow roughly linearly with `relative_time_s`
  (slope ≈ `drift_ppm × 1e-6`).
- `error_anchored` should be small and scatter-bounded by
  `anchor_uncertainty_s`. If it has structure, the linear model isn't
  enough.

The `correction_s` column makes this directly visible without external
reference: it's the OLS's own estimate of what the naive mapping
would have got wrong at that pick.

---

## 7. Caveats

- `recording_*.txt` is **not** reference. Each anchor row has its own
  write-latency. The OLS fit averages this out; do not trust a single
  anchor.
- Drift is assumed **linear** over the session. The cubic-improvement
  metric is the cheap sanity check; if it ever exceeds half the RMSE,
  re-examine.
- The epoch offset table is current to 2017-01-01 (last announced offset).
  Update `_LEAP_SECOND_TABLE` if a new one is announced.
- `--offset-s` is required when the stream (wav) and media (container file) do **not**
  start simultaneously. the source app normally synchronises them, but
  custom rigs may not.

---

## 8. License / contact

Internal client deliverable. Source of canonical implementation:
`data_pipeline/time_sync.py` in the data_to_frames repository.
