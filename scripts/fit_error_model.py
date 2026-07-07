r"""Fit the empirical measurement-error model (Wave 4 of the accuracy
program) from cross-device residuals.

For each of the 3 day14 Post-processing A/B pairs we have two independent The external solver
solutions of the *same drive*, from two different devices (``app.pos`` /
``logger.pos``). Neither is ground truth, but their disagreement at
matched epochs is a direct, honest estimate of the horizontal error each
solution actually carries — which is exactly what ``ErrorModel`` needs to
fit a per-(quality, ns-bucket) sigma.

Per pair:
  1. Parse both ``.pos`` files with ``parse_rtkpos``.
  2. Time-match rows by whole-second ``utc_s`` (both are 1 Hz).
  3. Remove the pair's constant lever-arm/sensor head offset by subtracting the
     median east/north offset between the two paths (the devices are
     mounted at different points on the vehicle, so a constant baseline
     offset is expected and is not "error" — only the *residual* after
     removing it reflects measurement noise).
  4. For each matched app-row, record (app_row, horizontal_residual_m)
     after offset removal — the app row's own (quality, ns) bin is what
     ``ErrorModel`` bins on, and its horizontal disagreement with the
     logger (offset-corrected) is treated as its empirical error sample.

The samples from all 3 pairs are pooled and fit with ``fit_error_model``.
Prints the per-bin sigma table + global sigma, and saves the fit to
``docs/findings/error_model.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_pipeline.error_model import fit_error_model, _ns_bucket
from data_pipeline.geo import llh_to_ecef, ecef_to_enu
from data_pipeline.parsers import PosRow, parse_rtkpos

DAY14_AB = Path(r"C:/Aj/gps/day14/ppk_ab_2026-06-28")
PAIRS = ["pair1_190336", "pair2_202751", "pair3_205044"]
OUT_JSON = REPO / "docs" / "findings" / "error_model.json"

QNAME = {1: "Fix", 2: "Float", 3: "SBAS", 4: "DGPS", 5: "Single", 6: "PPP"}


def _match_by_second(a: list[PosRow], b: list[PosRow]) -> list[tuple[PosRow, PosRow]]:
    """Pair rows from two 1 Hz .pos streams keyed on whole-second utc_s."""
    b_by_sec = {round(r.utc_s): r for r in b}
    out = []
    for ra in a:
        rb = b_by_sec.get(round(ra.utc_s))
        if rb is not None:
            out.append((ra, rb))
    return out


def _enu_horizontal(ref_llh: tuple[float, float, float], llh: tuple[float, float, float]) -> tuple[float, float]:
    """East, North (m) of ``llh`` relative to ``ref_llh``."""
    x, y, z = llh_to_ecef(*llh)
    e, n, _u = ecef_to_enu(x, y, z, ref_llh)
    return e, n


def pair_residual_samples(app_rows: list[PosRow], logger_rows: list[PosRow]) -> list[tuple[PosRow, float]]:
    """Return [(app_row, horizontal_residual_m), ...] for one A/B pair.

    Residual = |app - logger| in the local Local-frame sample, after removing the
    pair's constant east/north offset (median over all matched epochs).
    """
    matched = _match_by_second(app_rows, logger_rows)
    if not matched:
        return []

    ref_llh = (matched[0][0].lat_deg, matched[0][0].lon_deg, matched[0][0].h_m)

    raw_en: list[tuple[float, float]] = []
    for ra, rb in matched:
        ea, na = _enu_horizontal(ref_llh, (ra.lat_deg, ra.lon_deg, ra.h_m))
        eb, nb = _enu_horizontal(ref_llh, (rb.lat_deg, rb.lon_deg, rb.h_m))
        raw_en.append((ea - eb, na - nb))

    de_sorted = sorted(d[0] for d in raw_en)
    dn_sorted = sorted(d[1] for d in raw_en)
    med_de = de_sorted[len(de_sorted) // 2]
    med_dn = dn_sorted[len(dn_sorted) // 2]

    samples: list[tuple[PosRow, float]] = []
    for (ra, _rb), (de, dn) in zip(matched, raw_en):
        res_e = de - med_de
        res_n = dn - med_dn
        horiz = (res_e ** 2 + res_n ** 2) ** 0.5
        samples.append((ra, horiz))
    return samples


def main() -> int:
    all_samples: list[tuple[PosRow, float]] = []
    for name in PAIRS:
        pdir = DAY14_AB / name
        app_pos = pdir / "app.pos"
        logger_pos = pdir / "logger.pos"
        if not app_pos.exists() or not logger_pos.exists():
            print(f"skip {name}: missing app.pos/logger.pos under {pdir}")
            continue
        app_rows = parse_rtkpos(app_pos)
        logger_rows = parse_rtkpos(logger_pos)
        samples = pair_residual_samples(app_rows, logger_rows)
        print(f"{name}: {len(app_rows)} app rows, {len(logger_rows)} logger rows, "
              f"{len(samples)} matched residual samples")
        all_samples.extend(samples)

    if not all_samples:
        print("no residual samples collected — check DAY14_AB path")
        return 1

    model = fit_error_model(all_samples)

    # Per-bin sample counts, for context alongside the fitted sigma.
    counts: dict[tuple, int] = {}
    for row, _err in all_samples:
        q = int(getattr(row, "quality", 0))
        nsb = _ns_bucket(int(getattr(row, "ns", 0) or 0))
        counts[(q, nsb)] = counts.get((q, nsb), 0) + 1

    print(f"\n{len(all_samples)} total residual samples from {len(PAIRS)} pairs")
    print(f"\n{'quality':>8} {'ns_bucket':>10} {'n':>6} {'sigma_m':>10}")
    for (q, nsb), n in sorted(counts.items()):
        sigma = model.bins.get((q, nsb))
        fit_tag = "" if sigma is not None else "  (n<5, not fit -> global fallback)"
        sigma_disp = sigma if sigma is not None else model.global_sigma
        qname = QNAME.get(q, str(q))
        print(f"{q:>3}({qname:<6}) {nsb:>10} {n:>6} {sigma_disp:>10.4f}{fit_tag}")
    print(f"\n{'global':>19} {len(all_samples):>6} {model.global_sigma:>10.4f}")

    out = {
        "n_samples": len(all_samples),
        "pairs": PAIRS,
        "global_sigma_m": model.global_sigma,
        "bins": {
            f"q{q}_{nsb}": {"sigma_m": sigma, "n": counts.get((q, nsb), 0)}
            for (q, nsb), sigma in model.bins.items()
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
