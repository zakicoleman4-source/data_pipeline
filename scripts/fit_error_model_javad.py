r"""Refit the empirical measurement-error model (Wave 4 follow-up) against
The reference unit ground truth instead of cross-device residuals.

Motivation
----------
``scripts/fit_error_model.py`` fits ``ErrorModel`` from *cross-device*
residuals: two independent devices Post-processing-solving the same drive against the
same base. That is common-mode-BLIND to a float-ambiguity bias that both
devices share (same base, same config, same ambiguity-resolution behaviour)
-- both devices can agree with each other while both are ~2 m off from the
truth. The resulting q2 (float) sigma was a suspiciously tight 0.070 m.

This script instead fits against the reference unit survey-grade GT log, which
*does* see the true position and therefore the true float bias. If the
q2/float bins come out with materially larger sigma here than in the
cross-device fit, that is direct evidence the cross-device fit was blind to
a real error source.

Method
------
For each of the 3 day14 "dodge" sessions with The reference unit overlap:
  1. Parse ``rover.pos`` (device Post-processing solution) and the shared The reference unit
     ``gt_log0628a.pos`` with ``parse_rtkpos``.
  2. Time-match rows by whole-second ``utc_s`` (both ~1 Hz).
  3. Remove the session's constant east/north offset (median over all
     matched epochs) -- the reference unit GT has its own sensor head/datum reference,
     so a constant baseline offset between subject and The reference unit is a lever-arm
     / reference artefact, not "error". Only the *residual after* removing
     that constant offset reflects genuine epoch-varying measurement noise
     -- exactly the quantity a per-epoch sigma should predict. (NOTE: if
     the device's error is itself a *constant* bias, offset-removal will
     absorb it into the median and hide it from the fit -- see the
     printed comparison and findings doc for how this plays out.)
  4. For each matched subject row, record (rover_row, horizontal_residual_m)
     -- the row's own (quality, ns) bin is what ``ErrorModel`` bins on.

Samples from all 3 sessions are pooled and fit with ``fit_error_model``.
Prints the per-bin table, compares it against the existing cross-device fit
in ``docs/findings/error_model.json``, and saves this fit to
``docs/findings/error_model_javad.json``.

Also runs a catch-rate check on dodge190336 (see ``catch_rate`` below):
of the epochs whose true residual vs The reference unit exceeds 1 m, what fraction does
the reference unit-fit ``ErrorModel`` flag as uncertain (predicted 2-sigma > 1 m)?
Compared against the same check using the cross-device-fit model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_pipeline.error_model import ErrorModel, fit_error_model, _ns_bucket
from data_pipeline.geo import llh_to_ecef, ecef_to_enu
from data_pipeline.parsers import PosRow, parse_rtkpos

DODGE_ROOT = Path(r"C:/Aj/gps/day14/solved_2026-06-28/dodge")
SESSIONS = ["20260628_190336_677", "20260628_202751_411", "20260628_205044_886"]
GT_POS = Path(r"C:/Aj/gps/day14/solved_2026-06-28/gt/gt_log0628a.pos")

OUT_JSON = REPO / "docs" / "findings" / "error_model_javad.json"
CROSS_DEVICE_JSON = REPO / "docs" / "findings" / "error_model.json"

CATCH_RATE_SESSION = "20260628_190336_677"
CATCH_RATE_THRESHOLD_M = 1.0

QNAME = {1: "Fix", 2: "Float", 3: "SBAS", 4: "DGPS", 5: "Single", 6: "PPP"}


def _match_by_second(a: list[PosRow], b: list[PosRow]) -> list[tuple[PosRow, PosRow]]:
    """Pair rows from two ~1 Hz .pos streams keyed on whole-second utc_s."""
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


def session_residual_samples(
    rover_rows: list[PosRow], gt_rows: list[PosRow]
) -> tuple[list[tuple[PosRow, float]], list[tuple[PosRow, PosRow, float]]]:
    """Return ([(rover_row, horizontal_residual_m), ...], matched_detail).

    Residual = |subject - the reference unit| in the local Local-frame sample, after removing the
    session's constant east/north offset (median over all matched epochs).
    ``matched_detail`` also carries the gt row and the pre-offset-removal
    residual, for the catch-rate check (which needs the *raw* vs-The reference unit
    residual, not the offset-corrected one -- see main()/catch_rate()).
    """
    matched = _match_by_second(rover_rows, gt_rows)
    if not matched:
        return [], []

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
    detail: list[tuple[PosRow, PosRow, float]] = []
    for (ra, rb), (de, dn) in zip(matched, raw_en):
        res_e = de - med_de
        res_n = dn - med_dn
        horiz = (res_e ** 2 + res_n ** 2) ** 0.5
        raw_horiz = (de ** 2 + dn ** 2) ** 0.5
        samples.append((ra, horiz))
        detail.append((ra, rb, raw_horiz))
    return samples, detail


def catch_rate(
    model: ErrorModel,
    detail: list[tuple[PosRow, PosRow, float]],
    threshold_m: float,
    use_raw_residual: bool,
) -> tuple[int, int, float]:
    """Of the epochs whose true residual vs The reference unit exceeds ``threshold_m``,
    what fraction have predicted 2*sigma_h(row) > threshold_m (i.e. the
    model flags them as uncertain)?

    ``use_raw_residual``: the offset-removed residual (what the model was
    fit on) still contains a constant per-session offset if True is passed
    as False... in practice we always use the RAW vs-The reference unit residual here,
    because that is the actual real-world error a downstream consumer
    cares about (the session's own datum offset is not observable at
    inference time -- there is no The reference unit to subtract against in production).
    Kept as a parameter for clarity / experimentation.
    """
    flagged = 0
    exceeding = 0
    for row, _gt, raw_horiz in detail:
        if raw_horiz > threshold_m:
            exceeding += 1
            pred_2sigma = 2.0 * model.sigma_h(row)
            if pred_2sigma > threshold_m:
                flagged += 1
    rate = (flagged / exceeding) if exceeding else float("nan")
    return flagged, exceeding, rate


def main() -> int:
    if not GT_POS.exists():
        print(f"Javad GT file not found: {GT_POS}")
        return 1

    gt_rows = parse_rtkpos(GT_POS)
    print(f"Javad GT: {len(gt_rows)} rows from {GT_POS}")

    all_samples: list[tuple[PosRow, float]] = []
    per_session_detail: dict[str, list[tuple[PosRow, PosRow, float]]] = {}

    for name in SESSIONS:
        rover_pos = DODGE_ROOT / name / "rover.pos"
        if not rover_pos.exists():
            print(f"skip {name}: missing rover.pos under {DODGE_ROOT / name}")
            continue
        rover_rows = parse_rtkpos(rover_pos)
        samples, detail = session_residual_samples(rover_rows, gt_rows)
        print(f"{name}: {len(rover_rows)} rover rows, {len(samples)} matched vs Javad")
        all_samples.extend(samples)
        per_session_detail[name] = detail

    if not all_samples:
        print("no residual samples collected -- check DODGE_ROOT / GT_POS paths")
        return 1

    model = fit_error_model(all_samples)

    # Per-bin sample counts, for context alongside the fitted sigma.
    counts: dict[tuple, int] = {}
    for row, _err in all_samples:
        q = int(getattr(row, "quality", 0))
        nsb = _ns_bucket(int(getattr(row, "ns", 0) or 0))
        counts[(q, nsb)] = counts.get((q, nsb), 0) + 1

    print(f"\n{len(all_samples)} total residual samples from {len(SESSIONS)} dodge sessions (vs Javad)")
    print(f"\n{'quality':>12} {'ns_bucket':>10} {'n':>6} {'sigma_m':>10}")
    for (q, nsb), n in sorted(counts.items()):
        sigma = model.bins.get((q, nsb))
        fit_tag = "" if sigma is not None else "  (n<5, not fit -> global fallback)"
        sigma_disp = sigma if sigma is not None else model.global_sigma
        qname = QNAME.get(q, str(q))
        print(f"{q:>3}({qname:<6}) {nsb:>10} {n:>6} {sigma_disp:>10.4f}{fit_tag}")
    print(f"\n{'global':>23} {len(all_samples):>6} {model.global_sigma:>10.4f}")

    # --- Compare against the cross-device fit -------------------------------
    cross_device_bins: dict[str, dict] = {}
    if CROSS_DEVICE_JSON.exists():
        cross_device_bins = json.loads(CROSS_DEVICE_JSON.read_text()).get("bins", {})

    print("\n--- Javad-fit vs cross-device-fit sigma, per bin ---")
    print(f"{'bin':>10} {'cross_device_m':>15} {'javad_m':>12} {'ratio':>8}")
    all_bin_keys = sorted(
        set(cross_device_bins.keys())
        | {f"q{q}_{nsb}" for (q, nsb) in model.bins.keys()}
    )
    for key in all_bin_keys:
        cp_sigma = cross_device_bins.get(key, {}).get("sigma_m")
        q_str, nsb = key.split("_", 1)
        q = int(q_str[1:])
        jv_sigma = model.bins.get((q, nsb))
        cp_disp = f"{cp_sigma:.4f}" if cp_sigma is not None else "n/a"
        jv_disp = f"{jv_sigma:.4f}" if jv_sigma is not None else "n/a"
        ratio_disp = f"{(jv_sigma / cp_sigma):.2f}x" if (cp_sigma and jv_sigma) else "n/a"
        print(f"{key:>10} {cp_disp:>15} {jv_disp:>12} {ratio_disp:>8}")

    q2_hi_cp = cross_device_bins.get("q2_hi", {}).get("sigma_m")
    q2_hi_jv = model.bins.get((2, "hi"))
    if q2_hi_cp and q2_hi_jv:
        verdict = "LARGER" if q2_hi_jv > q2_hi_cp else "NOT larger"
        print(
            f"\nq2_hi (float, high-ns): Javad-fit sigma ({q2_hi_jv:.4f} m) is "
            f"{verdict} than cross-device-fit sigma ({q2_hi_cp:.4f} m)."
        )

    # --- Save the fit --------------------------------------------------
    out = {
        "n_samples": len(all_samples),
        "sessions": SESSIONS,
        "gt_source": str(GT_POS),
        "global_sigma_m": model.global_sigma,
        "bins": {
            f"q{q}_{nsb}": {"sigma_m": sigma, "n": counts.get((q, nsb), 0)}
            for (q, nsb), sigma in model.bins.items()
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # --- Catch-rate check (STEP 2) --------------------------------------
    detail = per_session_detail.get(CATCH_RATE_SESSION)
    if detail:
        flagged_jv, exceeding, rate_jv = catch_rate(
            model, detail, CATCH_RATE_THRESHOLD_M, use_raw_residual=True
        )
        print(
            f"\n--- Catch-rate check on {CATCH_RATE_SESSION} "
            f"(threshold={CATCH_RATE_THRESHOLD_M} m) ---"
        )
        print(
            f"Epochs with true residual vs Javad > {CATCH_RATE_THRESHOLD_M} m: {exceeding}"
        )
        print(
            f"Javad-fit model: flagged (predicted 2-sigma > {CATCH_RATE_THRESHOLD_M} m) "
            f"= {flagged_jv}/{exceeding}  catch-rate = {rate_jv:.1%}"
            if exceeding else "no epochs exceed threshold"
        )

        # Cross-device-fit model, for comparison, loaded from its own JSON.
        if CROSS_DEVICE_JSON.exists():
            cp_data = json.loads(CROSS_DEVICE_JSON.read_text())
            cp_bins: dict[tuple, float] = {}
            for k, v in cp_data.get("bins", {}).items():
                q_str, nsb = k.split("_", 1)
                cp_bins[(int(q_str[1:]), nsb)] = v["sigma_m"]
            cp_model = ErrorModel(bins=cp_bins, global_sigma=cp_data.get("global_sigma_m", 1.0))
            flagged_cp, exceeding_cp, rate_cp = catch_rate(
                cp_model, detail, CATCH_RATE_THRESHOLD_M, use_raw_residual=True
            )
            print(
                f"Cross-device-fit model: flagged = {flagged_cp}/{exceeding_cp}  "
                f"catch-rate = {rate_cp:.1%}"
                if exceeding_cp else "no epochs exceed threshold"
            )
            print(
                f"\nCatch-rate: cross-device-fit {rate_cp:.1%} -> Javad-fit {rate_jv:.1%}"
            )
    else:
        print(f"\nno detail collected for catch-rate session {CATCH_RATE_SESSION}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
