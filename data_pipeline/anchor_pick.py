"""Motion model-arbitrated Post-processing + FLP-bent anchor selection.

Problem
-------
Post-processing has environment noise spikes. FLP-bent (device-Reference warped onto Post-processing anchors)
fixes those spikes — but loses local detail when device Reference itself is
off (residual to Post-processing > reject_k · sigma → anchor downweighted → bend
collapses to raw device Reference). On reference session around (-103, +424) Local-frame, Post-processing
is 1.6-2.5 m off GT but FLP-bent is 6-9 m off, and ``trust`` < 0.3.

Solution
--------
At every epoch, we have TWO candidate positions for that Post-processing timestamp:

    z_ppk  : the raw Post-processing row
    z_flp  : the FLP-bent output sampled at the same timestamp

We arbitrate between them using the Motion model cumulative-position prediction
from the previous accepted anchor. Whichever candidate agrees more
closely with the Motion model-extrapolated position is preferred (closer-fit
weight = exp(-(d/sigma)²/2)). Output is the weighted average.

This gives Post-processing precedence when device Reference drifts (because Motion model path
agrees with Post-processing), and FLP-bent precedence when Post-processing spikes (because the
Motion model path agrees with device Reference / FLP-bent).

The resulting per-epoch hybrid anchor can then be fed back into
:func:`vio_trajectory.fit_vio_anchored_trajectory` as the Post-processing input.

API
---
``arbitrate_anchors(pos_rows, flp_lat, flp_lon, flp_h, vio_vels)``
returns a new ``list[PosRow]`` at the original Post-processing timestamps with the
chosen / averaged positions.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from typing import Optional, Sequence

import numpy as np

from .geo import ecef_to_enu, llh_to_ecef
from .parsers import PosRow


def _cum_vio(vio_vels, max_gap_s=5.0):
    vio_t = np.asarray([t for t, _ in vio_vels], dtype=np.float64)
    vio_v = np.asarray([v for _, v in vio_vels], dtype=np.float64)
    dt = np.diff(vio_t, prepend=vio_t[0])
    dt[dt > max_gap_s] = 0.0
    cum = np.cumsum(vio_v * dt[:, None], axis=0)
    return vio_t, cum


def _interp_cum(vio_t, cum, t):
    j = int(np.searchsorted(vio_t, t))
    if j == 0:
        return cum[0].copy()
    if j >= len(vio_t):
        return cum[-1].copy()
    t0, t1 = float(vio_t[j - 1]), float(vio_t[j])
    if t1 <= t0:
        return cum[j - 1].copy()
    a = (t - t0) / (t1 - t0)
    return cum[j - 1] + a * (cum[j] - cum[j - 1])


def arbitrate_anchors(
    pos_rows: Sequence[PosRow],
    flp_lat: Sequence[float],
    flp_lon: Sequence[float],
    flp_h: Sequence[float],
    vio_vels: Sequence[tuple[float, np.ndarray]],
    sigma_ref_m: float = 5.0,
    log: Optional[object] = None,
) -> tuple[list[PosRow], dict]:
    """Pick / average Post-processing and FLP-bent per epoch using Motion model arbitration.

    Returns
    -------
    rows
        New :class:`PosRow` list with the same timestamps as
        ``pos_rows``. ``lat_deg / lon_deg / h_m`` are the
        Motion model-weighted average. Other fields are copied from Post-processing.
    stats : dict
        Counts of where Post-processing won vs FLP-bent vs ~equal.
    """
    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    # Input validation
    if not (math.isfinite(sigma_ref_m) and sigma_ref_m > 0):
        raise ValueError(
            f"arbitrate_anchors: sigma_ref_m must be > 0 (got {sigma_ref_m}). "
            "Typical values 1-10 m; 5.0 is the tuned default for reference session."
        )
    n = len(pos_rows)
    if len(flp_lat) != n or len(flp_lon) != n or len(flp_h) != n:
        raise ValueError(
            f"arbitrate_anchors: length mismatch — pos_rows={n}, "
            f"flp_lat={len(flp_lat)}, flp_lon={len(flp_lon)}, "
            f"flp_h={len(flp_h)}. All must match (one FLP value per PPK epoch)."
        )
    if n == 0 or not vio_vels:
        return list(pos_rows), {"n": 0}

    ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)

    # Local-frame coords of both candidate streams.
    ppk_enu = np.array([
        ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref)
        for r in pos_rows
    ], dtype=np.float64)
    flp_enu = np.array([
        ecef_to_enu(*llh_to_ecef(la, lo, hh), ref) if (math.isfinite(la) and math.isfinite(lo)) else (np.nan, np.nan, np.nan)
        for la, lo, hh in zip(flp_lat, flp_lon, flp_h)
    ], dtype=np.float64)

    vio_t, cum_vio = _cum_vio(vio_vels)

    rows: list[PosRow] = []
    last_chosen_enu: Optional[np.ndarray] = None
    last_chosen_t: Optional[float] = None
    n_ppk_won = 0
    n_flp_won = 0
    n_tied = 0
    n_no_arb = 0

    # Output position EN0(E,N,U); convert back to LLH per epoch.
    from .geo import _A, _E2
    rlat = math.radians(ref[0]); rlon = math.radians(ref[1])
    sl, cl = math.sin(rlat), math.cos(rlat)
    so, co = math.sin(rlon), math.cos(rlon)
    ref_ecef = np.array(llh_to_ecef(*ref))

    def _enu_to_llh(e, n, u):
        dx = -so*e - sl*co*n + cl*co*u
        dy = co*e - sl*so*n + cl*so*u
        dz = cl*n + sl*u
        x = ref_ecef[0]+dx; y = ref_ecef[1]+dy; z = ref_ecef[2]+dz
        p = math.sqrt(x*x+y*y); lon = math.atan2(y, x)
        lat = math.atan2(z, p*(1.0-_E2))
        for _ in range(5):
            sl_ = math.sin(lat)
            nN = _A / math.sqrt(1.0-_E2*sl_*sl_)
            lat = math.atan2(z+_E2*nN*sl_, p)
        sl_ = math.sin(lat)
        nN = _A / math.sqrt(1.0-_E2*sl_*sl_)
        h = (p/math.cos(lat)-nN) if abs(math.cos(lat))>1e-9 else (abs(z)/sl_-nN*(1.0-_E2))
        return math.degrees(lat), math.degrees(lon), h

    for i, pr in enumerate(pos_rows):
        z_ppk = ppk_enu[i]
        z_flp = flp_enu[i]
        flp_ok = math.isfinite(z_flp[0])

        if last_chosen_enu is None or not flp_ok:
            # Bootstrap: no prior reference. Default to Post-processing.
            chosen = z_ppk
            n_no_arb += 1
        else:
            # Predict expected position from last anchor + Motion model integration.
            cum_now = _interp_cum(vio_t, cum_vio, pr.utc_s)
            cum_prev = _interp_cum(vio_t, cum_vio, last_chosen_t)
            vio_delta = cum_now - cum_prev
            expected = last_chosen_enu + vio_delta

            # Disagreement of each candidate vs Motion model prediction.
            d_ppk = float(np.linalg.norm(z_ppk[:2] - expected[:2]))
            d_flp = float(np.linalg.norm(z_flp[:2] - expected[:2]))

            # Gaussian weights.
            w_ppk = math.exp(-(d_ppk / sigma_ref_m) ** 2 / 2.0)
            w_flp = math.exp(-(d_flp / sigma_ref_m) ** 2 / 2.0)
            tot = w_ppk + w_flp
            if tot < 1e-9:
                chosen = 0.5 * (z_ppk + z_flp)
                n_tied += 1
            else:
                chosen = (w_ppk * z_ppk + w_flp * z_flp) / tot
                # Track which "won" the weighting.
                if w_ppk > 1.5 * w_flp:
                    n_ppk_won += 1
                elif w_flp > 1.5 * w_ppk:
                    n_flp_won += 1
                else:
                    n_tied += 1

        lat, lon, h = _enu_to_llh(*chosen)
        rows.append(PosRow(
            utc_s=pr.utc_s, lat_deg=lat, lon_deg=lon, h_m=h,
            quality=pr.quality, vn=pr.vn, ve=pr.ve, vu=pr.vu, ns=pr.ns,
            sd_n=pr.sd_n, sd_e=pr.sd_e, sd_u=pr.sd_u,
        ))
        last_chosen_enu = chosen
        last_chosen_t = pr.utc_s

    stats = {
        "n": n, "n_ppk_won": n_ppk_won, "n_flp_won": n_flp_won,
        "n_tied": n_tied, "n_no_arb": n_no_arb,
    }
    _log(f"[arb] PPK won {n_ppk_won} ({100*n_ppk_won/n:.1f}%)  "
         f"FLP won {n_flp_won} ({100*n_flp_won/n:.1f}%)  "
         f"tied {n_tied} ({100*n_tied/n:.1f}%)  "
         f"bootstrap {n_no_arb}")
    return rows, stats
