"""Build the sync player + assert motion sensor-trust on a real day12 (old) and day14 (new) session."""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_pipeline.pipeline import RawInputs
from data_pipeline.stages import frames, viewers

CASES = [
    # (label, raw_dir, pos_file, expect_vel_source)  expect None = accept any.
    # day14 = new format (13-col sensors, boottime, Rate-signal pos).
    ("day14", Path(r"C:/Aj/gps/day14/s21/20241104_083229_651"),
     Path(r"C:/Aj/gps/day14/solved_2026-06-28/s21/20241104_083229_651/rover.pos"), "doppler"),
    # day12 = OLD format (legacy 7-col sensors + video_ns timing, no capture_meta).
    # Its solved rover.pos is a normal 24-col The external solver solve (has velocity), so the
    # panel takes the rate-signal path; the coords fallback is covered by unit tests.
    # The point here: a real old-format session renders the panel end-to-end.
    ("day12", Path(r"C:/Aj/gps/DAY12/dodge1/20260505_152247_472"),
     Path(r"C:/Aj/gps/DAY12/phone_to_external/_e2e_test/rover.pos"), None),
]


def run_case(label, raw_dir, pos, expect, out_root):
    raw = RawInputs.from_folder(raw_dir)
    out = out_root / label
    out.mkdir(parents=True, exist_ok=True)
    fr = frames.run(video=raw.recording_mp4, out_dir=out, fps=1.0, fmt="png", log=print)
    res = viewers.build_sync_player(
        video=raw.recording_mp4, pos_file=pos, frame_times_csv=fr.frame_times_csv,
        recording_map=raw.recording_txt, out_html=out / "sync_player.html",
        sensors_txt=raw.sensors_txt, data_log=raw.measurements_txt,
        wav=None, audio_anchor=None, show_spectrogram=False,
        capture_meta=raw.capture_meta_json, video_anchor=raw.video_anchor_txt, log=print)
    it = res.imu_trust
    print(f"[{label}] imu_trust={it}")
    assert it is not None, f"{label}: no imu_trust"
    if expect is not None:
        assert it["flags"]["vel_source"] == expect, \
            f"{label}: vel_source {it['flags']['vel_source']} != {expect}"
    assert "verdict" in it["flags"], f"{label}: no verdict"
    print(f"[{label}] OK  vel_source={it['flags']['vel_source']} verdict={it['flags']['verdict']}"
          f"  ->  {out/'sync_player.html'}")
    return out / "sync_player.html"


def main():
    out_root = Path(tempfile.gettempdir()) / "imu_trust_verify"
    for label, raw_dir, pos, expect in CASES:
        if not raw_dir.is_dir() or not pos.is_file():
            print(f"[{label}] SKIP (missing {raw_dir if not raw_dir.is_dir() else pos})")
            continue
        run_case(label, raw_dir, pos, expect, out_root)
    print("ALL CASES DONE")


if __name__ == "__main__":
    raise SystemExit(main())
