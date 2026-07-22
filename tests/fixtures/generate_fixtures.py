"""Generate synthetic test fixtures for regression testing.

This script creates:
- recording_small.txt: 100 (video_ns, UTC) anchors with +3.2 ppm drift and ~30 ms jitter
- sample_small.pos: 60 Post-processing rows at 1 Hz baseline
- extracted_frame_times_small.csv: 600 extracted sample timestamps at ~6 fps
"""

import random
import math
from pathlib import Path


def generate_recording_txt(path: Path, n_anchors: int = 100, drift_ppm: float = 3.2):
    """Generate a session file with anchors."""
    random.seed(42)  # Deterministic.
    # Start at a reasonable POSIX timestamp (2024-01-01 12:00:00 UTC = 1704110400.0).
    start_utc_s = 1704110400.0
    session_duration_s = 60.0  # 60-second session.

    # Media clock: start at some reasonable value.
    start_video_ns = 1_000_000_000  # 1 billion nanoseconds.

    # Drift: +3.2 ppm means the source clock is 3.2 microseconds slower per second.
    # slope = 1 + (drift_ppm / 1e6)
    slope = 1.0 + (drift_ppm / 1e6)

    with open(path, "w") as f:
        for i in range(n_anchors):
            # Uniform distribution of anchors over the session.
            t_session = i * session_duration_s / (n_anchors - 1) if n_anchors > 1 else 0

            # Media clock with drift.
            video_ns = start_video_ns + t_session * 1e9 * slope

            # UTC with drift and jitter (~30 ms std dev).
            utc_s = start_utc_s + t_session
            jitter_s = random.gauss(0, 0.030)
            utc_s += jitter_s

            f.write(f"{video_ns:.0f},{utc_s:.6f},{utc_s:.6f}\n")


def generate_sample_pos(path: Path, n_rows: int = 60, start_utc_s: float = 1704110400.0):
    """Generate a .pos file with Post-processing rows at 1 Hz."""
    # Reference point: somewhere reasonable (e.g., near Cupertino, CA).
    lat_base = 37.3382
    lon_base = -122.0324
    h_base = 10.0  # meters.

    # Add small perturbations to simulate a moving path.
    random.seed(42)

    with open(path, "w") as f:
        # Header comment.
        f.write("% Sample RTKLIB .pos file\n")
        f.write("% Date Time Lat Lon H Q Sdn Sde Sdu Sdne Sdeu Sdun Ns Ng Age Ratio\n")

        for i in range(n_rows):
            t_utc = start_utc_s + i  # 1 Hz sampling.

            # Convert POSIX to date/time string.
            import datetime as dt
            dt_obj = dt.datetime.fromtimestamp(t_utc, dt.timezone.utc)
            date_str = dt_obj.strftime("%Y/%m/%d")
            time_str = dt_obj.strftime("%H:%M:%S.%f")[:12]  # Keep 6 digits of fractional.

            # Add small random walk to position.
            lat = lat_base + (i - n_rows // 2) * 0.00001
            lon = lon_base + (i - n_rows // 2) * 0.00001
            h = h_base + random.gauss(0, 0.1)

            q = 1  # Fix quality (1 = fixed).
            # Standard deviations (in meters).
            sdn, sde, sdu = 0.01, 0.01, 0.03
            sdne, sdeu, sdun = 0.0, 0.0, 0.0
            ns, ng = 20, 5  # Number of sources, Source-group sources.
            age, ratio = 0.0, 10.0

            # Format: lat lon h q sdn sde sdu sdne sdeu sdun ns ng age ratio
            f.write(
                f"{date_str} {time_str} {lat:.8f} {lon:.8f} {h:.4f} {q} "
                f"{sdn:.4f} {sde:.4f} {sdu:.4f} {sdne:.4f} {sdeu:.4f} {sdun:.4f} "
                f"{ns} {ng} {age:.1f} {ratio:.1f}\n"
            )


def generate_frame_times_csv(path: Path, n_frames: int = 600, fps: float = 6.0):
    """Generate sample times CSV (image_name, t_video_s)."""
    with open(path, "w") as f:
        f.write("image,t_video_s\n")
        for i in range(n_frames):
            t_video_s = i / fps  # Regular sampling at 6 fps.
            image_name = f"IMG_{i:06d}.jpg"
            f.write(f"{image_name},{t_video_s:.6f}\n")


if __name__ == "__main__":
    fixture_dir = Path(__file__).parent
    generate_recording_txt(fixture_dir / "recording_small.txt", n_anchors=100, drift_ppm=3.2)
    generate_sample_pos(fixture_dir / "sample_small.pos", n_rows=60)
    generate_frame_times_csv(fixture_dir / "extracted_frame_times_small.csv", n_frames=600)
    print(f"Generated fixtures in {fixture_dir}")
