"""
Parse the source app sensors_*.txt files.

Columns: GPS_seconds, gx, gy, gz, ax, ay, az
  - GPS_seconds: seconds since Reference epoch (1980-01-06 00:00:00 UTC)
  - gx/gy/gz: rate sensor angular velocity (rad/s)
  - ax/ay/az: linear sensor including gravity (m/s², ~9.81 when stationary)

Usage:
    python parse_sensors.py path/to/sensors_*.txt
    python parse_sensors.py path/to/sensors_*.txt --to-csv output.csv
"""

import csv
import sys
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

GPS_EPOCH_UNIX = 315964800  # 1980-01-06 00:00:00 UTC as Unix timestamp
GPS_UTC_LEAP_SECONDS = 18  # current as of 2026


@dataclass
class ImuSample:
    utc_timestamp: float  # Unix seconds (UTC)
    gx: float             # rate sensor x (rad/s)
    gy: float             # rate sensor y (rad/s)
    gz: float             # rate sensor z (rad/s)
    ax: float             # linear sensor x (m/s²)
    ay: float             # linear sensor y (m/s²)
    az: float             # linear sensor z (m/s²)

    @property
    def utc_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.utc_timestamp, tz=timezone.utc)


def parse_sensors(path: str | Path) -> list[ImuSample]:
    """Parse a the source app sensors_*.txt file into ImuSample list."""
    samples = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                gps_s = float(parts[0])
                utc_s = gps_s + GPS_EPOCH_UNIX - GPS_UTC_LEAP_SECONDS
                samples.append(ImuSample(
                    utc_timestamp=utc_s,
                    gx=float(parts[1]),
                    gy=float(parts[2]),
                    gz=float(parts[3]),
                    ax=float(parts[4]),
                    ay=float(parts[5]),
                    az=float(parts[6]),
                ))
            except (ValueError, IndexError):
                continue
    samples.sort(key=lambda s: s.utc_timestamp)
    return samples


def to_csv(samples: list[ImuSample], out_path: str | Path) -> None:
    """Write parsed samples to CSV."""
    names = [f.name for f in fields(ImuSample)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(names + ["utc_iso"])
        for s in samples:
            w.writerow([getattr(s, n) for n in names] + [s.utc_datetime.isoformat()])


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    samples = parse_sensors(path)
    print(f"Parsed {len(samples):,} IMU samples from {path.name}")

    if samples:
        dt_start = samples[0].utc_datetime
        dt_end = samples[-1].utc_datetime
        duration = samples[-1].utc_timestamp - samples[0].utc_timestamp
        rate = (len(samples) - 1) / duration if duration > 0 else 0
        print(f"  Start:    {dt_start.strftime('%Y-%m-%d %H:%M:%S.%f')} UTC")
        print(f"  End:      {dt_end.strftime('%Y-%m-%d %H:%M:%S.%f')} UTC")
        print(f"  Duration: {duration:.1f} s ({duration/60:.1f} min)")
        print(f"  Rate:     {rate:.1f} Hz")
        print(f"\n  First sample: gx={samples[0].gx:.4f} gy={samples[0].gy:.4f} gz={samples[0].gz:.4f}"
              f"  ax={samples[0].ax:.4f} ay={samples[0].ay:.4f} az={samples[0].az:.4f}")

    if "--to-csv" in sys.argv:
        idx = sys.argv.index("--to-csv")
        out = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else path.with_suffix(".csv")
        to_csv(samples, out)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
