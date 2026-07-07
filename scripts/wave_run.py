# scripts/wave_run.py
"""Durable-ledger driver for the deep-accuracy MEGA-program (token-window phasing).

Usage:
  python scripts/wave_run.py status         # print ledger + first incomplete wave
  python scripts/wave_run.py record "TEXT"  # append a timestamped line to the ledger
  python scripts/wave_run.py next            # print the first line containing 'NOT STARTED' or 'IN PROGRESS'
"""
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / ".superpowers" / "sdd" / "progress.md"


def _read():
    return LEDGER.read_text(encoding="utf-8") if LEDGER.exists() else ""


def status():
    print(_read())
    print("--- next ---")
    next_wave()


def next_wave():
    for line in _read().splitlines():
        if "NOT STARTED" in line or "IN PROGRESS" in line:
            print(line)
            return
    print("ALL WAVES COMPLETE")


def record(text):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    print("recorded:", text)


def main(argv):
    if not argv or argv[0] == "status":
        status()
    elif argv[0] == "next":
        next_wave()
    elif argv[0] == "record" and len(argv) > 1:
        record(" ".join(argv[1:]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
