#!/usr/bin/env bash
# data_to_frames Unix installer wrapper.
#   - Resolves python3 on PATH
#   - Calls install.py which handles pip + gtsam-via-conda fallback
#
# Usage:
#   ./setup.sh             # required deps + try gtsam
#   ./setup.sh --dev       # + pytest + pyinstaller
#   ./setup.sh --no-gtsam  # skip gtsam entirely
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_CMD="${PYTHON_CMD:-python3}"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_CMD=python
    else
        echo "ERROR: neither python3 nor python found on PATH." >&2
        echo "Install Python 3.10+ first." >&2
        exit 1
    fi
fi

echo "=== data_to_frames installer ==="
"$PYTHON_CMD" "$SCRIPT_DIR/install.py" "$@"
rc=$?

echo
if [ $rc -ne 0 ]; then
    echo "Installer exited with code $rc. See messages above."
else
    echo "Installation OK. Launch GUI with:  $PYTHON_CMD -m data_pipeline"
fi
exit $rc
