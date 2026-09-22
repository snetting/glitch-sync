#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Could not find ${PYTHON_BIN}. Set PYTHON_BIN to a supported Python executable." >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating virtual environment in ${VENV_DIR}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
echo "Checking GlitchSync dependencies..."
"${VENV_PYTHON}" -m pip install --disable-pip-version-check -r "${ROOT_DIR}/requirements.txt"

echo "Launching GlitchSync ${ROOT_DIR}/glitch_sync.py..."
exec "${VENV_PYTHON}" "${ROOT_DIR}/glitch_sync.py" --gui "$@"
