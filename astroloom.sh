#!/usr/bin/env bash
# Lanzador de AstroLoom para Linux y macOS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/python3" ]; then
    PYTHON_EXEC=".venv/bin/python3"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif [ -f "venv/bin/python3" ]; then
    PYTHON_EXEC="venv/bin/python3"
elif [ -f "venv/bin/python" ]; then
    PYTHON_EXEC="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXEC="python"
else
    echo "[ERROR] No se encontró Python 3 instalado en el sistema."
    exit 1
fi

exec "$PYTHON_EXEC" run_app.py "$@"
