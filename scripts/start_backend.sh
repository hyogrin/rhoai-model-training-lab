#!/usr/bin/env bash
# start_backend.sh — Start the FastAPI RAG harness backend
# Usage: bash scripts/start_backend.sh [--host HOST] [--port PORT] [--background]
set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
HOST="127.0.0.1"
PORT="8000"
BACKGROUND=false
WORKERS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${PROJECT_ROOT}/.backend.pid"
LOG_FILE="${PROJECT_ROOT}/.backend.log"

# ─── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

# ─── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Start the FastAPI RAG harness backend.

Options:
  --host HOST         Bind address (default: 127.0.0.1)
  --port PORT         Bind port (default: 8000)
  --workers N         Number of uvicorn workers (default: 1)
  --background        Run in background with PID/log management
  -h, --help          Show this help

When started with --background:
  - PID is saved to ${PID_FILE}
  - Logs are written to ${LOG_FILE}
  - Use scripts/stop_backend.sh to stop

EOF
    exit "${1:-0}"
}

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --workers)
            WORKERS="$2"; shift 2 ;;
        --background)
            BACKGROUND=true; shift ;;
        -h|--help)
            usage 0 ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            usage 1 ;;
    esac
done

# ─── Load .env ─────────────────────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Override from env if set
HOST="${BACKEND_HOST:-${HOST}}"
PORT="${BACKEND_PORT:-${PORT}}"

# ─── Port conflict detection ──────────────────────────────────────────────────
echo -e "${CYAN}Checking port ${PORT}...${NC}"
if command -v lsof &>/dev/null; then
    EXISTING_PID="$(lsof -ti :"${PORT}" 2>/dev/null || true)"
    if [[ -n "${EXISTING_PID}" ]]; then
        echo -e "${RED}Error: Port ${PORT} is already in use by PID ${EXISTING_PID}${NC}" >&2
        echo -e "${YELLOW}Use --port to specify a different port, or stop the existing process.${NC}" >&2
        exit 1
    fi
elif command -v ss &>/dev/null; then
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        echo -e "${RED}Error: Port ${PORT} is already in use.${NC}" >&2
        exit 1
    fi
fi
echo -e "  ✅  Port ${PORT} is available"

# ─── Check if already running from background ─────────────────────────────────
if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}")"
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo -e "${YELLOW}Backend already running (PID ${OLD_PID}).${NC}"
        echo -e "${YELLOW}Stop it first: bash scripts/stop_backend.sh${NC}"
        exit 1
    else
        rm -f "${PID_FILE}"
    fi
fi

# ─── Start the backend ────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

UVICORN_CMD="python3 -m uvicorn rhoai_model_training_lab.api:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"

echo -e "${CYAN}Starting backend...${NC}"
echo -e "  Host: ${HOST}"
echo -e "  Port: ${PORT}"
echo -e "  Workers: ${WORKERS}"

if ${BACKGROUND}; then
    echo -e "  Mode: background"
    echo -e "  PID file: ${PID_FILE}"
    echo -e "  Log file: ${LOG_FILE}"
    echo ""

    nohup ${UVICORN_CMD} > "${LOG_FILE}" 2>&1 &
    BACKEND_PID=$!

    echo "${BACKEND_PID}" > "${PID_FILE}"

    # Wait briefly and check if process started
    sleep 2
    if kill -0 "${BACKEND_PID}" 2>/dev/null; then
        echo -e "${GREEN}✅  Backend started (PID ${BACKEND_PID})${NC}"
        echo -e "  URL: http://${HOST}:${PORT}"
        echo -e "  Health: http://${HOST}:${PORT}/healthz"
        echo ""
        echo -e "  Stop with: bash scripts/stop_backend.sh"
        echo -e "  Logs: tail -f ${LOG_FILE}"
    else
        echo -e "${RED}❌  Backend failed to start. Check logs: ${LOG_FILE}${NC}" >&2
        rm -f "${PID_FILE}"
        exit 1
    fi
else
    echo -e "  Mode: foreground (Ctrl+C to stop)"
    echo -e "  URL: http://${HOST}:${PORT}"
    echo ""
    exec ${UVICORN_CMD}
fi
