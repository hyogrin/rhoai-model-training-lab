#!/usr/bin/env bash
# stop_backend.sh — Stop the backend started with start_backend.sh --background
# Usage: bash scripts/stop_backend.sh
set -euo pipefail

# ─── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${PROJECT_ROOT}/.backend.pid"
LOG_FILE="${PROJECT_ROOT}/.backend.log"

# ─── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# ─── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Stop the FastAPI backend started by start_backend.sh --background.
Only kills processes started by start_backend.sh (tracked via PID file).

Options:
  --force       Send SIGKILL instead of SIGTERM
  -h, --help    Show this help

EOF
    exit "${1:-0}"
}

FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=true; shift ;;
        -h|--help)
            usage 0 ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            usage 1 ;;
    esac
done

# ─── Check PID file ───────────────────────────────────────────────────────────
if [[ ! -f "${PID_FILE}" ]]; then
    echo -e "${YELLOW}No PID file found at ${PID_FILE}${NC}"
    echo -e "${YELLOW}Backend may not be running, or was started in foreground mode.${NC}"
    exit 0
fi

PID="$(cat "${PID_FILE}")"

if [[ -z "${PID}" ]]; then
    echo -e "${YELLOW}PID file is empty. Cleaning up.${NC}"
    rm -f "${PID_FILE}"
    exit 0
fi

# ─── Verify the process is actually a backend process ──────────────────────────
if ! kill -0 "${PID}" 2>/dev/null; then
    echo -e "${YELLOW}Process ${PID} is not running. Cleaning up PID file.${NC}"
    rm -f "${PID_FILE}"
    exit 0
fi

# Verify it's a uvicorn/python process (safety check)
PROC_CMD=""
if command -v ps &>/dev/null; then
    PROC_CMD="$(ps -p "${PID}" -o command= 2>/dev/null || true)"
fi

if [[ -n "${PROC_CMD}" ]] && ! echo "${PROC_CMD}" | grep -qE '(uvicorn|python|rhoai)'; then
    echo -e "${RED}Error: PID ${PID} does not appear to be a backend process.${NC}" >&2
    echo -e "${RED}Command: ${PROC_CMD}${NC}" >&2
    echo -e "${YELLOW}Refusing to kill. Remove ${PID_FILE} manually if stale.${NC}" >&2
    exit 1
fi

# ─── Send signal ───────────────────────────────────────────────────────────────
if ${FORCE}; then
    echo -e "Sending SIGKILL to PID ${PID}..."
    kill -9 "${PID}" 2>/dev/null || true
else
    echo -e "Sending SIGTERM to PID ${PID}..."
    kill -15 "${PID}" 2>/dev/null || true
fi

# ─── Wait for termination ─────────────────────────────────────────────────────
echo "Waiting for process to terminate..."
WAIT_SECONDS=10
for i in $(seq 1 ${WAIT_SECONDS}); do
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo -e "${GREEN}✅  Backend stopped (PID ${PID})${NC}"
        rm -f "${PID_FILE}"
        exit 0
    fi
    sleep 1
done

# Process didn't stop gracefully
if kill -0 "${PID}" 2>/dev/null; then
    echo -e "${YELLOW}Process did not terminate in ${WAIT_SECONDS}s. Sending SIGKILL...${NC}"
    kill -9 "${PID}" 2>/dev/null || true
    sleep 1

    if ! kill -0 "${PID}" 2>/dev/null; then
        echo -e "${GREEN}✅  Backend force-stopped (PID ${PID})${NC}"
        rm -f "${PID_FILE}"
    else
        echo -e "${RED}❌  Failed to stop backend (PID ${PID})${NC}" >&2
        exit 1
    fi
else
    echo -e "${GREEN}✅  Backend stopped (PID ${PID})${NC}"
    rm -f "${PID_FILE}"
fi
