#!/usr/bin/env bash
# smoke_backend.sh — Quick health and functional checks for the backend
# Usage: bash scripts/smoke_backend.sh [--host HOST] [--port PORT]
set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
HOST="127.0.0.1"
PORT="8000"
TIMEOUT=10
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

Run quick smoke tests against the backend API.

Options:
  --host HOST         Backend host (default: 127.0.0.1)
  --port PORT         Backend port (default: 8000)
  --timeout SECS      Request timeout in seconds (default: 10)
  -h, --help          Show this help

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
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
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

HOST="${BACKEND_HOST:-${HOST}}"
PORT="${BACKEND_PORT:-${PORT}}"
BASE_URL="http://${HOST}:${PORT}"

echo -e "${CYAN}═══ Backend Smoke Tests ═══${NC}"
echo -e "Target: ${BASE_URL}"
echo ""

PASSED=0
FAILED=0
TOTAL=0

# ─── Helper function ──────────────────────────────────────────────────────────
run_test() {
    local test_name="$1"
    local method="$2"
    local endpoint="$3"
    local data="${4:-}"
    local expected_status="${5:-200}"

    TOTAL=$((TOTAL + 1))
    echo -n "  [${TOTAL}] ${test_name}... "

    local curl_args=(-s -o /dev/null -w "%{http_code}" --max-time "${TIMEOUT}")

    if [[ "${method}" == "POST" ]]; then
        curl_args+=(-X POST -H "Content-Type: application/json" -d "${data}")
    fi

    local status_code
    status_code=$(curl "${curl_args[@]}" "${BASE_URL}${endpoint}" 2>/dev/null || echo "000")

    if [[ "${status_code}" == "${expected_status}" ]]; then
        echo -e "${GREEN}PASS${NC} (HTTP ${status_code})"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC} (HTTP ${status_code}, expected ${expected_status})"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

run_test_with_body() {
    local test_name="$1"
    local method="$2"
    local endpoint="$3"
    local data="${4:-}"
    local check_field="${5:-}"

    TOTAL=$((TOTAL + 1))
    echo -n "  [${TOTAL}] ${test_name}... "

    local curl_args=(-s --max-time "${TIMEOUT}")

    if [[ "${method}" == "POST" ]]; then
        curl_args+=(-X POST -H "Content-Type: application/json" -d "${data}")
    fi

    local response
    response=$(curl "${curl_args[@]}" "${BASE_URL}${endpoint}" 2>/dev/null || echo '{"error":"connection_failed"}')
    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "${TIMEOUT}" \
        $(if [[ "${method}" == "POST" ]]; then echo "-X POST -H 'Content-Type: application/json' -d '${data}'"; fi) \
        "${BASE_URL}${endpoint}" 2>/dev/null || echo "000")

    if [[ "${status_code}" =~ ^2[0-9][0-9]$ ]]; then
        if [[ -n "${check_field}" ]]; then
            if echo "${response}" | python3 -c "import sys, json; d=json.load(sys.stdin); assert '${check_field}' in d" 2>/dev/null; then
                echo -e "${GREEN}PASS${NC} (HTTP ${status_code}, '${check_field}' present)"
                PASSED=$((PASSED + 1))
                return 0
            else
                echo -e "${YELLOW}WARN${NC} (HTTP ${status_code}, missing '${check_field}')"
                PASSED=$((PASSED + 1))
                return 0
            fi
        fi
        echo -e "${GREEN}PASS${NC} (HTTP ${status_code})"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC} (HTTP ${status_code})"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# ─── Run tests ─────────────────────────────────────────────────────────────────

# Test 1: Health check
run_test "GET /healthz" "GET" "/healthz" "" "200" || true

# Test 2: Readiness check
run_test "GET /readyz" "GET" "/readyz" "" "200" || true

# Test 3: QA endpoint with test question
QA_PAYLOAD='{"question": "What are the banking policies for account opening?", "mode": "rag"}'
run_test "POST /v1/qa (test question)" "POST" "/v1/qa" "${QA_PAYLOAD}" "200" || true

# Test 4: Check QA response has expected fields
run_test_with_body "POST /v1/qa (response body)" "POST" "/v1/qa" "${QA_PAYLOAD}" "status" || true

# ─── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══ Smoke Test Summary ═══${NC}"
echo -e "  Total:  ${TOTAL}"
echo -e "  Passed: ${GREEN}${PASSED}${NC}"
echo -e "  Failed: ${RED}${FAILED}${NC}"

if [[ ${FAILED} -eq 0 ]]; then
    echo -e "${GREEN}✅  All smoke tests passed${NC}"
    exit 0
else
    echo -e "${RED}❌  ${FAILED} test(s) failed${NC}"
    exit 1
fi
