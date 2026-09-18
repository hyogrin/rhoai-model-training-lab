#!/usr/bin/env bash
# setup_env.sh — Environment setup for RHOAI Model Training Lab
# Usage: bash scripts/setup_env.sh --profile <lora|osft|backend|evaluation|preparation|all>
set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
PROFILE=""
VENV_DIR=".venv"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── Colors / symbols ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'
OK="✅"
FAIL="❌"
WARN="⚠️"

# ─── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --profile <PROFILE> [--venv DIR]

Profiles:
  lora          Install for LoRA fine-tuning (requires GPU)
  osft          Install for OSFT fine-tuning (requires GPU)
  backend       Install for RAG harness backend
  evaluation    Install for τ evaluation
  preparation   Install for SDG data preparation
  all           Install all optional dependencies

Options:
  --profile PROFILE   Required. One of: lora, osft, backend, evaluation, preparation, all
  --venv DIR          Virtual-environment directory (default: .venv)
  -h, --help          Show this help

EOF
    exit "${1:-0}"
}

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"; shift 2 ;;
        --venv)
            VENV_DIR="$2"; shift 2 ;;
        -h|--help)
            usage 0 ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            usage 1 ;;
    esac
done

if [[ -z "${PROFILE}" ]]; then
    echo -e "${RED}Error: --profile is required.${NC}" >&2
    usage 1
fi

VALID_PROFILES="lora osft backend evaluation preparation all"
if ! echo "${VALID_PROFILES}" | tr ' ' '\n' | grep -qx "${PROFILE}"; then
    echo -e "${RED}Error: Invalid profile '${PROFILE}'. Must be one of: ${VALID_PROFILES}${NC}" >&2
    exit 1
fi

# ─── Status tracking ──────────────────────────────────────────────────────────
declare -a STATUS_LINES=()
add_status() {
    local symbol="$1" label="$2"
    STATUS_LINES+=("${symbol}  ${label}")
}

# ─── 1. Check Python version ──────────────────────────────────────────────────
echo -e "${CYAN}Checking Python version...${NC}"
PYTHON_CMD=""
for candidate in python3 python; do
    if command -v "${candidate}" &>/dev/null; then
        PYTHON_CMD="${candidate}"
        break
    fi
done

if [[ -z "${PYTHON_CMD}" ]]; then
    add_status "${FAIL}" "Python not found"
    echo -e "${RED}Python interpreter not found.${NC}" >&2
    for line in "${STATUS_LINES[@]}"; do echo -e "  ${line}"; done
    exit 1
fi

PYTHON_VERSION="$("${PYTHON_CMD}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
PYTHON_MAJOR="$("${PYTHON_CMD}" -c 'import sys; print(sys.version_info.major)')"
PYTHON_MINOR="$("${PYTHON_CMD}" -c 'import sys; print(sys.version_info.minor)')"

if [[ "${PYTHON_MAJOR}" -lt "${MIN_PYTHON_MAJOR}" ]] || \
   { [[ "${PYTHON_MAJOR}" -eq "${MIN_PYTHON_MAJOR}" ]] && [[ "${PYTHON_MINOR}" -lt "${MIN_PYTHON_MINOR}" ]]; }; then
    add_status "${FAIL}" "Python ${PYTHON_VERSION} < required ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}"
    echo -e "${RED}Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required, found ${PYTHON_VERSION}.${NC}" >&2
    for line in "${STATUS_LINES[@]}"; do echo -e "  ${line}"; done
    exit 1
fi
add_status "${OK}" "Python ${PYTHON_VERSION}"
echo -e "  ${OK}  Python ${PYTHON_VERSION}"

# ─── 2. Create / activate virtualenv ──────────────────────────────────────────
VENV_PATH="${PROJECT_ROOT}/${VENV_DIR}"
echo -e "${CYAN}Setting up virtualenv at ${VENV_DIR}...${NC}"

if [[ ! -d "${VENV_PATH}" ]]; then
    "${PYTHON_CMD}" -m venv "${VENV_PATH}"
    add_status "${OK}" "Created venv: ${VENV_DIR}"
    echo -e "  ${OK}  Created venv: ${VENV_DIR}"
else
    add_status "${OK}" "Existing venv: ${VENV_DIR}"
    echo -e "  ${OK}  Existing venv: ${VENV_DIR}"
fi

# Activate
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
add_status "${OK}" "Activated venv"
echo -e "  ${OK}  Activated venv"

# Upgrade pip
pip install --quiet --upgrade pip setuptools wheel
add_status "${OK}" "pip upgraded"
echo -e "  ${OK}  pip upgraded"

# ─── 3. Install the package with profile-specific extras ──────────────────────
echo -e "${CYAN}Installing rhoai-model-training-lab [${PROFILE}]...${NC}"

# Map profile → pip extras
case "${PROFILE}" in
    lora)         EXTRAS="lora" ;;
    osft)         EXTRAS="osft" ;;
    backend)      EXTRAS="backend" ;;
    evaluation)   EXTRAS="eval" ;;
    preparation)  EXTRAS="sdg" ;;
    all)          EXTRAS="all" ;;
esac

cd "${PROJECT_ROOT}"
if pip install -e ".[${EXTRAS}]" 2>&1; then
    add_status "${OK}" "Installed [${EXTRAS}]"
    echo -e "  ${OK}  Installed [${EXTRAS}]"
else
    add_status "${FAIL}" "pip install failed"
    echo -e "  ${FAIL}  pip install -e '.[${EXTRAS}]' failed" >&2
    for line in "${STATUS_LINES[@]}"; do echo -e "  ${line}"; done
    exit 1
fi

# ─── 4. GPU check for training profiles ───────────────────────────────────────
GPU_PROFILES="lora osft all"
if echo "${GPU_PROFILES}" | tr ' ' '\n' | grep -qx "${PROFILE}"; then
    echo -e "${CYAN}Checking GPU availability...${NC}"
    if command -v nvidia-smi &>/dev/null; then
        GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"
        if [[ -n "${GPU_INFO}" ]]; then
            add_status "${OK}" "GPU: ${GPU_INFO}"
            echo -e "  ${OK}  GPU: ${GPU_INFO}"
        else
            add_status "${WARN}" "nvidia-smi present but no GPU detected"
            echo -e "  ${WARN}  nvidia-smi present but no GPU detected"
        fi
    else
        add_status "${WARN}" "nvidia-smi not found — GPU unavailable"
        echo -e "  ${WARN}  nvidia-smi not found — GPU unavailable"
    fi

    # Check CUDA via PyTorch
    CUDA_AVAILABLE="$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
    if [[ "${CUDA_AVAILABLE}" == "True" ]]; then
        CUDA_DEVICE="$(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
        add_status "${OK}" "PyTorch CUDA: ${CUDA_DEVICE}"
        echo -e "  ${OK}  PyTorch CUDA: ${CUDA_DEVICE}"
    elif [[ "${CUDA_AVAILABLE}" == "False" ]]; then
        add_status "${WARN}" "PyTorch CUDA not available — training requires a GPU"
        echo -e "  ${WARN}  PyTorch CUDA not available — training requires a GPU"
    else
        add_status "${WARN}" "Could not check PyTorch CUDA (torch may not be installed yet)"
        echo -e "  ${WARN}  Could not check PyTorch CUDA"
    fi
fi

# ─── 5. Verify required env vars from .env ────────────────────────────────────
echo -e "${CYAN}Checking environment variables...${NC}"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a; source "${ENV_FILE}"; set +a
    add_status "${OK}" "Loaded .env"
    echo -e "  ${OK}  Loaded .env"
else
    add_status "${WARN}" ".env not found — copy .env.example to .env and fill required values"
    echo -e "  ${WARN}  .env not found — copy .env.example to .env and fill required values"
fi

# Profile-specific required env vars
declare -a REQUIRED_VARS=()
case "${PROFILE}" in
    lora|osft)
        REQUIRED_VARS=(BASE_MODEL_ID) ;;
    backend)
        REQUIRED_VARS=(BACKEND_HOST BACKEND_PORT EMBEDDING_MODEL_ID) ;;
    evaluation)
        REQUIRED_VARS=(BASE_MODEL_ID BASE_SERVING_ENDPOINT MLFLOW_TRACKING_URI) ;;
    preparation)
        REQUIRED_VARS=(SDG_TEACHER_ENDPOINT SDG_TEACHER_API_KEY SDG_TEACHER_MODEL) ;;
    all)
        REQUIRED_VARS=(BASE_MODEL_ID) ;;
esac

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    val="${!var:-}"
    if [[ -z "${val}" ]]; then
        MISSING_VARS+=("${var}")
    fi
done

if [[ ${#MISSING_VARS[@]} -eq 0 ]]; then
    add_status "${OK}" "All required env vars set for profile '${PROFILE}'"
    echo -e "  ${OK}  All required env vars set"
else
    for mv in "${MISSING_VARS[@]}"; do
        add_status "${WARN}" "Missing env var: ${mv}"
        echo -e "  ${WARN}  Missing env var: ${mv}"
    done
fi

# ─── 6. Print summary ─────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Setup Summary — profile: ${PROFILE}${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
for line in "${STATUS_LINES[@]}"; do
    echo -e "  ${line}"
done
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"

# Check if any FAIL items
HAS_FAIL=false
for line in "${STATUS_LINES[@]}"; do
    if [[ "${line}" == *"${FAIL}"* ]]; then
        HAS_FAIL=true
        break
    fi
done

if ${HAS_FAIL}; then
    echo -e "${RED}Setup completed with errors. Review issues above.${NC}"
    exit 1
else
    echo -e "${GREEN}Setup complete. Activate the venv with: source ${VENV_DIR}/bin/activate${NC}"
    exit 0
fi
