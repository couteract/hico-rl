#!/usr/bin/env bash
# B01R Gate Test Runner
# Executes all required tests and generates structured report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
declare -a PASSED_TESTS=()
declare -a FAILED_TESTS=()
TOTAL_TESTS=0

# GPU test flag
export HIMEM_RUN_GPU_TESTS="${HIMEM_RUN_GPU_TESTS:-0}"

echo "=========================================="
echo "B01R Gate Test Suite"
echo "=========================================="
echo "GPU Tests: $([ "$HIMEM_RUN_GPU_TESTS" = "1" ] && echo "ENABLED" || echo "DISABLED (set HIMEM_RUN_GPU_TESTS=1 to enable)")"
echo ""

# Helper function to run a test
run_test() {
    local test_file="$1"
    local test_name="$2"
    local skip_reason="${3:-}"

    ((TOTAL_TESTS++))

    if [ -n "$skip_reason" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $test_name: $skip_reason"
        return 0
    fi

    echo -n "Running $test_name... "

    if python -m pytest "$test_file" -v --tb=short > "/tmp/himem_test_$TOTAL_TESTS.log" 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASSED_TESTS+=("$test_name")
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        FAILED_TESTS+=("$test_name")
        echo "  Log: /tmp/himem_test_$TOTAL_TESTS.log"
        return 1
    fi
}

# CPU Tests (always run)
echo "=== CPU Tests ==="
run_test "tests/himem/test_himem_protocol.py" "Protocol CPU"
run_test "tests/himem/test_event_mamba_cpu.py" "Event Mamba CPU"
run_test "tests/memory/himem/test_cpu_basic.py" "CPU Basic"

# GPU Tests (conditional)
if [ "$HIMEM_RUN_GPU_TESTS" = "1" ]; then
    echo ""
    echo "=== GPU Tests ==="

    # Check CUDA availability
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        run_test "tests/himem/test_event_mamba_gpu.py" "Event Mamba GPU"
        run_test "tests/memory/himem/test_gpu_backends.py" "GPU Backends"
        run_test "tests/memory/himem/test_mamba2_official.py" "Mamba2 Official GPU"
    else
        echo -e "${YELLOW}[SKIP]${NC} GPU tests: CUDA not available"
    fi
else
    echo ""
    echo "=== GPU Tests (Skipped) ==="
    echo "Set HIMEM_RUN_GPU_TESTS=1 to enable GPU tests"
fi

# Recorder Tests
echo ""
echo "=== Recorder Tests ==="
run_test "tests/himem/test_recorded_protocol.py" "Recorded Protocol"

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total: $TOTAL_TESTS"
echo -e "${GREEN}Passed: ${#PASSED_TESTS[@]}${NC}"
echo -e "${RED}Failed: ${#FAILED_TESTS[@]}${NC}"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    exit 1
else
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
