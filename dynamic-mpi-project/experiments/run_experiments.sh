#!/usr/bin/env bash
# ==============================================================================
# Dynamic MPI Project - Automated Experiment Runner
# ==============================================================================
# Usage:
#   ./run_experiments.sh [options]
#
# Options:
#   -c <config_file>  Path to configuration file (default: configs/default.conf)
#   -n <num_procs>    Number of MPI processes (overrides config)
#   -t <target>       Executable to run: all, baseline, reproduction, extension (default: all)
#   -h                Show help message
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/configs/default.conf"
RESULTS_DIR="${SCRIPT_DIR}/results"
BUILD_DIR="${PROJECT_ROOT}/build"

TARGET="all"
PROC_OVERRIDE=""

show_help() {
    echo "Dynamic MPI Project - Benchmark Runner"
    echo "Usage: $0 [-c <config_file>] [-n <num_procs>] [-t <target>] [-h]"
    echo ""
    echo "Targets:"
    echo "  baseline      Run baseline 3D stencil benchmark"
    echo "  reproduction  Run paper dynamic protocol reproduction"
    echo "  extension     Run adaptive chunking extension"
    echo "  all           Run all three benchmarks sequentially (default)"
    exit 0
}

while getopts "c:n:t:h" opt; do
    case "${opt}" in
        c) CONFIG_FILE="${OPTARG}" ;;
        n) PROC_OVERRIDE="${OPTARG}" ;;
        t) TARGET="${OPTARG}" ;;
        h) show_help ;;
        *) echo "Invalid option. Use -h for help." >&2; exit 1 ;;
    esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: Configuration file '${CONFIG_FILE}' not found." >&2
    exit 1
fi

# shellcheck source=configs/default.conf
source "${CONFIG_FILE}"

NUM_PROCS="${PROC_OVERRIDE:-${MPI_PROCESSES}}"
mkdir -p "${RESULTS_DIR}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_FILE="${RESULTS_DIR}/benchmark_${TARGET}_np${NUM_PROCS}_${TIMESTAMP}.log"

echo "=========================================================="
echo " Starting Benchmark Run"
echo " Date/Time : $(date)"
echo " Target    : ${TARGET}"
echo " Processes : ${NUM_PROCS}"
echo " Config    : ${CONFIG_FILE}"
echo " Output    : ${OUTPUT_FILE}"
echo "=========================================================="

# Determine MPI launcher
if command -v srun &>/dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MPI_EXEC="srun -n ${NUM_PROCS}"
elif command -v mpirun &>/dev/null; then
    MPI_EXEC="mpirun -np ${NUM_PROCS}"
elif command -v mpiexec &>/dev/null; then
    MPI_EXEC="mpiexec -np ${NUM_PROCS}"
else
    echo "Error: Neither srun, mpirun, nor mpiexec was found in PATH." >&2
    exit 1
fi

run_single_target() {
    local name="$1"
    local bin="${BUILD_DIR}/${name}"

    if [[ ! -x "${bin}" ]]; then
        echo "Error: Binary '${bin}' not found or not executable." >&2
        echo "Please build the project first (e.g. 'make ${name}' or 'make all')." >&2
        return 1
    fi

    echo "--- Running ${name} (${NUM_PROCS} processes) ---" | tee -a "${OUTPUT_FILE}"
    ${MPI_EXEC} "${bin}" 2>&1 | tee -a "${OUTPUT_FILE}"
    echo "" | tee -a "${OUTPUT_FILE}"
}

{
    echo "# Dynamic MPI Benchmark Output"
    echo "# Timestamp: ${TIMESTAMP}"
    echo "# Processes: ${NUM_PROCS}"
    echo "# Grid Size: ${GRID_NX}x${GRID_NY}x${GRID_NZ}"
    echo "# Iterations: ${NUM_ITERATIONS}"
    echo ""
} > "${OUTPUT_FILE}"

case "${TARGET}" in
    baseline)
        run_single_target "baseline"
        ;;
    reproduction)
        run_single_target "reproduction"
        ;;
    extension)
        run_single_target "extension"
        ;;
    all)
        run_single_target "baseline"
        run_single_target "reproduction"
        run_single_target "extension"
        ;;
    *)
        echo "Unknown target '${TARGET}'. Valid: baseline, reproduction, extension, all." >&2
        exit 1
        ;;
esac

echo "=========================================================="
echo " Benchmark run completed successfully."
echo " Results saved to: ${OUTPUT_FILE}"
echo "=========================================================="

