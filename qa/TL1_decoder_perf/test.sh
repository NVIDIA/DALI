#!/bin/bash -e
# used pip packages
pip_packages='numpy'
target_dir=./internal_tools

# One-time pre-step: install nsys (NVIDIA Nsight Systems) if not present
do_once() {
    if command -v nsys &>/dev/null; then
        echo "nsys already installed: $(nsys --version)"
        return
    fi
    DISTRO=$(source /etc/lsb-release && echo "$DISTRIB_RELEASE" | tr -d .)
    ARCH=$(dpkg --print-architecture)
    apt update && apt install -y --no-install-recommends gnupg curl
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu${DISTRO}/${ARCH} /" \
        | tee /etc/apt/sources.list.d/nvidia-devtools.list
    curl -fsSL "https://developer.download.nvidia.com/devtools/repos/ubuntu${DISTRO}/${ARCH}/nvidia.pub" \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg
    apt update && apt install -y nsight-systems-cli
}

LOG_RN50="dali_rn50.log"
LOG_RN50_TP="dali_rn50_new_tp.log"
LOG_NDD="dali_ndd.log"
LOG_NDD_CAPTURE="dali_ndd_capture.log"

function CLEAN_AND_EXIT {
    rm -rf ${LOG_RN50}
    rm -rf ${LOG_RN50_TP}
    rm -rf ${LOG_NDD}
    rm -rf ${LOG_NDD_CAPTURE}
    exit $1
}

# Run a single benchmark; if NSYS_REP is set, wrap with nsys profiling.
# The profile filename is derived from the log filename (e.g. foo.log -> foo.nsys-rep).
run_bench() {
    local log_file="$1"
    shift
    if [ -n "${NSYS_REP}" ]; then
        local profile_name="${log_file%.log}.nsys-rep"
        nsys profile -o "${profile_name}" --stats=true "$@" | tee "${log_file}"
    else
        "$@" | tee "${log_file}"
    fi
}

# Per-platform benchmark settings, keyed by "<CPU arch>/<GPU compute capability>".
set_platform_params() {
    local arch compute_cap
    arch=$(uname -m)
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0)
    case "${arch}/${compute_cap}" in
        x86_64/*)  # x86 + any GPU (tuned on Hopper)
            CPU_LIST="0-127"
            NUM_CORES=70
            HW_LOAD=0.12
            MIN_PERF=19000
            MIN_PERF_NDD=14000
            ;;
        aarch64/10.7)  # Vera Rubin (VR)
            # 2 sockets x 88 cores x 2 threads; 0-87 + 176-263 are socket 0 and its hyperthread siblings (local to GPU 0)
            CPU_LIST="0-87,176-263"
            NUM_CORES=174  # leave 2 cores for internal DALI tasks
            HW_LOAD=0.13
            MIN_PERF=40000
            MIN_PERF_NDD=27000
            ;;
        aarch64/9.0|aarch64/10.0|aarch64/10.3)  # Grace Hopper (GH200), Grace Blackwell (GB200, GB300)
            CPU_LIST="0-71"
            NUM_CORES=72
            HW_LOAD=0.11
            MIN_PERF=29000
            MIN_PERF_NDD=20000
            ;;
        *)
            echo "Unsupported platform: got ${arch}/${compute_cap}, expected x86_64 with any GPU, or aarch64 with GPU compute capability 9.0, 10.0, 10.3 or 10.7." >&2
            CLEAN_AND_EXIT 1
            ;;
    esac
}

# SPEC:DA-11356-002_v04 - run all benchmarks (optionally with nsys when NSYS_REP is set per run).
run_all_benchmarks() {
    TASKSET="taskset --cpu-list ${CPU_LIST}"
    BENCH_ARGS="--width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -j ${NUM_CORES} --hw_load ${HW_LOAD}"
    run_bench "${LOG_RN50}" ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p rn50
    DALI_USE_NEW_THREAD_POOL=1 run_bench "${LOG_RN50_TP}" ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p rn50
    run_bench "${LOG_NDD}" ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p ndd_rn50
    run_bench "${LOG_NDD_CAPTURE}" ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p ndd_rn50 --ndd-capture
}

test_body() {
    set_platform_params

    # First run: all benchmarks without nsys
    unset NSYS_REP
    run_all_benchmarks

    # Regex: extract "Total Throughput: X frames/sec" -> X
    extract_perf() {
        grep -oP 'Total Throughput: \K[0-9]+(\.[0-9]+)?(?= frames/sec)' "$1"
    }

    perf_check() {
        local value=$(extract_perf "$1")
        local min_value=$2
        local percent=${3:-0}
        local tolerance=$(awk -v p="$percent" 'BEGIN{print p/100}')
        echo "$value $min_value" | awk -v tol="$tolerance" '{
          lower = $2 * (1 - tol);
          if ($1 >= lower) {print "OK"} else {print "FAIL"}
        }'
    }

    PERF_RESULT=$(perf_check "${LOG_RN50}" "$MIN_PERF")
    PERF_RESULT_NDD=$(perf_check "${LOG_NDD}" "$MIN_PERF_NDD")
    PERF_RESULT_NDD_CAPTURE=$(perf_check "${LOG_NDD_CAPTURE}" "$MIN_PERF")
    PERF_RESULT_TP=$(perf_check "${LOG_RN50_TP}" "$(extract_perf "${LOG_RN50}")" 2)

    echo "PERF_RESULT=${PERF_RESULT}"
    echo "PERF_RESULT_NDD=${PERF_RESULT_NDD}"
    echo "PERF_RESULT_NDD_CAPTURE=${PERF_RESULT_NDD_CAPTURE}"
    echo "PERF_RESULT_TP=${PERF_RESULT_TP} (informational)"

    # PERF_RESULT_TP is informational only (new thread pool is experimental)
    if [[ "$PERF_RESULT" == "OK" && "$PERF_RESULT_NDD" == "OK" && "$PERF_RESULT_NDD_CAPTURE" == "OK" ]]; then
        CLEAN_AND_EXIT 0
    fi

    # On failure: re-run all benchmarks with nsys and save profiles to core_artifacts
    echo "Performance check failed; re-running all benchmarks with nsys profiling..."
    ARTIFACTS_DIR="${topdir}/core_artifacts"
    mkdir -p "${ARTIFACTS_DIR}"

    NSYS_REP="enabled" run_all_benchmarks

    cp -f *.nsys-rep "${ARTIFACTS_DIR}/" 2>/dev/null || true
    echo "nsys profiles saved to ${ARTIFACTS_DIR}"
    CLEAN_AND_EXIT 1
}
pushd ../..
source ./qa/test_template.sh
popd
