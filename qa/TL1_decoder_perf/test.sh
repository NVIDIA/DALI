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
    curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${DISTRO}/${ARCH}/3bf863cc.pub" \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg
    apt update && apt install -y nsight-systems-cli
}

LOG1="dali_legacy.log"
LOG2="dali_nvimgcodec.log"
LOG1_TP="dali_legacy_new_tp.log"
LOG2_TP="dali_nvimgcodec_new_tp.log"
LOG1_NDD="dali_ndd_legacy.log"
LOG2_NDD="dali_ndd_nvimgcodec.log"

function CLEAN_AND_EXIT {
    rm -rf ${LOG1}
    rm -rf ${LOG2}
    rm -rf ${LOG1_TP}
    rm -rf ${LOG2_TP}
    rm -rf ${LOG1_NDD}
    rm -rf ${LOG2_NDD}
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

# SPEC:DA-11356-002_v04 - run all benchmarks (optionally with nsys when NSYS_REP is set per run).
run_all_benchmarks() {
    if [ "$(uname -p)" == "x86_64" ]; then
        # Hopper
        TASKSET="taskset --cpu-list 0-127"
        BENCH_ARGS="--width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -j 70 --hw_load 0.12"
    else
        # GraceHopper
        TASKSET="taskset --cpu-list 0-71"
        BENCH_ARGS="--width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -j 72 --hw_load 0.11"
    fi
    run_bench "${LOG1}"     ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p rn50
    run_bench "${LOG2}"     ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p rn50 --experimental_decoder
    DALI_USE_NEW_THREAD_POOL=1 run_bench "${LOG1_TP}" ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p rn50
    DALI_USE_NEW_THREAD_POOL=1 run_bench "${LOG2_TP}" ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p rn50 --experimental_decoder
    run_bench "${LOG1_NDD}" ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p ndd_rn50
    run_bench "${LOG2_NDD}" ${TASKSET} python hw_decoder_bench.py ${BENCH_ARGS} -p ndd_rn50 --experimental_decoder
}

test_body() {
    if [ "$(uname -p)" == "x86_64" ]; then
        MIN_PERF=19000
        MIN_PERF2=18000  # TODO(janton): target is to be 19000 as well
        MIN_PERF_NDD=14000
        MIN_PERF2_NDD=14000  # TODO(janton): remove this second value.
    else
        MIN_PERF=29000
        MIN_PERF2=29000  # TODO(janton): remove this second value.
        MIN_PERF_NDD=20000
        MIN_PERF2_NDD=20000  # TODO(janton): remove this second value.
    fi

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

    PERF_RESULT1=$(perf_check "${LOG1}" "$MIN_PERF")
    PERF_RESULT2=$(perf_check "${LOG2}" "$MIN_PERF2")
    PERF_RESULT1_NDD=$(perf_check "${LOG1_NDD}" "$MIN_PERF_NDD")
    PERF_RESULT2_NDD=$(perf_check "${LOG2_NDD}" "$MIN_PERF2_NDD")
    PERF_RESULT3=$(perf_check "${LOG2}" "$(extract_perf "${LOG1}")" 5)
    PERF_RESULT3_NDD=$(perf_check "${LOG2_NDD}" "$(extract_perf "${LOG1_NDD}")" 5)
    PERF_RESULT1_TP=$(perf_check "${LOG1_TP}" "$(extract_perf "${LOG1}")" 2)
    PERF_RESULT2_TP=$(perf_check "${LOG2_TP}" "$(extract_perf "${LOG2}")" 2)

    echo "PERF_RESULT1=${PERF_RESULT1}"
    echo "PERF_RESULT2=${PERF_RESULT2}"
    echo "PERF_RESULT3=${PERF_RESULT3}"
    echo "PERF_RESULT1_TP=${PERF_RESULT1_TP} (informational)"
    echo "PERF_RESULT2_TP=${PERF_RESULT2_TP} (informational)"
    echo "PERF_RESULT1_NDD=${PERF_RESULT1_NDD}"
    echo "PERF_RESULT2_NDD=${PERF_RESULT2_NDD}"
    echo "PERF_RESULT3_NDD=${PERF_RESULT3_NDD}"

    # don't check experimental decoder performance with dynamic mode (PERF_RESULT2_NDD, PERF_RESULT3_NDD)
    # PERF_RESULT1_TP and PERF_RESULT2_TP are informational only (new thread pool is experimental)
    if [[ "$PERF_RESULT1" == "OK" && "$PERF_RESULT2" == "OK" && "$PERF_RESULT3" == "OK" && "$PERF_RESULT1_NDD" == "OK" ]]; then
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
