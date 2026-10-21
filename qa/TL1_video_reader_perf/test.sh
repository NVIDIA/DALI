#!/bin/bash -e
# used pip packages
pip_packages='numpy nvidia-ml-py==11.450.51'
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

LOG_PERF="video_reader_perf.log"
JSON_PERF="video_reader_perf.json"
JSON_HASH_LOW="video_reader_hash_low.json"
JSON_HASH_HIGH="video_reader_hash_high.json"

# Scenarios of internal_tools/video_reader_bench.py run for performance; see its --help
PERF_SCENARIOS="lanes,engines,single_stream,reader_depth,cache_key,startup,seq_longgop,cpu"
# Scenarios whose output is checked to be independent of num_threads and reader_depth
HASH_SCENARIOS="lanes,reader_depth,cache_key,cpu"
THREADS_LOW=1
THREADS_HIGH=16
BENCH_ARGS="--repeat 3 --min_time 3 -j ${THREADS_LOW},4,${THREADS_HIGH} --readers 1,2,4"

function CLEAN_AND_EXIT {
    rm -rf ${LOG_PERF} ${JSON_PERF} ${JSON_HASH_LOW} ${JSON_HASH_HIGH}
    exit $1
}

# Run the benchmark; if NSYS_REP is set, wrap it with nsys profiling (child processes included).
run_bench() {
    local log_file="$1"
    shift
    if [ -n "${NSYS_REP}" ]; then
        local profile_name="${log_file%.log}.nsys-rep"
        nsys profile -o "${profile_name}" --trace=cuda,nvtx,osrt "$@" | tee "${log_file}"
    else
        "$@" | tee "${log_file}"
    fi
}

test_body() {
    if [ "$(uname -p)" == "x86_64" ]; then
        # Hopper
        TASKSET="taskset --cpu-list 0-127"
        # Absolute minimums in frames/s (see qa/TL1_video_reader_perf/check_results.py);
        # empty: reported for reference only, not checked
        MIN_LANES=
        MIN_SINGLE_STREAM=
        MIN_CPU=
        MIN_ENGINES=
    else
        # GraceHopper
        TASKSET="taskset --cpu-list 0-71"
        MIN_LANES=
        MIN_SINGLE_STREAM=
        MIN_CPU=
        MIN_ENGINES=
    fi

    unset NSYS_REP
    PERF_RUN="OK"
    run_bench "${LOG_PERF}" ${TASKSET} python video_reader_bench.py ${BENCH_ARGS} \
        -s ${PERF_SCENARIOS} --json ${JSON_PERF} || PERF_RUN="FAIL"

    # The output must not depend on num_threads or on the reader's prefetch_queue_depth
    HASH_RESULT="OK"
    for threads in ${THREADS_LOW} ${THREADS_HIGH}; do
        [ ${threads} == ${THREADS_LOW} ] && json=${JSON_HASH_LOW} || json=${JSON_HASH_HIGH}
        ${TASKSET} python video_reader_bench.py -s ${HASH_SCENARIOS} -j ${threads} \
            --hash --hash_batches 4 --json ${json} || HASH_RESULT="FAIL"
    done
    python video_reader_bench.py --compare ${JSON_HASH_LOW} ${JSON_HASH_HIGH} --ignore threads \
        || HASH_RESULT="FAIL"
    python video_reader_bench.py --compare ${JSON_HASH_HIGH} ${JSON_HASH_HIGH} \
        --ignore reader_depth || HASH_RESULT="FAIL"

    CHECK_RESULT="OK"
    python ${topdir}/qa/TL1_video_reader_perf/check_results.py ${JSON_PERF} --tolerance 10 \
        --min lanes=${MIN_LANES} --min single_stream=${MIN_SINGLE_STREAM} \
        --min cpu=${MIN_CPU} --min engines=${MIN_ENGINES} || CHECK_RESULT="FAIL"

    echo "PERF_RUN=${PERF_RUN}"
    echo "HASH_RESULT=${HASH_RESULT}"
    echo "CHECK_RESULT=${CHECK_RESULT}"

    if [[ "$PERF_RUN" == "OK" && "$HASH_RESULT" == "OK" && "$CHECK_RESULT" == "OK" ]]; then
        CLEAN_AND_EXIT 0
    fi

    # On failure: re-run the performance scenarios once with nsys and save the profile
    echo "Video reader checks failed; re-running the benchmark with nsys profiling..."
    ARTIFACTS_DIR="${topdir}/core_artifacts"
    mkdir -p "${ARTIFACTS_DIR}"

    NSYS_REP="enabled" run_bench "${LOG_PERF}" ${TASKSET} python video_reader_bench.py \
        --repeat 1 --min_time 3 -j ${THREADS_LOW},${THREADS_HIGH} --readers 1,4 \
        -s ${PERF_SCENARIOS} || true

    cp -f *.nsys-rep ${JSON_PERF} "${ARTIFACTS_DIR}/" 2>/dev/null || true
    echo "nsys profile and results saved to ${ARTIFACTS_DIR}"
    CLEAN_AND_EXIT 1
}
pushd ../..
source ./qa/test_template.sh
popd
