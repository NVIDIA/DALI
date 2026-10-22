#!/bin/bash -e
# used pip packages; torch (pinned per CUDA version in qa/setup_packages.py) is needed by the
# reference libraries, see install_reference_libs
pip_packages='numpy nvidia-ml-py==11.450.51 torch'
target_dir=./internal_tools

# One-time pre-step: FFmpeg shared libraries for torchcodec (x86_64 only, see
# install_reference_libs), and nsys (NVIDIA Nsight Systems)
do_once() {
    apt update
    if [ "$(uname -m)" == "x86_64" ]; then
        apt install -y --no-install-recommends ffmpeg
    fi
    if command -v nsys &>/dev/null; then
        echo "nsys already installed: $(nsys --version)"
        return
    fi
    DISTRO=$(source /etc/lsb-release && echo "$DISTRIB_RELEASE" | tr -d .)
    ARCH=$(dpkg --print-architecture)
    apt install -y --no-install-recommends gnupg curl
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
LOG_REF="video_reader_ref.log"
JSON_REF="video_reader_ref.json"

# Other libraries run for reference on the same samples (video_reader_bench.py --backends), on
# x86_64 only; one run each, not checked against minimums. Installed by install_reference_libs.
REF_SCENARIOS="lanes,single_stream,cache_key,cpu"
REF_ARGS="--repeat 1 --min_time 3 -j 1,16"

# Installs PyNvVideoCodec and torchcodec (x86_64 only) and sets REF_BACKENDS. torchcodec must
# match the torch version installed through pip_packages, and is built for the same CUDA version.
install_reference_libs() {
    local torch_ver cuda_tag tc_ver
    torch_ver=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    cuda_tag=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
    case "${torch_ver}" in
        2.7.*) tc_ver=0.5 ;;
        2.11.*) tc_ver=0.11.1 ;;
        *) echo "No torchcodec version known for torch ${torch_ver}; update test.sh"; return 1 ;;
    esac
    pip install PyNvVideoCodec==2.0.2 || return 1
    # --no-deps: keep the torch from pip_packages
    pip install --no-deps "torchcodec==${tc_ver}+${cuda_tag}" \
        --index-url "https://download.pytorch.org/whl/${cuda_tag}" || return 1
    REF_BACKENDS="pynvvideocodec,torchcodec"
    # torchcodec 0.5 has no set_cuda_backend, i.e. no torchcodec_beta
    [ "${tc_ver}" != "0.5" ] && REF_BACKENDS="${REF_BACKENDS},torchcodec_beta"
    echo "Reference backends: ${REF_BACKENDS}"
}

# Scenarios of internal_tools/video_reader_bench.py run for performance; see its --help
PERF_SCENARIOS="lanes,engines,single_stream,reader_depth,cache_key,startup,seq_longgop,cpu"
# Scenarios whose output is checked to be independent of num_threads and reader_depth
HASH_SCENARIOS="lanes,reader_depth,cache_key,cpu"
THREADS_LOW=1
THREADS_HIGH=16
BENCH_ARGS="--repeat 3 --min_time 3 -j ${THREADS_LOW},4,${THREADS_HIGH} --readers 1,2,4"

function CLEAN_AND_EXIT {
    rm -rf ${LOG_PERF} ${JSON_PERF} ${JSON_HASH_LOW} ${JSON_HASH_HIGH} ${LOG_REF} ${JSON_REF}
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

    # The same samples decoded by other libraries, for reference (x86_64 only). A failure here
    # (installation or decoding) fails the test, so that the comparison does not silently
    # disappear.
    REF_RUN="OK"
    if [ "$(uname -m)" != "x86_64" ]; then
        REF_RUN="SKIPPED"
    elif install_reference_libs; then
        run_bench "${LOG_REF}" ${TASKSET} python video_reader_bench.py ${REF_ARGS} \
            -s ${REF_SCENARIOS} -b dali,${REF_BACKENDS} --json ${JSON_REF} || REF_RUN="FAIL"
    else
        REF_RUN="FAIL"
    fi

    echo "PERF_RUN=${PERF_RUN}"
    echo "HASH_RESULT=${HASH_RESULT}"
    echo "CHECK_RESULT=${CHECK_RESULT}"
    echo "REF_RUN=${REF_RUN}"

    if [[ "$PERF_RUN" == "OK" && "$HASH_RESULT" == "OK" && "$CHECK_RESULT" == "OK" \
          && "$REF_RUN" != "FAIL" ]]; then
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
