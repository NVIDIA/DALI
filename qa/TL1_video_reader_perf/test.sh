#!/bin/bash -e
# used pip packages (the reference libraries are installed by install_reference_libs)
pip_packages='numpy nvidia-ml-py==11.450.51'
target_dir=./internal_tools

# One-time pre-step: FFmpeg shared libraries for torchcodec (x86_64 only, see
# install_reference_libs), and nsys (NVIDIA Nsight Systems)
do_once() {
    apt update
    if [ "$(uname -m)" == "x86_64" ]; then
        # torchcodec loads the FFmpeg shared libraries and links against the shared libpython
        apt install -y --no-install-recommends ffmpeg \
            "libpython$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
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
JSON_HASH="video_reader_hash.json"
LOG_REF="video_reader_ref.log"
JSON_REF="video_reader_ref.json"

# DALI's legacy fn.readers.video, its video decoder (fn.readers.file + fn.decoders.video) and
# other libraries run for reference on the same samples
# (video_reader_bench.py --backends), on x86_64 only; one short run each, not checked. The DALI
# numbers are in the performance run. The libraries are installed by install_reference_libs.
REF_SCENARIOS="lanes,single_stream,cache_key,cpu"
REF_ARGS="--repeat 1 --iters 4 --warmup 1 --min_time 2 -j 1,16"

# Installs PyNvVideoCodec, and torch with the matching CUDA-enabled torchcodec from the PyTorch
# index for this CUDA version (x86_64 only), and sets REF_BACKENDS. torch is needed only here,
# so it is pinned here rather than in pip_packages, which would follow the DALI framework tests.
install_reference_libs() {
    local cuda_tag
    case "${CUDA_VERSION}" in
        12*) cuda_tag=cu129 ;;
        13*) cuda_tag=cu130 ;;
        *) echo "No PyTorch index known for CUDA_VERSION=${CUDA_VERSION}; update test.sh"; return 1 ;;
    esac
    pip install PyNvVideoCodec==2.0.2 || return 1
    pip install "torch==2.11.0+${cuda_tag}" "torchcodec==0.15.0+${cuda_tag}" \
        --index-url "https://download.pytorch.org/whl/${cuda_tag}" || return 1
    REF_BACKENDS="pynvvideocodec,torchcodec,torchcodec_nvdec"
    echo "Reference backends: ${REF_BACKENDS}"
}

# Scenarios of internal_tools/video_reader_bench.py run for performance; see its --help
PERF_SCENARIOS="lanes,engines,single_stream,reader_depth,cache_key,seq_longgop,random_access,cpu"
# Scenarios whose output is checked to be independent of num_threads and reader_depth
HASH_SCENARIOS="lanes,reader_depth,cache_key,cpu,seq_longgop,random_access"
BENCH_ARGS="--repeat 3 --iters 8 --min_time 3 -j 1,4,16 --readers 1,2,4"
# engines: one H100 or GH200 reaches 3.6x with 4 readers (several NVDEC engines)
ENGINES_SCALING=2.5

function CLEAN_AND_EXIT {
    rm -rf ${LOG_PERF} ${JSON_PERF} ${JSON_HASH} ${LOG_REF} ${JSON_REF}
    exit $1
}

# Prints the results (without per-run data) to the log, so that they are kept on success too.
# A missing or unreadable file (e.g. no reference run on GraceHopper) is skipped, not an error.
print_results() {
    for json in "$@"; do
        if [ -f "${json}" ]; then
            python -c "import json, sys
d = json.load(open(sys.argv[1]))
for r in d['results']:
    r.pop('runs', None)
print('RESULTS', sys.argv[1], json.dumps(d, separators=(',', ':')))" "${json}" \
                || echo "Could not print ${json}"
        fi
    done
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
        TASKSET="taskset --cpu-list 0-127"  # Hopper
    else
        TASKSET="taskset --cpu-list 0-71"  # GraceHopper
    fi
    # Absolute minimums in frames/s (see qa/TL1_video_reader_perf/check_results.py); empty:
    # reported for reference only, not checked.
    # TODO: set them per architecture (e.g. 0.85x the REFERENCE values) once a few CI runs
    # establish the baseline on these machines
    MIN_LANES=
    MIN_SINGLE_STREAM=
    MIN_CPU=
    MIN_ENGINES=

    unset NSYS_REP
    PERF_RUN="OK"
    run_bench "${LOG_PERF}" ${TASKSET} python video_reader_bench.py ${BENCH_ARGS} \
        -s ${PERF_SCENARIOS} --json ${JSON_PERF} || PERF_RUN="FAIL"

    # The output must not depend on num_threads or on the reader's prefetch_queue_depth: every
    # configuration is compared with all that differ from it only in those
    HASH_RESULT="OK"
    ${TASKSET} python video_reader_bench.py -s ${HASH_SCENARIOS} -j 1,16 --hash \
        --json ${JSON_HASH} || HASH_RESULT="FAIL"
    python video_reader_bench.py --compare ${JSON_HASH} ${JSON_HASH} \
        --ignore threads,reader_depth || HASH_RESULT="FAIL"

    CHECK_RESULT="OK"
    python ${topdir}/qa/TL1_video_reader_perf/check_results.py ${JSON_PERF} --tolerance 10 \
        --engines_scaling ${ENGINES_SCALING} \
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
            -s ${REF_SCENARIOS} -b dali_legacy,dali_decoder,${REF_BACKENDS} --json ${JSON_REF} || REF_RUN="FAIL"
    else
        REF_RUN="FAIL"
    fi

    echo "PERF_RUN=${PERF_RUN}"
    echo "HASH_RESULT=${HASH_RESULT}"
    echo "CHECK_RESULT=${CHECK_RESULT}"
    echo "REF_RUN=${REF_RUN}"
    print_results ${JSON_PERF} ${JSON_REF}

    if [[ "$PERF_RUN" == "OK" && "$HASH_RESULT" == "OK" && "$CHECK_RESULT" == "OK" \
          && "$REF_RUN" != "FAIL" ]]; then
        CLEAN_AND_EXIT 0
    fi

    # On failure: save the results; if the performance run or its checks failed, also re-run the
    # performance scenarios once with nsys and save the profile
    ARTIFACTS_DIR="${topdir}/core_artifacts"
    mkdir -p "${ARTIFACTS_DIR}"
    if [[ "$PERF_RUN" != "OK" || "$CHECK_RESULT" != "OK" ]]; then
        echo "Video reader performance checks failed; re-running with nsys profiling..."
        NSYS_REP="enabled" run_bench "${LOG_PERF}" ${TASKSET} python video_reader_bench.py \
            --repeat 1 --min_time 3 -j 1,16 --readers 1,4 -s ${PERF_SCENARIOS} || true
    fi
    cp -f *.nsys-rep ${JSON_PERF} ${JSON_HASH} ${JSON_REF} "${ARTIFACTS_DIR}/" 2>/dev/null || true
    echo "Results (and nsys profile, if any) saved to ${ARTIFACTS_DIR}"
    CLEAN_AND_EXIT 1
}
pushd ../..
source ./qa/test_template.sh
popd
