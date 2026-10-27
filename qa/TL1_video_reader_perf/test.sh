#!/bin/bash -e
# used pip packages (torch and torchcodec are installed by install_torchcodec)
pip_packages='numpy'
target_dir=./internal_tools

# One-time pre-step: the FFmpeg shared libraries and the shared libpython that torchcodec needs
# (x86_64 only). The torchcodec rows are for reference, so a failure here is not an error.
do_once() {
    if [ "$(uname -m)" == "x86_64" ]; then
        (apt update && apt install -y --no-install-recommends ffmpeg \
            "libpython$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')") \
            || echo "Could not install FFmpeg: no torchcodec rows"
    fi
}

LOG_PERF="video_reader_perf.log"
JSON_PERF="video_reader_perf.json"
LOG_HASH="video_reader_hash.log"
JSON_HASH="video_reader_hash.json"

function CLEAN_AND_EXIT {
    rm -rf ${LOG_PERF} ${JSON_PERF} ${LOG_HASH} ${JSON_HASH}
    exit $1
}

# Installs torch and the matching CUDA-enabled torchcodec from the PyTorch index for this CUDA
# version (x86_64 only). torch is needed only here, so it is pinned here rather than in
# pip_packages, which would follow the DALI framework tests. A failure is not an error: the
# benchmark then runs without the torchcodec rows (--no_torchcodec).
install_torchcodec() {
    local cuda_tag
    case "${CUDA_VERSION}" in
        12*) cuda_tag=cu129 ;;
        13*) cuda_tag=cu130 ;;
        *) echo "No PyTorch index known for CUDA_VERSION=${CUDA_VERSION}"; return 1 ;;
    esac
    [ "$(uname -m)" == "x86_64" ] || return 1
    pip install "torch==2.11.0+${cuda_tag}" "torchcodec==0.15.0+${cuda_tag}" \
        --index-url "https://download.pytorch.org/whl/${cuda_tag}" && python -c "import torchcodec"
}

test_body() {
    if [ "$(uname -p)" == "x86_64" ]; then
        TASKSET="taskset --cpu-list 0-127"  # Hopper
    else
        TASKSET="taskset --cpu-list 0-71"  # GraceHopper
    fi
    TORCHCODEC_ARGS=""
    install_torchcodec || { echo "No torchcodec rows"; TORCHCODEC_ARGS="--no_torchcodec"; }

    # Throughput of every configuration. Failing: a configuration that fails, or more threads
    # (or a deeper reader queue) that are more than 10% slower than fewer. The torchcodec rows
    # are only reported.
    PERF_RUN="OK"
    ${TASKSET} python video_reader_bench.py --check ${TORCHCODEC_ARGS} --json ${JSON_PERF} \
        | tee ${LOG_PERF} || PERF_RUN="FAIL"

    # The output must not depend on num_threads or on the reader's prefetch_queue_depth: the
    # configurations of a group are compared with each other
    HASH_RUN="OK"
    ${TASKSET} python video_reader_bench.py --hash --json ${JSON_HASH} \
        | tee ${LOG_HASH} || HASH_RUN="FAIL"

    echo "PERF_RUN=${PERF_RUN}"
    echo "HASH_RUN=${HASH_RUN}"
    for json in ${JSON_PERF} ${JSON_HASH}; do
        [ -f ${json} ] && echo "RESULTS ${json} $(tr -d ' \n' < ${json})"
    done

    if [[ "$PERF_RUN" == "OK" && "$HASH_RUN" == "OK" ]]; then
        CLEAN_AND_EXIT 0
    fi

    # On failure: save the logs and results
    ARTIFACTS_DIR="${topdir}/core_artifacts"
    mkdir -p "${ARTIFACTS_DIR}"
    cp -f ${LOG_PERF} ${JSON_PERF} ${LOG_HASH} ${JSON_HASH} "${ARTIFACTS_DIR}/" 2>/dev/null || true
    echo "Results saved to ${ARTIFACTS_DIR}"
    CLEAN_AND_EXIT 1
}
pushd ../..
source ./qa/test_template.sh
popd
