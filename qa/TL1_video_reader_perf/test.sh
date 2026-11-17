#!/bin/bash -e
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Per-platform benchmark settings, keyed by "<CPU arch>/<GPU compute capability>".
set_platform_params() {
    local arch compute_cap
    arch=$(uname -m)
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0)
    case "${arch}/${compute_cap}" in
        x86_64/*)  # x86 + any GPU (tuned on Hopper)
            CPU_LIST="0-127"
            ;;
        aarch64/10.7)  # Vera Rubin
            # 2 sockets x 88 cores x 2 threads; 0-87 + 176-263 are socket 0 and its hyperthread
            # siblings (local to GPU 0)
            CPU_LIST="0-87,176-263"
            ;;
        aarch64/9.0|aarch64/10.0|aarch64/10.3)  # Grace Hopper (GH200), Grace Blackwell (GB200, GB300)
            CPU_LIST="0-71"
            ;;
        *)
            echo "Unsupported platform: got ${arch}/${compute_cap}, expected x86_64 with any GPU, or aarch64 with GPU compute capability 9.0, 10.0, 10.3 or 10.7." >&2
            CLEAN_AND_EXIT 1
            ;;
    esac
}

test_body() {
    set_platform_params
    TASKSET="taskset --cpu-list ${CPU_LIST}"
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
