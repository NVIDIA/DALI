#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy paddlepaddle-gpu'
target_dir=./dali/test/python

one_config_only=true

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    # Paddle 3.4's libpir reports an external alloc/dealloc mismatch during ASan teardown.
    # Keep the framework coverage in regular CI, but skip it under sanitizers until Paddle fixes it.
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        for fw in "paddle"; do
            python test_RN50_data_fw_iterators.py --framework ${fw} --gpus ${NUM_GPUS} -b 13 \
                --workers 3 --prefetch 2 -i 100 --epochs 2
            python test_RN50_data_fw_iterators.py --framework ${fw} --gpus ${NUM_GPUS} -b 13 \
                --workers 3 --prefetch 2 -i 2 --epochs 2 --fp16
        done
        ${python_new_invoke_test} -A 'paddle' test_fw_iterators_detection
        ${python_new_invoke_test} -A 'paddle' test_fw_iterators
    fi
}

pushd ../..
source ./qa/test_template.sh
popd
