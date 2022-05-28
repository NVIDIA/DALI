#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy mxnet'
target_dir=./dali/test/python

one_config_only=true

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    # it takes very long time to run it with sanitizers on and provides little value so turn it off
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        for fw in "mxnet"; do
            python test_RN50_data_fw_iterators.py --framework ${fw} --gpus ${NUM_GPUS} -b 13 \
                --workers 3 --prefetch 2 -i 100 --epochs 2
            python test_RN50_data_fw_iterators.py --framework ${fw} --gpus ${NUM_GPUS} -b 13 \
                --workers 3 --prefetch 2 -i 2 --epochs 2 --fp16
        done
    fi
    ${python_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*mxnet*' test_fw_iterators_detection.py
    ${python_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*mxnet*' test_fw_iterators.py
    ${python_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*gluon*' test_fw_iterators.py
}

pushd ../..
source ./qa/test_template.sh
popd
