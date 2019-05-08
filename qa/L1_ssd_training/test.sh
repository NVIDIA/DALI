#!/bin/bash -e
# used pip packages
pip_packages="numpy torch torchvision mlperf_compliance matplotlib Cython"

pushd ../..

cd docs/examples/pytorch/single_stage_detector/

test_body() {
    # workaround due to errors while pycocotools is int "pip_packages" above
    pip install -I pycocotools

    #install APEX
    git clone https://github.com/nvidia/apex
    pushd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
    popd

    python main.py --backbone resnet50 --warmup 300 --bs 64 --data /data/coco/coco-2017/coco2017/ --data_pipeline dali --epochs=4 --target 0.1
}

source ../../../../qa/test_template.sh

popd
