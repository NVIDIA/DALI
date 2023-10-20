#!/bin/bash -e
# used pip packages
pip_packages='numpy pillow torch torchvision mlperf_compliance matplotlib<3.5.3 Cython pycocotools'
target_dir=./docs/examples/use_cases/pytorch/single_stage_detector/

test_body() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    torchrun --nproc_per_node=${NUM_GPUS} main.py --backbone resnet50 --warmup 300 --bs 64 --eval-batch-size 8 --epochs 4 --data /data/coco/coco-2017/coco2017/ --data_pipeline dali --target 0.085
}

pushd ../..
source ./qa/test_template.sh
popd
