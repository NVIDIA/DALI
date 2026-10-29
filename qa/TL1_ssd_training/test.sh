#!/bin/bash -e
# used pip packages
pip_packages='numpy pillow torch torchvision mlperf_compliance matplotlib Cython pycocotools'
target_dir=./docs/examples/use_cases/pytorch/single_stage_detector/

test_body() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    prepare_coco_dataset
    torchrun --nproc_per_node=${NUM_GPUS} main.py --backbone resnet50 --warmup 300 --bs 256 --eval-batch-size 8 --epochs 4 --data ${DATA_DIR} --data_pipeline dali --target 0.085
    ((IS_TMP_DIR)) && rm -rf ${DATA_DIR} || true
}

pushd ../..
source ./qa/test_template.sh
popd
