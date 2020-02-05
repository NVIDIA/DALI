#!/bin/bash -e
# used pip packages
# Fixing numpy to 1.17.0 version to avoid the error about not being able to implicitly convert from float64 to integer
# TODO(janton): remove explicit pillow version installation when torch fixes the issue with PILLOW_VERSION not being defined
pip_packages="numpy==1.17.0 pillow==6.2.2 torch torchvision mlperf_compliance matplotlib Cython"
target_dir=./docs/examples/use_cases/pytorch/single_stage_detector/

test_body() {
    # workaround due to errors while pycocotools is int "pip_packages" above
    pip install -I pycocotools

    #install APEX
    git clone https://github.com/nvidia/apex
    pushd apex
    export CUDA_HOME=/usr/local/cuda-$(python -c "import torch; print('.'.join(torch.version.cuda.split('.')[0:2]))")
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
    unset CUDA_HOME
    popd

    python -m torch.distributed.launch --nproc_per_node=8 main.py --backbone resnet50 --warmup 300 --bs 64 --eval-batch-size 8 --epochs 4 --data /data/coco/coco-2017/coco2017/ --data_pipeline dali --target 0.085
}

pushd ../..
source ./qa/test_template.sh
popd

