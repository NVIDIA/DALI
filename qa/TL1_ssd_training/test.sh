#!/bin/bash -e
# used pip packages
pip_packages="numpy pillow torch torchvision mlperf_compliance matplotlib Cython pycocotools"
target_dir=./docs/examples/use_cases/pytorch/single_stage_detector/

test_body() {
    apt-get update && apt-get install -y gcc-7 g++-7

    #install APEX
    git clone https://github.com/nvidia/apex
    pushd apex
    # get the lastest stable and tested APEX version
    git checkout 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
    export CUDA_HOME=/usr/local/cuda-$(python -c "import torch; print('.'.join(torch.version.cuda.split('.')[0:2]))")
    export CC=gcc-7
    export CXX=g++-7
    # this makes nvcc to use linked gcc-7 as a host compiler, not the default system one
    ln -s $(which gcc-7) ${CUDA_HOME}/bin/gcc
    ln -s $(which g++-7) ${CUDA_HOME}/bin/g++
    # build wheel first
    pip wheel -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
    # for some reason the `pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .` doesn't install
    # wheel but only builds the binaries with the setuptools>=50.0, so now we builds wheel explicitly and then install it
    pip install apex*.whl
    unset CUDA_HOME
    popd

    python -m torch.distributed.launch --nproc_per_node=8 main.py --backbone resnet50 --warmup 300 --bs 64 --eval-batch-size 8 --epochs 4 --data /data/coco/coco-2017/coco2017/ --data_pipeline dali --target 0.085
}

pushd ../..
source ./qa/test_template.sh
popd

