#!/bin/bash -e
# used pip packages
pip_packages="numpy paddle"
target_dir=./docs/examples/use_cases/paddle/ssd/

test_body() {
    python train.py --check-loss-steps=300 -b 8 /data/coco/coco-2017/coco2017/
}

pushd ../..
source ./qa/test_template.sh
popd
