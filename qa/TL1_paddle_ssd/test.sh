#!/bin/bash -e
# used pip packages
# rarfile>= 3.2 breaks python 3.5 compatibility
pip_packages="paddlepaddle-gpu"
target_dir=./docs/examples/use_cases/paddle/ssd/

test_body() {
    python train.py --check-loss-steps=300 -b 8 /data/coco/coco-2017/coco2017/
}

pushd ../..
source ./qa/test_template.sh
popd
