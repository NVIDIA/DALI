#!/bin/bash -e
# used pip packages
pip_packages="numpy Cython paddle"
target_dir=./docs/examples/paddle/ssd/

test_body() {
    # workaround due to errors while pycocotools is in "pip_packages" above
    pip install -I pycocotools
    python train.py --check-loss-steps=300 -b 8 /data/coco/coco-2017/coco2017/
}

pushd ../..
source ./qa/test_template.sh
popd
