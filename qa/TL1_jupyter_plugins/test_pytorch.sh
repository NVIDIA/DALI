#!/bin/bash -e

# used pip packages
pip_packages="pillow jupyter matplotlib torchvision torch pytorch-lightning"
target_dir=./docs/examples/

do_once() {
  mkdir -p idx_files
}

test_body() {
  # dummy
  exclude_files="#"

  # test code
  find frameworks/pytorch/ -name "*.ipynb" | sed "/${exclude_files}/d" | xargs -i jupyter nbconvert \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600 {}
  python${PYVER:0:1} use_cases/pytorch/resnet50/main.py -t
}

pushd ../..
source ./qa/test_template.sh
popd
