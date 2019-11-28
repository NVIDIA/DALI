#!/bin/bash -e

# used pip packages
pip_packages="jupyter matplotlib torchvision torch"
target_dir=./docs/examples

do_once() {
  mkdir -p idx_files
}

test_body() {
  # dummy
  black_list_files="#"

  # test code
  find pytorch/* -name "*.ipynb" | sed "/${black_list_files}/d" | xargs -i jupyter nbconvert \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600 {}
  python${PYVER:0:1} pytorch/resnet50/main.py -t
}

pushd ../..
source ./qa/test_template.sh
popd
