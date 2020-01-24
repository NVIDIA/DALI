#!/bin/bash -e

# used pip packages
# TODO(janton): remove explicit pillow version installation when torch fixes the issue with PILLOW_VERSION not being defined
pip_packages="pillow==6.2.2 jupyter matplotlib torchvision torch"
target_dir=./docs/examples/

do_once() {
  mkdir -p idx_files
}

test_body() {
  # dummy
  black_list_files="#"

  # test code
  find frameworks/pytorch/ -name "*.ipynb" | sed "/${black_list_files}/d" | xargs -i jupyter nbconvert \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600 {}
  python${PYVER:0:1} use_cases/pytorch/resnet50/main.py -t
}

pushd ../..
source ./qa/test_template.sh
popd
