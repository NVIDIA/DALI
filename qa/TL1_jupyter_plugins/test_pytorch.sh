#!/bin/bash -e

# used pip packages
pip_packages='pillow jupyter matplotlib torchvision torch fsspec==2023.1.0 pytorch-lightning tensorboard'
target_dir=./docs/examples/

do_once() {
  mkdir -p idx_files
}

test_body() {
  # dummy
  exclude_files="#"
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

  # test code
  find frameworks/pytorch/ -name "*.ipynb" | sed "/${exclude_files}/d" | xargs -i jupyter nbconvert \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600 {}
}

pushd ../..
source ./qa/test_template.sh
popd
