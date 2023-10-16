#!/bin/bash -e

# used pip packages
# to be run inside a MXNet container - so don't need to list it here as a pip package dependency
pip_packages='jupyter matplotlib<3.5.3'
target_dir=./docs/examples/

do_once() {
  mkdir -p idx_files
}

test_body() {
  # dummy
  exclude_files="#"

  # test code
  find frameworks/mxnet -name "*.ipynb" | sed "/${exclude_files}/d" | xargs -i jupyter nbconvert \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600 {}
  jupyter nbconvert --to notebook --inplace --execute \
                    --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                    --ExecutePreprocessor.timeout=600 use_cases/mxnet/mxnet-resnet50.ipynb
}

pushd ../..
source ./qa/test_template.sh
popd
