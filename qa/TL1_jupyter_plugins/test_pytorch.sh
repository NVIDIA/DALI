#!/bin/bash -e

# used pip packages
# nvidia-index provides a stub for tensorboard which collides with one required by pytorch-lightning
# pin version which is not replaced
pip_packages="pillow jupyter matplotlib torchvision torch pytorch-lightning tensorboard==2.2.2"
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
