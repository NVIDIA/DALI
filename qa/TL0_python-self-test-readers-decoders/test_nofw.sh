#!/bin/bash -e
# used pip packages
# moto + flask provide the mock S3 server used by reader/test_s3.py (plain `moto` does not pull
# flask in - that lives in the moto[server] extra, which also drags in cfn-lint, docker and more),
# boto3 seeds the bucket
pip_packages='${python_test_runner_package} numpy librosa scipy nvidia-ml-py==11.450.51 psutil dill cloudpickle pillow opencv-python-headless astropy av lmdb moto==5.2.3 flask flask-cors boto3'

target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
  test_no_fw
}

pushd ../..
source ./qa/test_template.sh
popd
