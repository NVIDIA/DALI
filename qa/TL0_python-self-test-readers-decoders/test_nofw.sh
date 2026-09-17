#!/bin/bash -e
# used pip packages
# moto + flask provide the mock S3 server used by reader/test_s3.py (plain `moto` does not pull
# flask in - that lives in the moto[server] extra, which also drags in cfn-lint, docker and more),
# boto3 seeds the bucket.
# flask is pinned to 3.0 on purpose: 3.1 requires blinker>=1.9, and upgrading the blinker that
# the base image installs through apt fails with `uninstall-no-record-file`. flask 3.0 asks for
# blinker>=1.6.2, which the preinstalled one already satisfies, so nothing is uninstalled.
pip_packages='${python_test_runner_package} numpy librosa scipy nvidia-ml-py==11.450.51 psutil dill cloudpickle pillow opencv-python-headless astropy av lmdb moto==5.2.3 flask==3.0.3 flask-cors boto3'

target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
  test_no_fw
}

pushd ../..
source ./qa/test_template.sh
popd
