#!/bin/bash -e
# used pip packages
# moto + flask provide the mock S3 server used by reader/test_s3.py (plain `moto` does not pull
# flask in - that lives in the moto[server] extra, which also drags in cfn-lint, docker and more),
# boto3 seeds the bucket. Pin matches TL0_python-self-test-readers-decoders/test_nofw.sh.
pip_packages='${python_test_runner_package} dataclasses numpy opencv-python-headless pillow librosa scipy nvidia-ml-py==11.450.51 numba lz4 psutil dill cloudpickle astropy av moto==5.2.3 flask==3.0.3 flask-cors boto3'
target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {
  test_no_fw
}

pushd ../..
source ./qa/test_template.sh
popd
