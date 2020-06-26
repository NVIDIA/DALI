#!/bin/bash -e
# used pip packages
# lock numba version as 0.50 changed module location and librosa hasn't catched up in 7.2 yet
pip_packages="nose numpy librosa torch numba<=0.49"
target_dir=./dali/test/python

test_body() {
    nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_pytorch_operator.py
    nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_dltensor_operator.py
    nosetests --verbose test_torch_pipeline_rnnt.py
    nosetests --verbose test_external_source_pytorch_cpu.py
    nosetests --verbose test_external_source_pytorch_gpu.py
    nosetests --verbose test_external_source_pytorch_dlpack.py
    nosetests --verbose test_backend_impl_torch_dlpack.py
}

pushd ../..
source ./qa/test_template.sh
popd
