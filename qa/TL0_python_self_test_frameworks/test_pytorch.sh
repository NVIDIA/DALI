#!/bin/bash -e
# used pip packages
pip_packages="nose numpy librosa==0.8.1 torch psutil"
target_dir=./dali/test/python

test_body() {
    nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_pytorch_operator.py
    nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_dltensor_operator.py
    nosetests --verbose test_torch_pipeline_rnnt.py
    nosetests --verbose test_external_source_pytorch_cpu.py
    nosetests --verbose test_external_source_pytorch_gpu.py
    nosetests --verbose test_external_source_pytorch_dlpack.py
    nosetests --verbose test_external_source_parallel_pytorch.py
    nosetests --verbose test_backend_impl_torch_dlpack.py
    nosetests --verbose test_dali_fork_torch.py
    nosetests --verbose --attr 'pytorch' test_external_source_impl_utils.py
    nosetests --verbose --attr 'pytorch' test_pipeline_debug.py
}

pushd ../..
source ./qa/test_template.sh
popd
