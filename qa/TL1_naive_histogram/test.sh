#!/bin/bash -e

pip_packages='${python_test_runner_package} numpy build scikit-build-core'

do_once() {
  apt-get update && apt-get -y install wget
  wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-Linux-x86_64.sh
  bash cmake-*.sh --skip-license --prefix=/usr
  rm cmake-*.sh
}

test_body() {
    # Build for the GPU the test runs on - the default architectures don't cover all GPUs
    export CUDAARCHS=native
    pushd $(pwd)/docs/examples/custom_operations/custom_operator/naive_histogram
    (mkdir build && cd build && cmake .. && make -j"$(grep ^processor /proc/cpuinfo | wc -l)")
    ${python_new_invoke_test} test_naive_histogram

    # Package the plugin as a source distribution and install it, as described in the README
    python -m build --sdist --no-isolation
    pip install --no-build-isolation --no-cache-dir dist/dali_naive_histogram-*.tar.gz
    (cd /tmp && python -c "import dali_naive_histogram; from nvidia.dali import fn; fn.naive_histogram")
    # The package refuses to load the plugin with a DALI version other than the one it was built with
    (cd /tmp && python -c "
import nvidia.dali as dali
dali.__version__ = 'mismatched'
try:
    import dali_naive_histogram
except RuntimeError as e:
    assert 'was built against DALI' in str(e), e
else:
    raise AssertionError('DALI version mismatch was not detected')
")
    pip uninstall -y dali-naive-histogram
    rm -rf build dist
    popd
}

pushd ../..
source ./qa/test_template.sh
popd
