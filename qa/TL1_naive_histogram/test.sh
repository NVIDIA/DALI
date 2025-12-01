#!/bin/bash -e

pip_packages='${python_test_runner_package} numpy'

do_once() {
  apt-get update && apt-get -y install wget
  wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-Linux-x86_64.sh
  bash cmake-*.sh --skip-license --prefix=/usr
  rm cmake-*.sh
}

test_body() {
    pushd $(pwd)/docs/examples/custom_operations/custom_operator/naive_histogram
    (mkdir build && cd build && cmake .. && make -j"$(grep ^processor /proc/cpuinfo | wc -l)")
    ${python_invoke_test} test_naive_histogram.py
    popd
}

pushd ../..
source ./qa/test_template.sh
popd
