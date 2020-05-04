#!/bin/bash -e

test_body() {
    mkdir -p /tmp/repo_that_uses_dali
    pushd /tmp/repo_that_uses_dali
    git init
    git submodule add https://github.com/NVIDIA/DALI
    pushd DALI
    git submodule sync --recursive
    git submodule update --init --recursive
    popd
    touch main_stub.cc
    echo "
cmake_minimum_required(VERSION 3.13)
project(UsingDali)

add_subdirectory(DALI)

add_library(test_lib SHARED main_stub.cc)
target_link_libraries(test_lib PUBLIC DALI::dali DALI::dali_core DALI::dali_kernels DALI::dali_operators)
" > CMakeLists.txt
    mkdir build
    pushd build
    cmake ..
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
}

pushd ../..
source ./qa/test_template.sh
popd
