#!/bin/bash -e

test_body() {
    export TMP_PATH="$(mktemp -d)"
    mkdir -p /tmp/custom_cpp_lib
    git clone https://github.com/NVIDIA/DALI /tmp/dali
    pushd /tmp/custom_cpp_lib
    echo "
#include <dali/c_api.h>
int main() {
  daliInitialize();
  daliInitOperators();
}
" > main_stub.cc
    echo "
cmake_minimum_required(VERSION 3.13)
project(CustomCppLib)
include(/tmp/dali/tools/find_dali.cmake)
add_library(test_lib SHARED main_stub.cc)
find_dali(DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)
target_include_directories(test_lib ${DALI_INCLUDE_DIR})
target_link_directories(test_lib ${DALI_LIB_DIR})
target_link_libraries(test_lib ${DALI_LIBRARIES})
" > CMakeLists.txt
    mkdir build
    pushd build
    cmake ..
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
    popd
    popd
}

pushd ../..
source ./qa/test_template.sh
popd
