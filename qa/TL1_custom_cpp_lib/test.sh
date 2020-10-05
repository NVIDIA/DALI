#!/bin/bash -e

do_once() {
  apt-get update
  apt-get -y install cmake
}

test_body() {
    export TMP_PATH="$(mktemp -d)"
    pushd $TMP_PATH
    echo "
#include <dali/c_api.h>
int main() {
  daliInitialize();
  daliInitOperators();
}
" > main_stub.cc
    echo "
cmake_minimum_required(VERSION 3.7)
project(CustomCppLib)
include(/opt/dali/tools/find_dali.cmake)
add_library(test_lib SHARED main_stub.cc)
find_dali(DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)
target_include_directories(test_lib \${DALI_INCLUDE_DIR})
link_directories(\${DALI_LIB_DIR})
target_link_libraries(test_lib \${DALI_LIBRARIES})
" > CMakeLists.txt
    cmake .
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
    popd
}

pushd ../..
source ./qa/test_template.sh
popd
