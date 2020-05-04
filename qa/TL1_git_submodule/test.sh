#!/bin/bash -e

test_body() {
    mkdir -p /tmp/repo_that_uses_dali
    cd /tmp/repo_that_uses_dali
    git init
    git submodule add -b dali_as_submodule https://github.com/AlexBula/DALI
    cd DALI
    git submodule sync --recursive
    git submodule update --init --recursive
    cd -
    echo "
cmake_minimum_required(VERSION 3.13)
project(UsingDali)

add_subdirectory(DALI)
" > CMakeLists.txt
    mkdir build
    cd build
    cmake ..
    make 
}

pushd ../..
source ./qa/test_template.sh
popd
