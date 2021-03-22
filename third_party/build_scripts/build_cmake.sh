#!/bin/bash -xe

# CMake
pushd third_party/CMake
./bootstrap --parallel=$(grep ^processor /proc/cpuinfo | wc -l)
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
make install
popd
