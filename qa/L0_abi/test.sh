#!/bin/bash -e

pushd ../..

# Run linter
cd build-docker-release
# Check if we export only dali::, _fini and _init symbols from libdali.so
nm -gC --defined-only ./lib/libdali.so | grep -v "dali::" | grep -i " t " | grep -vx ".*T dali.*" | grep -vx ".*T _fini" | grep -vxq ".*T _init" && exit 1
nm -gC --defined-only ./dali/python/nvidia/dali/plugin/libdali_tf.so  | grep -v "dali::" | grep -i " t " | grep -vx ".*T dali.*" | grep -vx ".*T _fini" | grep -vxq ".*T _init" && exit 1
echo "Done"

popd
