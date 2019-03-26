#!/bin/bash

# Based on https://clang.llvm.org/docs/SanitizerCoverage.html

mkdir -p build_coverage_sanitizer
pushd build_coverage_sanitizer
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_LMDB=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCOVERAGE_SANITIZER=ON ..
make -j"$(nproc)"

# Run
ASAN_OPTIONS=protect_shadow_gap=0:detect_odr_violation=0:coverage=1 ./dali/python/nvidia/dali/test/dali_test.bin
# Symbolize
# TODO get the name of generated file
# sancov-6.0 -symbolize dali_test.bin.8991.sancov ./dali/python/nvidia/dali/test/dali_test.bin > dali_test.bin.symcov

# Get the coverage-report-server.py from https://github.com/llvm-mirror/llvm/blob/master/tools/sancov/coverage-report-server.py
# Run the server to analize
# coverage-report-server.py --symcov dali_test.bin.symcov --srcpath ..

popd