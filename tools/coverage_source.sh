#!/bin/bash -ex

# Based on https://clang.llvm.org/docs/SourceBasedCodeCoverage.html

mkdir -p build_coverage_source
pushd build_coverage_source
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_LMDB=ON -DCMAKE_BUILD_TYPE=Debug -DCOVERAGE_SOURCE=ON ..
make -j$(nproc)

export DALI_EXTRA_PATH=dali_extra
source ../qa/setup_dali_extra.sh

# Run, and allow it to fail
LLVM_PROFILE_FILE="dali_test.profraw" ./dali/python/nvidia/dali/test/dali_test.bin || true
# Index the raw profile.
llvm-profdata merge -sparse dali_test.profraw -o dali_test.profdata
# Create file and function level reports
llvm-cov report ./dali/python/nvidia/dali/test/dali_test.bin -instr-profile=dali_test.profdata > cov-report
llvm-cov report ./dali/python/nvidia/dali/test/dali_test.bin -show-functions -instr-profile=dali_test.profdata -Xdemangler=llvm-cxxfilt .. > cov-report-functions
# Create a line-oriented coverage report
llvm-cov show -format html ./dali/python/nvidia/dali/test/dali_test.bin -instr-profile=dali_test.profdata -Xdemangler=llvm-cxxfilt > cov-src.html

popd