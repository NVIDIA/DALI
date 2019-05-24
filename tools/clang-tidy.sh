#!/bin/bash -ex

# set header filter
HEADER_FILTER=".*/dali/pipeline/.*|.*dali/image/.*|.*dali/kernels/.*|.*/include/dali/.*"

# Path to clang build
DALI_ROOT=$(pwd)
DALI_BUILD_PATH=${DALI_BUILD_PATH:-"${DALI_ROOT}/build_clang"}
DALI_FORMAT_PATH=${DALI_FORMAT_PATH:-"${DALI_ROOT}/.clang-format"}
DALI_SOURCES=${DALI_SOURCES:-"${DALI_ROOT}/dali/.*"}

RUN_CLANG_TIDY=${RUN_CLANG_TIDY:-"run-clang-tidy-6.0.py"}


checks_filters=(
    "performance-*,-performance-inefficient-vector-operation,-performance-noexcept-move-constructor,-performance-inefficient-string-concatenation"
    "modernize-*"
    "bugprone-*"
    "readability-*"
    "clang-analyzer-core*"
    "clang-analyzer-cplusplus*"
    "clang-analyzer-deadcode*"
    "clang-analyzer-nullability*"
    "clang-analyzer-unix*"
)

checks_names=(
    "performance"
    "modernize"
    "bugprone"
    "readability"
    "clang-analyzer-core"
    "clang-analyzer-cplusplus"
    "clang-analyzer-deadcode"
    "clang-analyzer-nullability"
    "clang-analyzer-unix"
)

for i in ${!checks_filters[@]}
do
    check="${checks_filters[$i]}"
    CHECK_PATTERN="-*,${check}"
    FILE_OUTPUT="tidy-check-${checks_names[$i]}"
    ${RUN_CLANG_TIDY} -checks=${CHECK_PATTERN} \
        -header-filter ${HEADER_FILTER} \
        -p ${DALI_BUILD_PATH} \
        -extra-arg format-style=${DALI_FORMAT_PATH} "${DALI_SOURCES}" > ${FILE_OUTPUT}
done

