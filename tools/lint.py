#!/usr/bin/env python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import itertools
import re
import argparse
import sys

# Linter script, that calls cpplint.py, specifically for DALI repo.
# This will be called in `make lint` cmake target

# Q: How to configure git hook for pre-push linter check?
# A: Create a file `.git/hooks/pre-push`:
#
# #!/bin/sh
# DALI_ROOT_DIR=$(git rev-parse --show-toplevel)
# python $DALI_ROOT_DIR/tools/lint.py $DALI_ROOT_DIR
# ret=$?
# if [ $ret -ne 0 ]; then
#     exit 1
# fi
# exit 0


# Specifies, which files are to be excluded
# These filters are regexes, not typical unix-like path specification
negative_filters = [
    ".*core/dynlink_cuda.cc",
    ".*operators/reader/nvdecoder/nvcuvid.h",
    ".*operators/reader/nvdecoder/cuviddec.h",
    ".*operators/reader/nvdecoder/dynlink_nvcuvid.cc",
    ".*operators/reader/nvdecoder/dynlink_nvcuvid.h",
    ".*dali/core/dynlink_cuda.h",
    ".*operators/generic/transpose/cutt/cutt.h",
    ".*operators/generic/transpose/cutt/cutt.cc",
    ".*operators/generic/transpose/cutt/cuttplan.h",
    ".*operators/generic/transpose/cutt/cuttplan.cc",
    ".*operators/generic/transpose/cutt/cuttkernel.cu",
    ".*operators/generic/transpose/cutt/cuttkernel.h",
    ".*operators/generic/transpose/cutt/calls.h",
    ".*operators/generic/transpose/cutt/cuttGpuModel.h",
    ".*operators/generic/transpose/cutt/cuttGpuModel.cc",
    ".*operators/generic/transpose/cutt/cuttGpuModelKernel.h",
    ".*operators/generic/transpose/cutt/cuttGpuModelKernel.cu",
    ".*operators/generic/transpose/cutt/CudaMemcpy.h",
    ".*operators/generic/transpose/cutt/CudaMemcpy.cu",
    ".*operators/generic/transpose/cutt/CudaUtils.h",
    ".*operators/generic/transpose/cutt/CudaUtils.cu",
    ".*operators/generic/transpose/cutt/cuttTypes.h",
    ".*operators/generic/transpose/cutt/int_vector.h",
    ".*operators/generic/transpose/cutt/LRUCache.h",
    ".*python/dummy.cu"
]


def negative_filtering(patterns: list, file_list):
    """
    Patterns shall be a list of regex patterns
    """
    if len(patterns) == 0:
        return file_list
    prog = re.compile(patterns.pop())
    it = (i for i in file_list if not prog.search(i))
    return negative_filtering(patterns, it)


def gather_files(path: str, patterns: list, antipatterns: list):
    """
    Gather files, based on `path`, that match `patterns` unix-like specification
    and do not match `antipatterns` regexes
    """
    curr_path = os.getcwd()
    os.chdir(path)
    positive_iterators = [glob.iglob(os.path.join('**', pattern), recursive=True) for pattern in
                          patterns]
    linted_files = itertools.chain(*positive_iterators)
    linted_files = (os.path.join(path, file) for file in linted_files)
    linted_files = negative_filtering(antipatterns.copy(), linted_files)
    ret = list(linted_files)
    os.chdir(curr_path)
    return ret


def gen_cmd(dali_root_dir, file_list, process_includes=False):
    """
    Command for calling cpplint.py
    """
    return "python " + \
           os.path.join(dali_root_dir, "third_party", "cpplint.py") + \
           " --quiet --linelength=100 --root=" + \
           os.path.join(dali_root_dir, "include" if process_includes else "") + \
           " " + ' '.join(file_list)


def main(dali_root_dir):
    cc_files = gather_files(os.path.join(dali_root_dir, "dali"),
                            ["*.cc", "*.h", "*.cu", "*.cuh"], negative_filters)
    inc_files = gather_files(os.path.join(dali_root_dir, "include"),
                             ["*.h", "*.cuh", "*.inc", "*.inl"], negative_filters)
    cc_code = os.system(gen_cmd(dali_root_dir=dali_root_dir, file_list=cc_files, process_includes=False))
    inc_code = os.system(gen_cmd(dali_root_dir=dali_root_dir, file_list=inc_files, process_includes=True))
    if cc_code != 0 or inc_code != 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run linter check for DALI files")
    parser.add_argument('dali_root_path', type=str,
                        help='Root path of DALI repository (pointed directory should contain `.git` folder)')
    args = parser.parse_args()
    main(str(args.dali_root_path))
