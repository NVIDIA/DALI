#!/usr/bin/env python

# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import subprocess

# Linter script, that calls cpplint.py, specifically for DALI repo.
# This will be called in `make lint` cmake target

# Q: How to configure git hook for pre-push linter check?
# A: Create a file `.git/hooks/pre-push`:
#
# #!/bin/sh
# DALI_ROOT_DIR=$(git rev-parse --show-toplevel)
# python $DALI_ROOT_DIR/internal_tools/lint.py $DALI_ROOT_DIR --nproc=10
# ret=$?
# if [ $ret -ne 0 ]; then
#     exit 1
# fi
# exit 0


# Specifies, which files are to be excluded
# These filters are regexes, not typical unix-like path specification
negative_filters = [
    ".*operators/video/*",
    ".*operators/sequence/optical_flow/optical_flow_impl/nvOpticalFlowCuda.h",
    ".*operators/sequence/optical_flow/optical_flow_impl/nvOpticalFlowCommon.h",
    ".*python/dummy.cu",
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
    positive_iterators = [
        glob.iglob(os.path.join("**", pattern), recursive=True) for pattern in patterns
    ]
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
    if not file_list:
        return ["true"]
    cmd = [
        "python",
        os.path.join(dali_root_dir, "third_party", "cpplint.py"),
        "--quiet",
        "--linelength=100",
        "--headers=h,cuh",
        "--root=" + os.path.join(dali_root_dir, "include" if process_includes else ""),
    ]
    cmd.extend(file_list)
    return cmd


def lint(dali_root_dir, file_list, process_includes, n_subproc):
    """
    n_subprocesses: how many subprocesses to use for linter processing
    Returns: 0 if lint passed, 1 otherwise
    """
    if len(file_list) == 0:
        return 0
    cmds = []
    diff = int(len(file_list) / n_subproc)
    for process_idx in range(n_subproc - 1):
        cmds.append(
            gen_cmd(
                dali_root_dir=dali_root_dir,
                file_list=file_list[process_idx * diff : (process_idx + 1) * diff],
                process_includes=process_includes,
            )
        )
    cmds.append(
        gen_cmd(
            dali_root_dir=dali_root_dir,
            file_list=file_list[(n_subproc - 1) * diff :],
            process_includes=process_includes,
        )
    )
    subprocesses = [
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in cmds
    ]
    success = True
    for subproc in subprocesses:
        stdout, stderr = subproc.communicate()
        success *= not bool(subproc.poll())
        if len(stderr) > 0:
            print(stderr.decode("utf-8"))
    return 0 if success else 1


def main(dali_root_dir, n_subproc=1, file_list=None):
    cc_files = gather_files(
        os.path.join(dali_root_dir, "dali"),
        ["*.cc", "*.h", "*.cu", "*.cuh"] if file_list is None else file_list,
        negative_filters,
    )
    inc_files = gather_files(
        os.path.join(dali_root_dir, "include"),
        ["*.h", "*.cuh", "*.inc", "*.inl"] if file_list is None else file_list,
        negative_filters,
    )
    tf_plugin_files = gather_files(
        os.path.join(dali_root_dir, "dali_tf_plugin"),
        ["*.cc", "*.h", "*.cu", "*.cuh"] if file_list is None else file_list,
        negative_filters,
    )
    sdist_plugin_files = gather_files(
        os.path.join(dali_root_dir, "plugins"),
        ["*.cc", "*.h", "*.cu", "*.cuh"] if file_list is None else file_list,
        negative_filters,
    )

    cc_code = lint(
        dali_root_dir=dali_root_dir, file_list=cc_files, process_includes=False, n_subproc=n_subproc
    )
    inc_code = lint(
        dali_root_dir=dali_root_dir, file_list=inc_files, process_includes=True, n_subproc=n_subproc
    )

    tf_plugin_code = lint(
        dali_root_dir=dali_root_dir,
        file_list=tf_plugin_files,
        process_includes=False,
        n_subproc=n_subproc,
    )

    sdist_plugin_code = lint(
        dali_root_dir=dali_root_dir,
        file_list=sdist_plugin_files,
        process_includes=False,
        n_subproc=n_subproc,
    )

    if cc_code != 0 or inc_code != 0 or tf_plugin_code != 0 or sdist_plugin_code != 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run linter check for DALI files. Gather all code-files "
        "(h, cuh, cc, cu, inc, inl, py) and perform linter check on them."
    )
    parser.add_argument(
        "dali_root_path",
        type=str,
        help="Root path of DALI repository " "(pointed directory should contain `.git` folder)",
    )
    parser.add_argument(
        "--nproc", type=int, default=1, help="Number of processes to spawn for linter verification"
    )
    parser.add_argument(
        "--file-list", nargs="*", help="List of files. This overrides the default scenario"
    )
    args = parser.parse_args()
    assert args.nproc > 0
    main(str(args.dali_root_path), args.nproc, file_list=args.file_list)
