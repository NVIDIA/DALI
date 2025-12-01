#!/usr/bin/env python

# Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
import sys


def get_list_elm_match(value, elms):
    """Check if any element in the elms list matches the value"""
    return any(e in value for e in elms)


def check_ldd_out(lib, linked_lib, bundled_lib_names, allowed_libs):
    # Gather all libs that may be linked with 'lib' and don't need to be bundled
    # Entries from 'lib' key in allowed_libs should cover all 'lib*' libs
    # Empty key is used for all libs
    allowed_libs_to_check = []
    for k in allowed_libs.keys():
        if k in lib:
            allowed_libs_to_check += allowed_libs[k]

    return linked_lib in bundled_lib_names or get_list_elm_match(linked_lib, allowed_libs_to_check)


def main():
    allowed_libs = {
        "": [
            "linux-vdso.so.1",
            "libm.so.6",
            "libpthread.so.0",
            "libc.so.6",
            "/lib64/ld-linux",
            "/lib/ld-linux",
            "libdl.so.2",
            "librt.so.1",
            "libstdc++.so.6",
            "libgcc_s.so.1",
            "libasan.so",
            "liblsan.so",
            "libubsan.so",
            "libtsan.so",
        ]
    }

    bundled_libs = sys.argv[1:]

    # Gather all names of bundled libs without path
    bundled_lib_names = []
    for lib in bundled_libs:
        beg = lib.rfind("/")
        bundled_lib_names.append(lib[beg + 1 :])

    print("Checking bundled libs linkage:")
    failing = False
    for lib_path, lib_name in zip(bundled_libs, bundled_lib_names):
        print(f"- {lib_name}")
        ldd = subprocess.Popen(["ldd", lib_path], stdout=subprocess.PIPE)
        for lib in ldd.stdout:
            lib = lib.decode().strip("\t").strip("\n")
            linked_lib = lib.split()[0]
            if not check_ldd_out(lib_name, linked_lib, bundled_lib_names, allowed_libs):
                print(
                    f"ERROR: The library: '{linked_lib}' should be bundled in whl "
                    f"or removed from the dynamic link dependency",
                    file=sys.stderr,
                )
                failing = True

    if failing:
        sys.exit(1)

    print("-> OK")


if __name__ == "__main__":
    main()
