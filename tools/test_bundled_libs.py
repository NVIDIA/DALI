#!/usr/bin/env python

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function
from sys import argv
import subprocess

def check_ldd_out(lib, linked_lib, bundled_lib_names, allowed_libs):
    # Gather all libs that may be linked with 'lib' and don't need to be bundled
    # Entries from 'lib' key in allowed_libs should cover all 'lib*' libs
    # Empty key is used for all libs
    allowed_libs_to_check = []
    for k in allowed_libs.keys():
        if k in lib:
            allowed_libs_to_check += allowed_libs[k]

    return linked_lib in bundled_lib_names or linked_lib in allowed_libs_to_check

def main():
    allowed_libs = {"": ["linux-vdso.so.1",
                        "libm.so.6",
                        "libpthread.so.0",
                        "libc.so.6",
                        "/lib64/ld-linux-x86-64.so.2",
                        "libdl.so.2",
                        "librt.so.1",
                        "libstdc++.so.6",
                        "libgcc_s.so.1",
                        "libz.so.1"
                    ],
                    "libdali_tf": ["libtensorflow_framework.so"]
                    }

    bundled_libs = argv[1:]

    # Gather all names of bundled libs without path
    bundled_lib_names = []
    for lib in bundled_libs:
        beg = lib.rfind('/')
        bundled_lib_names.append(lib[beg + 1:])

    print("Checking bundled libs linkage:")
    for lib_path, lib_name in zip(bundled_libs, bundled_lib_names):
        print ("- " + lib_name)
        ldd = subprocess.Popen(["ldd", lib_path], stdout=subprocess.PIPE)
        for l in ldd.stdout:
            l = l.decode().strip('\t').strip('\n')
            linked_lib = l.split()[0]
            if not check_ldd_out(lib_name, linked_lib, bundled_lib_names, allowed_libs):
                print('Library: "' + linked_lib + '" should be bundled in whl or removed from the dynamic link dependency')
                exit(1)
    print("-> OK")

if __name__ == '__main__':
    main()
