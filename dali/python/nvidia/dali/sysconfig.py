# Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def get_include_dir():
    """Get the path to the directory containing C++ header files.

    Returns:
        String representing the path to the include directory
    """
    # Import inside the function to avoid circular import as dali imports sysconfig
    import nvidia.dali as dali

    return os.path.join(os.path.dirname(dali.__file__), "include")


def get_lib_dir():
    """Get the path to the directory containing DALI library.

    Returns:
        String representing the path to the library directory
    """
    import nvidia.dali as dali

    return os.path.dirname(dali.__file__)


def get_include_flags():
    """Get the include flags for custom operators

    Returns:
        The compilation flags
    """
    flags = []
    flags.append("-I%s" % get_include_dir())
    return flags


def get_compile_flags():
    """Get the compilation flags for custom operators

    Returns:
        The compilation flags
    """
    import nvidia.dali.backend as b

    flags = []
    flags.append("-I%s" % get_include_dir())
    flags.append("-D_GLIBCXX_USE_CXX11_ABI=%d" % b.GetCxx11AbiFlag())
    return flags


def get_link_flags():
    """Get the link flags for custom operators

    Returns:
        The link flags
    """
    flags = []
    flags.append("-L%s" % get_lib_dir())
    flags.append("-ldali")
    return flags
