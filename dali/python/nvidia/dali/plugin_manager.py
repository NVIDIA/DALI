# Copyright (c) 2018, 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import backend as b
from nvidia.dali import ops as ops


def load_library(library_path: str, global_symbols: bool = False):
    """Loads a DALI plugin, containing one or more operators.

    Args:
        library_path: Path to the plugin library (relative or absolute)
        global_symbols: If ``True``, the library is loaded with ``RTLD_GLOBAL`` flag or equivalent;
            otherwise ``RTLD_LOCAL`` is used. Some libraries (for example Halide) require being
            loaded with ``RTLD_GLOBAL`` - use this setting if your plugin uses any such library.

    Returns:
        None.

    Raises:
        RuntimeError: when unable to load the library.
    """
    b.LoadLibrary(library_path, global_symbols)
    ops.Reload()


def load_directory(plugin_dir_path: str, global_symbols: bool = False):
    """Loads a DALI plugin directory, containing one or more DALI plugins, following the pattern:
    {plugin_dir_path}/{sub_path}/libdali_{plugin_name}.so

    Args:
        plugin_dir_path: Path to the directory to search for plugins
        global_symbols: If ``True``, the library is loaded with ``RTLD_GLOBAL`` flag or equivalent;
            otherwise ``RTLD_LOCAL`` is used. Some libraries (for example Halide) require being
            loaded with ``RTLD_GLOBAL`` - use this setting if your plugin uses any such library.

    Returns:
        None.

    Raises:
        RuntimeError: when unable to load the library.
    """
    b.LoadDirectory(plugin_dir_path, global_symbols)
    ops.Reload()
