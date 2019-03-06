# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

def load_library(library_path):
    """Loads a DALI plugin, containing one or more operators.

    Args:
        library_path: Path to the plugin library (relative or absolute)
    
    Returns:
        None.

    Raises:
        RuntimeError: when unable to load the library.
    """
    b.LoadLibrary(library_path)
    ops.Reload()