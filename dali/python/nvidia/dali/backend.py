# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.backend_impl import *
from . import __cuda_version__
import warnings
import os
import sys

# Note: If we ever need to add more complex functionality
# for importing the DALI c++ extensions, we can do it here

default_plugins = [
]

def deprecation_warning(what):
    # show only this warning
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.warn(what, Warning, stacklevel=2)

initialized = False
if not initialized:
    Init(OpSpec("CPUAllocator"), OpSpec("PinnedCPUAllocator"), OpSpec("GPUAllocator"))
    initialized = True

    # py39 warning
    if sys.version_info[0] == 3 and sys.version_info[1] == 9:
        deprecation_warning("DALI support for Python 3.9 is experimental and some functionalities "
                            "may not work.")

    if __cuda_version__ < 102:
        deprecation_warning("DALI 1.3 is the last official release that supports CUDA 10.0. "
                            "The next release will support only 10.2 from 10.x familly. "
                            "Please update your environment to CUDA version 10.2 or newer.")

    for lib in default_plugins:
        LoadLibrary(os.path.join(os.path.dirname(__file__), lib))
