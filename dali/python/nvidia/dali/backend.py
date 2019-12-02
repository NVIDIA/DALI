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
import warnings
import os
import sys

# Note: If we ever need to add more complex functionality
# for importing the DALI c++ extensions, we can do it here

default_plugins = [
    'libpython_function_plugin.so'
]

initialized = False
if not initialized:
    Init(OpSpec("CPUAllocator"), OpSpec("PinnedCPUAllocator"), OpSpec("GPUAllocator"))
    initialized = True

    # py27 deprecation
    if sys.version_info[0] < 3:
        # show only this warning
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.warn("DALI 0.17 is the last official release for Python 2.7, which"
                        "reaches the end of life on January 1st, 2020. To stay up to date with"
                        "DALI, please upgrade to Python 3.5 or later.", Warning, stacklevel=2)

    for lib in default_plugins:
        LoadLibrary(os.path.join(os.path.dirname(__file__), lib))
