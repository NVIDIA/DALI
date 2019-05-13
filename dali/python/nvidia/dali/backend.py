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

# Note: If we ever need to add more complex functionality
# for importing the DALI c++ extensions, we can do it here

default_plugins = [
    'libpython_function_plugin.so'
]

initialized = False
if not initialized:
    Init(OpSpec("CPUAllocator"), OpSpec("PinnedCPUAllocator"), OpSpec("GPUAllocator"))
    initialized = True

    # pybind11 deprecations
    def asCPU(self):
        warnings.warn("asCPU is deprecated since v0.7, please use as_cpu", DeprecationWarning, stacklevel=2)
        return self.as_cpu()
    TensorListGPU.asCPU = asCPU

    for lib in default_plugins:
        LoadLibrary(os.path.join(os.path.dirname(__file__), lib))
