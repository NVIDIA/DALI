# Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.backend_impl import (
    Init, OpSpec, LoadLibrary, GetCudaVersion, GetCufftVersion, GetNppVersion, GetNvjpegVersion)

# TODO: Handle forwarding imports from backend_impl
from nvidia.dali.backend_impl import *        # noqa: F401, F403

from . import __cuda_version__
import warnings
import os
import sys

# Note: If we ever need to add more complex functionality
# for importing the DALI c++ extensions, we can do it here

default_plugins = []


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
    if sys.version_info[0] == 3 and sys.version_info[1] >= 11:
        deprecation_warning("DALI support for Python {0}.{1} is experimental and some "
                            "functionalities may not work."
                            "".format(sys.version_info[0], sys.version_info[1]))

    if int(str(__cuda_version__)[:2]) < 11:
        deprecation_warning("DALI 1.21 is the last official release that supports CUDA 10.2. "
                            "Please update your environment to CUDA version 11 or newer.")

    for lib in default_plugins:
        LoadLibrary(os.path.join(os.path.dirname(__file__), lib))

cuda_checked = False


def check_cuda_runtime():
    """
    Checks the availability of CUDA runtime/GPU, and NPP, nvJEPG, and cuFFT libraries and prints an
    appropriate warning.
    """
    global cuda_checked
    if not cuda_checked:
        cuda_checked = True
        if GetCudaVersion() == -1:
            deprecation_warning("GPU is not available. Only CPU operators are available.")

        if GetCufftVersion() == -1:
            deprecation_warning("nvidia-dali-cuda120 is no longer shipped with CUDA runtime. "
                                "You need to install it separately. cuFFT is typically "
                                "provided with CUDA Toolkit installation or an appropriate wheel. "
                                "Please check "
                                "https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html"
                                "#pip-wheels-installation-linux "
                                "for the reference.")

        if GetNppVersion() == -1:
            deprecation_warning("nvidia-dali-cuda120 is no longer shipped with CUDA runtime. "
                                "You need to install it separately. NPP is typically "
                                "provided with CUDA Toolkit installation or an appropriate wheel. "
                                "Please check "
                                "https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html"
                                "#pip-wheels-installation-linux "
                                "for the reference.")

        if GetNvjpegVersion() == -1:
            deprecation_warning("nvidia-dali-cuda120 is no longer shipped with CUDA runtime. "
                                "You need to install it separately. nvJPEG is typically "
                                "provided with CUDA Toolkit installation or an appropriate wheel. "
                                "Please check "
                                "https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html"
                                "#pip-wheels-installation-linux "
                                "for the reference.")
