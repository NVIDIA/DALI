# Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    Init,
    OpSpec,
    GetCudaVersion,
    GetCufftVersion,
    GetNppVersion,
    GetNvjpegVersion,
)

# Re-expose some private symbols not imported with `import *`
from nvidia.dali.backend_impl import (  # noqa: F401
    _ExecutorType,
    _ExecutorFlags,
    _MakeExecutorType,
    _PipelineParams,
)

# TODO: Handle forwarding imports from backend_impl
from nvidia.dali.backend_impl import *  # noqa: F401, F403

from . import __cuda_version__
import warnings
import sys


_ExecutorType.__bool__ = lambda self: self.value != 0
_ExecutorType.__and__ = lambda x, y: _ExecutorType(x.value & y.value)
_ExecutorType.__or__ = lambda x, y: _ExecutorType(x.value | y.value)
_ExecutorType.__xor__ = lambda x, y: _ExecutorType(x.value ^ y.value)
_ExecutorType.__invert__ = lambda x: _ExecutorType(~x.value)

_ExecutorFlags.__bool__ = lambda self: self.value != 0
_ExecutorFlags.__and__ = lambda x, y: _ExecutorFlags(x.value & y.value)
_ExecutorFlags.__or__ = lambda x, y: _ExecutorFlags(x.value | y.value)
_ExecutorFlags.__xor__ = lambda x, y: _ExecutorFlags(x.value ^ y.value)
_ExecutorFlags.__invert__ = lambda x: _ExecutorFlags(~x.value)


def deprecation_warning(what):
    # show only this warning
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.warn(what, Warning, stacklevel=2)


initialized = False
if not initialized:
    Init(OpSpec("CPUAllocator"), OpSpec("PinnedCPUAllocator"), OpSpec("GPUAllocator"))
    initialized = True

    # py3.6 warning
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        deprecation_warning(
            "DALI 1.27 is the last release to support Python 3.6. "
            "Please update your environment to use Python 3.8, "
            "3.9, 3.10, or (experimentally) 3.11."
        )
    # py3.7 warning
    if sys.version_info[0] == 3 and sys.version_info[1] == 7:
        deprecation_warning(
            "DALI 1.31 is the last release to support Python 3.7. "
            "Please update your environment to use Python 3.8, "
            "3.9, 3.10, or (experimentally) 3.11."
        )
    # py3.8 warning
    if sys.version_info[0] == 3 and sys.version_info[1] == 8:
        deprecation_warning(
            "DALI 1.49 is the last release to support Python 3.8 "
            "Please update your environment to use Python 3.9, "
            "or newer."
        )
    # py3.13 warning
    if sys.version_info[0] == 3 and sys.version_info[1] == 13:
        deprecation_warning("Python 3.13 support is experimental and not officially tested.")

    if int(str(__cuda_version__)[:2]) < 11:
        deprecation_warning(
            "DALI 1.21 is the last official release that supports CUDA 10.2. "
            "Please update your environment to CUDA version 11 or newer."
        )

cuda_checked = False


def check_cuda_runtime():
    """
    Checks the availability of CUDA runtime/GPU, and NPP, nvJPEG, and cuFFT libraries and prints an
    appropriate warning.
    """
    global cuda_checked
    if not cuda_checked:
        cuda_checked = True
        if GetCudaVersion() == -1:
            deprecation_warning("GPU is not available. Only CPU operators are available.")

        if GetCufftVersion() == -1:
            deprecation_warning(
                "nvidia-dali-cuda120 is no longer shipped with CUDA runtime. "
                "You need to install it separately. cuFFT is typically "
                "provided with CUDA Toolkit installation or an appropriate wheel. "
                "Please check "
                "https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html"
                "#pip-wheels-installation-linux "
                "for the reference."
            )

        if GetNppVersion() == -1:
            deprecation_warning(
                "nvidia-dali-cuda120 is no longer shipped with CUDA runtime. "
                "You need to install it separately. NPP is typically "
                "provided with CUDA Toolkit installation or an appropriate wheel. "
                "Please check "
                "https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html"
                "#pip-wheels-installation-linux "
                "for the reference."
            )

        if GetNvjpegVersion() == -1:
            deprecation_warning(
                "nvidia-dali-cuda120 is no longer shipped with CUDA runtime. "
                "You need to install it separately. nvJPEG is typically "
                "provided with CUDA Toolkit installation or an appropriate wheel. "
                "Please check "
                "https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html"
                "#pip-wheels-installation-linux "
                "for the reference."
            )
