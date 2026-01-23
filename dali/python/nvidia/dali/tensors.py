# Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pylint: disable=no-name-in-module, unused-import
from nvidia.dali.backend import (  # noqa: F401
    TensorCPU as TensorCPU,
    TensorListCPU as TensorListCPU,
    TensorGPU as TensorGPU,
    TensorListGPU as TensorListGPU,
)

from nvidia.dali._tensor_formatting import (
    format_tensor,
    format_batch,
    PipelineTensorAdapter,
    PipelineBatchAdapter,
)


def _tensor_to_string(self, show_data=True):
    """Returns string representation of Tensor.

    Parameters
    ----------
    show_data : bool, optional
        Access and format the underlying data, by default True
    """
    return format_tensor(self, show_data=show_data, adapter=PipelineTensorAdapter())


def _tensorlist_to_string(self, show_data=True, indent=""):
    """Returns string representation of TensorList.

    Parameters
    ----------
    show_data : bool, optional
        Access and format the underlying data, by default True
    indent : str, optional
        optional indentation used in formatting, by default ""
    """
    return format_batch(self, show_data=show_data, indent=indent, adapter=PipelineBatchAdapter())
