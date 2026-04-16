# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal

from .operator import Operator


class ToPureTensor(Operator):
    """
    Signals the pipeline to return a ``torch.Tensor`` instead of the default output type.

    When placed in a ``Compose`` op list this operator does not modify the data inside the
    DALI pipeline.  It acts as a marker that causes ``PipelineWithLayout`` to return a plain
    ``torch.Tensor`` in CHW layout regardless of the pipeline's default output type.

    Parameters
    ----------
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the operator. Can be ``"cpu"`` or ``"gpu"``.
    """

    def __init__(self, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(device=device)

    def _kernel(self, data_input):
        return data_input


class PILToTensor(Operator):
    """
    Signals ``PipelineHWC`` to return a uint8 CHW ``torch.Tensor`` instead of ``PIL.Image``.

    When placed in a ``Compose`` op list this operator does not modify the data inside the
    DALI pipeline.  It acts as a marker that causes ``PipelineHWC`` (PIL input path) to
    return a CHW ``torch.Tensor`` with dtype ``uint8`` and values in [0, 255].

    Parameters
    ----------
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the operator. Can be ``"cpu"`` or ``"gpu"``.
    """

    def __init__(self, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(device=device)

    def _kernel(self, data_input):
        return data_input


class ToPILImage(Operator):
    """
    Signals ``PipelineCHW`` to return a ``PIL.Image`` instead of ``torch.Tensor``.

    When placed in a ``Compose`` op list this operator does not modify the data inside the
    DALI pipeline.  It acts as a marker that causes ``PipelineCHW`` (torch.Tensor input path)
    to convert the CHW output tensor back to a ``PIL.Image``.

    Parameters
    ----------
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the operator. Can be ``"cpu"`` or ``"gpu"``.
    """

    def __init__(self, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(device=device)

    def _kernel(self, data_input):
        return data_input
