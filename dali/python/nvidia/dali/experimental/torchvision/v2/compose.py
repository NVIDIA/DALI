# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Sequence, Union, Callable

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.backend import TensorListCPU, TensorListGPU

from .tensor import ToTensor

import numpy as np
import multiprocessing
from PIL import Image
import torch

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_THREADS = multiprocessing.cpu_count() // 2


def _to_torch_tensor(tensor_or_tl: Union[TensorListGPU, TensorListCPU]) -> torch.Tensor:
    if isinstance(tensor_or_tl, (TensorListGPU, TensorListCPU)):
        dali_tensor = tensor_or_tl.as_tensor()
    else:
        dali_tensor = tensor_or_tl

    return torch.from_dlpack(dali_tensor)


def to_torch_tensor(tensor_or_tl: Union[tuple, TensorListGPU, TensorListCPU]) -> torch.Tensor:

    if isinstance(tensor_or_tl, tuple) and len(tensor_or_tl) > 1:
        tl = []
        for elem in tensor_or_tl:
            tl.append(_to_torch_tensor(elem))
        return tuple(tl)
    else:
        if len(tensor_or_tl) == 1:
            tensor_or_tl = tensor_or_tl[0]
        return _to_torch_tensor(tensor_or_tl)


@pipeline_def(enable_conditionals=True, exec_dynamic=True, prefetch_queue_depth=1)
def _pipeline_function(op_list, layout="HWC"):

    input_node = fn.external_source(name="input_data", no_copy=True, layout=layout)
    for op in op_list:
        if isinstance(op, ToTensor) and op != op_list[-1]:
            raise NotImplementedError("ToTensor can only be the last operation in the pipeline")
        input_node = op(input_node)
    return input_node


class PipelineLayouted:
    def __init__(
        self,
        op_list: List[Callable[..., Union[Sequence[_DataNode], _DataNode]]],
        layout: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        **dali_pipeline_kwargs,
    ):
        self.convert_to_tensor = True if isinstance(op_list[-1], ToTensor) else False
        self.pipe = _pipeline_function(
            op_list,
            layout=layout,
            batch_size=batch_size,
            num_threads=num_threads,
            *dali_pipeline_kwargs,
        )

    def run(self, data_input):
        output = self.pipe.run(input_data=data_input)  # TODO: get stream

        if output is None:
            return output

        output = to_torch_tensor(output)
        # ToTensor
        if self.convert_to_tensor:
            if output.shape[-4] > 1:
                raise NotImplementedError("ToTensor does not currently work for batches")

        return output

    def get_layout(self) -> str: ...

    def get_channel_reverse_idx(self) -> int: ...

    def is_conversion_to_tensor(self) -> bool:
        return self.convert_to_tensor


class PipelineHWC(PipelineLayouted):
    """
    Handles PIL Images in HWC format

    This class prepares data to be passed to a DALI pipeline, runs the pipeline and converts
    pipeline output to a PIL Image
    """

    def __init__(
        self,
        op_list: List[Callable[..., Union[Sequence[_DataNode], _DataNode]]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        **dali_pipeline_kwargs,
    ):
        super().__init__(
            op_list,
            layout="HWC",
            batch_size=batch_size,
            num_threads=num_threads,
            *dali_pipeline_kwargs,
        )

    def _convert_tensor_to_image(self, in_tensor: torch.Tensor):

        channels = self.get_channel_reverse_idx()

        # TODO: consider when to convert to PIL.Image - e.g. if it make sense for channels < 3
        if in_tensor.shape[channels] == 1:
            mode = "L"
            in_tensor = in_tensor.squeeze(-1)
        elif in_tensor.shape[channels] == 3:
            mode = "RGB"
        else:
            raise ValueError(f"Unsupported number of channels: {channels}. Should be 1 or 3.")
        # We need to convert tensor to CPU, otherwise it will be unsable
        return Image.fromarray(in_tensor.cpu().numpy(), mode=mode)

    def run(self, data_input):
        if isinstance(data_input, Image.Image):
            _input = torch.as_tensor(np.array(data_input, copy=True)).unsqueeze(0)
        else:
            ValueError(
                "HWC layout is currently supported for PIL Images only.\
                Please check if samples have the same format."
            )

        output = super().run(_input)

        if self.is_conversion_to_tensor():
            return output

        if isinstance(output, tuple):
            output = self._convert_tensor_to_image(output[0])
        else:
            # batches
            if output.shape[0] > 1:
                output_list = []
                for i in range(output.shape[0]):
                    output_list.append(self._convert_tensor_to_image(output[i]))
                output = output_list
            else:
                output = self._convert_tensor_to_image(output[0])

        return output

    def get_layout(self) -> str:
        return "HWC"

    def get_channel_reverse_idx(self) -> int:
        return -1


class PipelineCHW(PipelineLayouted):
    """
    Handles torch.Tensors in CHW format

    This class prepares data to be passed to a DALI pipeline and runs the pipeline
    """

    def __init__(
        self,
        op_list: List[Callable[..., Union[Sequence[_DataNode], _DataNode]]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        **dali_pipeline_kwargs,
    ):
        super().__init__(
            op_list,
            layout="CHW",
            batch_size=batch_size,
            num_threads=num_threads,
            *dali_pipeline_kwargs,
        )

    def run(self, data_input):
        if isinstance(data_input, torch.Tensor):
            _input = data_input
            if data_input.ndim == 3:
                # DALI requires batch size to be present
                _input = data_input.unsqueeze(0)
        else:
            ValueError(
                "CHW layout is currently supported for torch.Tensor only.\
                Please check if samples have the same format."
            )

        output = super().run(_input)

        if data_input.ndim == 3:
            # DALI requires batch size to be present
            output = output.squeeze(0)
        return output

    def get_layout(self) -> str:
        return "CHW"

    def get_channel_reverse_idx(self) -> int:
        return -3


class Compose:
    """
    Composes transforms together in a single pipeline

    This class chaining multiple DALI operations in a sequential manner,
    similar to torchvision.transforms.Compose. The Compose implements a callable
    which runs a pipeline.

    """

    def __init__(
        self,
        op_list: List[Callable[..., Union[Sequence[_DataNode], _DataNode]]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        **dali_pipeline_kwargs,
    ):
        self.op_list = op_list
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.active_pipeline = None
        self.dali_pipeline_kwargs = dali_pipeline_kwargs

    def _build_pipeline(self, data_input):
        if isinstance(data_input, Image.Image):
            self.active_pipeline = PipelineHWC(
                self.op_list, self.batch_size, self.num_threads, *self.dali_pipeline_kwargs
            )
        elif isinstance(data_input, torch.Tensor):
            self.active_pipeline = PipelineCHW(
                self.op_list, self.batch_size, self.num_threads, *self.dali_pipeline_kwargs
            )
        else:
            raise ValueError("Currently only PILImages and torch.Tesors are supported")

    def __call__(self, data_input):
        """
        Runs a pipeline

        The Pipeline class builds a graph based on the operations list passed in the constructor.
        Next, whenever the Compose object is called it starts the pipeline and returns results.

        Args:
            data_input: Input tensor or PIL Image.
                In case of PIL image it will be converted to tensor before sending to pipeline
        """

        if not isinstance(data_input, (Image.Image, torch.Tensor)):
            raise TypeError(f"input should be PIL Image or torch.Tensor. Got {type(data_input)}")

        if self.active_pipeline is None:
            self._build_pipeline(data_input)

        return self.active_pipeline.run(data_input=data_input)
