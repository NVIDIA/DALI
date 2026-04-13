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

from abc import ABC, abstractmethod
import logging
from typing import List, Sequence, Callable, Union

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.backend import TensorListCPU, TensorListGPU

from .operator import _ValidateTensorOrImage
from .totensor import ToPureTensor, PILToTensor, ToPILImage
from .functional.totensor import to_pil_image

import numpy as np
import multiprocessing
from PIL import Image
import torch

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_THREADS = 1 if multiprocessing.cpu_count() == 1 else multiprocessing.cpu_count() // 2


def _to_torch_tensor(tensor_or_tl: TensorListGPU | TensorListCPU) -> torch.Tensor:
    if isinstance(tensor_or_tl, (TensorListGPU, TensorListCPU)):
        dali_tensor = tensor_or_tl.as_tensor()
    else:
        dali_tensor = tensor_or_tl

    return torch.from_dlpack(dali_tensor)


def to_torch_tensor(
    x: Union[tuple, "TensorListGPU", "TensorListCPU"],
) -> Union[torch.Tensor, tuple]:
    """
    Converts a DALI tensor or tensor list to a PyTorch tensor.

    Parameters
    ----------
        tensor_or_tl : tuple, TensorListGPU, TensorListCPU
            DALI tensor or tensor list.
    """
    if isinstance(x, (TensorListGPU, TensorListCPU)):
        return to_torch_tensor(x.as_tensor())
    elif isinstance(x, tuple):
        if len(x) == 1:
            return _to_torch_tensor(x[0])
        return tuple(to_torch_tensor(elem) for elem in x)
    else:
        return torch.from_dlpack(x)


@pipeline_def(enable_conditionals=True, exec_dynamic=True, prefetch_queue_depth=1)
def _pipeline_function(op_list: Sequence, layout: str = "HWC", input_device: str = "gpu"):
    """
    Builds a DALI pipeline from a list of operators.

    Parameters
    ----------
        op_list : list
            List of DALI operators.
        layout : str
            Layout of the data.
    """
    input_node = fn.external_source(
        name="input_data", no_copy=True, layout=layout, device=input_device
    )
    for op in op_list:
        input_node = op(input_node)
    return input_node


class PipelineWithLayout(ABC):
    """Base class for pipeline layouts.

    This class is a base class for DALI pipelines with a specific layout. It is used to handle
    the layout of the data.
    Single DALI Pipeline can only use one layout at a time.

    Parameters
    ----------
    op_list : list
        List of DALI operators.
    layout : str
        Layout of the data.
    batch_size : int, optional, default = DEFAULT_BATCH_SIZE
        Batch size.
    num_threads : int, optional, default = DEFAULT_NUM_THREADS
        Number of threads.
    **dali_pipeline_kwargs
        Additional keyword arguments for the DALI pipeline.
    """

    def _cuda_run(self, data_input):
        if isinstance(data_input, torch.Tensor) and data_input.is_cuda:
            device_id = data_input.device.index
        else:
            device_id = torch.cuda.current_device()

        stream = torch.cuda.current_stream(device=device_id)

        with torch.cuda.stream(stream):
            output = self.pipe.run(stream, input_data=data_input)

        return output

    def _cpu_run(self, data_input):
        return self.pipe.run(input_data=data_input)

    @staticmethod
    def _get_output_format(op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]]) -> str:
        output_type = "default"
        for op in op_list:
            if isinstance(op, (ToPureTensor, PILToTensor)):
                output_type = "tensor"
            elif isinstance(op, ToPILImage):
                output_type = "pil"

        return output_type

    def __init__(
        self,
        op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]],
        layout: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        **dali_pipeline_kwargs,
    ):
        self.output_type = PipelineWithLayout._get_output_format(op_list)
        self.device = op_list[0].device if len(op_list) > 0 else "cpu"
        self.torch_device_type = "cuda" if self.device == "gpu" else "cpu"

        self.pipe = _pipeline_function(
            op_list,
            layout=layout,
            input_device=self.device,
            batch_size=batch_size,
            num_threads=num_threads,
            **dali_pipeline_kwargs,
        )
        self._internal_run = self._cuda_run if torch.cuda.is_available() else self._cpu_run

    def _align_data_with_device(self, data_input):
        if self.torch_device_type != data_input.device.type:
            logging.warning(
                f"Pipeline device is {self.device}, but data is on {data_input.device.type}."
                " Copying!"
            )
            return data_input.cpu() if self.torch_device_type == "cpu" else data_input.cuda()

        return data_input

    def run(self, data_input):

        output = self._internal_run(self._align_data_with_device(data_input))

        if output is None:
            return output

        return to_torch_tensor(output)

    @abstractmethod
    def output_to_tensor(self, output: torch.Tensor) -> torch.Tensor:
        """Return the pipeline output as a CHW ``torch.Tensor``."""
        ...

    @abstractmethod
    def output_to_pil(self, output: torch.Tensor) -> Image.Image:
        """Return the pipeline output tensor as a ``PIL.Image``."""
        ...

    @abstractmethod
    def get_layout(self) -> str: ...

    @abstractmethod
    def get_channel_reverse_idx(self) -> int: ...

    @abstractmethod
    def verify_layout(self, data) -> None: ...

    def is_conversion_to_tensor(self) -> bool:
        return self.output_type == "tensor"


class PipelineHWC(PipelineWithLayout):
    """Handles ``PIL.Image`` in HWC format.

    This class prepares data to be passed to a DALI pipeline, runs the pipeline and converts
    the output to a ``PIL.Image``.

    Parameters
    ----------
    op_list : list
        List of DALI operators.
    batch_size : int, optional, default = DEFAULT_BATCH_SIZE
        Batch size.
    num_threads : int, optional, default = DEFAULT_NUM_THREADS
        Number of threads.
    **dali_pipeline_kwargs
        Additional keyword arguments for the DALI pipeline.
    """

    def __init__(
        self,
        op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        **dali_pipeline_kwargs,
    ):
        super().__init__(
            op_list,
            layout="HWC",
            batch_size=batch_size,
            num_threads=num_threads,
            **dali_pipeline_kwargs,
        )

    def run(self, data_input):
        if isinstance(data_input, Image.Image):
            _input = torch.as_tensor(np.array(data_input, copy=True)).unsqueeze(0)
            if data_input.mode == "L":
                _input = _input.unsqueeze(-1)
        else:
            raise ValueError("HWC layout is currently supported for PIL Images only.\
                Please check if samples have the same format.")

        output = super().run(_input)  # (H, W, C)

        if self.output_type == "tensor":
            return self.output_to_tensor(output)

        # default: PIL output
        if output.shape[0] > 1:
            return [self.output_to_pil(output[i]) for i in range(output.shape[0])]
        return self.output_to_pil(output[0])

    def output_to_tensor(self, output: torch.Tensor) -> torch.Tensor:
        """Convert a single-sample HWC tensor to a CHW tensor"""
        if output.shape[0] == 1:
            return output.squeeze(0).permute(2, 0, 1).contiguous()  # → (C, H, W)
        return output.permute(0, 3, 1, 2).contiguous()  # → (N, C, H, W)

    def output_to_pil(self, output: torch.Tensor) -> Image.Image:
        """Convert a single-sample HWC tensor to a ``PIL.Image``."""
        # Pipeline stores data as HWC; to_pil_image expects CHW, so permute first.
        return to_pil_image(output, input_layout="HWC")

    def get_layout(self) -> str:
        return "HWC"

    def get_channel_reverse_idx(self) -> int:
        return -1

    def verify_layout(self, data_input) -> None:
        if not isinstance(data_input, Image.Image):
            raise TypeError(f"The pipeline expects PIL.Images as input got {type(data_input)}")


class PipelineCHW(PipelineWithLayout):
    """Handles ``torch.Tensors`` in CHW format.

    This class prepares data to be passed to a DALI pipeline and runs the pipeline, converting
    the output to a ``torch.Tensor``.

    Parameters
    ----------
    op_list : list
        List of DALI operators.
    batch_size : int, optional, default = DEFAULT_BATCH_SIZE
        Batch size.
    num_threads : int, optional, default = DEFAULT_NUM_THREADS
        Number of threads.
    **dali_pipeline_kwargs
        Additional keyword arguments for the DALI pipeline.
    """

    def __init__(
        self,
        op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        **dali_pipeline_kwargs,
    ):
        super().__init__(
            op_list,
            layout="CHW",
            batch_size=batch_size,
            num_threads=num_threads,
            **dali_pipeline_kwargs,
        )

    def run(self, data_input):
        if isinstance(data_input, torch.Tensor):
            _input = data_input
            if data_input.ndim == 3:
                # DALI requires batch size to be present
                _input = data_input.unsqueeze(0)
        else:
            raise ValueError("CHW layout is currently supported for torch.Tensor only.\
                Please check if samples have the same format.")
        output = super().run(_input)

        if data_input.ndim == 3:
            # Remove the batch dimension we added above
            output = output.squeeze(0)

        if self.output_type == "pil":
            if output.ndim == 4:
                if output.shape[0] > 1:
                    return [self.output_to_pil(output[i]) for i in range(output.shape[0])]
                return self.output_to_pil(output[0])
            return self.output_to_pil(output)

        return output

    def output_to_tensor(self, output: torch.Tensor) -> torch.Tensor:
        """Return the pipeline output as a CHW ``torch.Tensor``."""
        return output

    def output_to_pil(self, output: torch.Tensor) -> Image.Image:
        return to_pil_image(output)

    def get_layout(self) -> str:
        return "CHW"

    def get_channel_reverse_idx(self) -> int:
        return -3

    def verify_layout(self, data_input) -> None:
        if not isinstance(data_input, torch.Tensor):
            raise TypeError(f"The pipeline expects torch.Tensor as input got {type(data_input)}")


class Compose:
    """
    Composes transforms together in a single pipeline

    This class chains multiple DALI operations in a sequential manner, similar to
    ``torchvision.transforms.Compose``. The ``Compose`` class implements a callable which runs
    the pipeline.

    Parameters
    ----------
    op_list : list
        List of DALI operators.
    batch_size : int, optional, default = DEFAULT_BATCH_SIZE
        Batch size.
    num_threads : int, optional, default = DEFAULT_NUM_THREADS
        Number of threads.
    **dali_pipeline_kwargs
        Additional keyword arguments for the DALI pipeline.
    """

    def __init__(
        self,
        op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]],
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
                self.op_list, self.batch_size, self.num_threads, **self.dali_pipeline_kwargs
            )
        elif isinstance(data_input, torch.Tensor):
            self.active_pipeline = PipelineCHW(
                self.op_list, self.batch_size, self.num_threads, **self.dali_pipeline_kwargs
            )
        else:
            raise ValueError("Currently only PILImages and torch.Tensors are supported")

    def __call__(self, data_input):
        """
        Runs the pipeline

        The ``Pipeline`` class builds a graph based on the operations list passed in
        the constructor. Next, whenever the ``Compose`` object is called it starts the pipeline
        and returns results.

        Parameters
        ----------
            data_input: Tensor or PIL Image
                In case of PIL image it will be converted to tensor before sending to pipeline
        """

        _ValidateTensorOrImage.verify(data_input)

        if self.active_pipeline is None:
            self._build_pipeline(data_input)

        self.active_pipeline.verify_layout(data_input)

        return self.active_pipeline.run(data_input=data_input)
