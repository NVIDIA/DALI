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

from nvidia.dali.pipeline import Pipeline
from nvidia.dali import _conditionals
from nvidia.dali._utils.dali_trace import set_tracing
from nvidia.dali.data_node import DataNode as _DataNode
import nvidia.dali.ops as ops
from nvidia.dali.backend import TensorListCPU, TensorListGPU

from .tensor import ToTensor

from PIL import Image
import numpy as np
import torch


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


class Compose(Pipeline):
    """
    Compose several DALI transforms together.

    This class inherits from nvidia.dali.Pipeline and allows chaining multiple
    DALI operations in a sequential manner, similar to torchvision.transforms.Compose.
    The Compose runs implements a callable which runs a pipeline.

    """

    def __init__(
        self,
        op_list: List[Callable[..., Union[Sequence[_DataNode], _DataNode]]],
        batch_size: int = 1,
        num_threads: int = 1,
        *args,
        **kwargs,
    ):
        # TODO: WAR tracing does not work with this code, building stack trace asserts
        set_tracing(enabled=False)
        if "prefetch_queue_depth" not in kwargs.keys():
            kwargs["prefetch_queue_depth"] = 1
        # TODO: the below is a hack but it follows the pipeline_def decorator logic
        # (_preprocess_pipe_func):
        self.define_graph = _conditionals._autograph.convert(recursive=True, user_requested=True)(
            self.define_graph
        )

        super().__init__(batch_size, num_threads, *args, **kwargs)
        self.op_list = op_list
        self.input = ops.ExternalSource()

        # The composed operators will have conditional expressions.
        # We need to enable conditionals by default
        try:
            Pipeline.push_current(self)
            self._conditionals_enabled = True
            self._condition_stack = _conditionals._ConditionStack()
            # Add all parameters to the pipeline as "know" nodes in the top scope.
            for arg in args:
                if isinstance(arg, _DataNode):
                    _conditionals.register_data_nodes(arg)
            for _, arg in kwargs.items():
                if isinstance(arg, _DataNode):
                    _conditionals.register_data_nodes(arg)
        finally:
            Pipeline.pop_current()

        with self:
            pipe_outputs = self.define_graph()
            if isinstance(pipe_outputs, tuple):
                po = pipe_outputs
            elif pipe_outputs is None:
                po = ()
            else:
                po = (pipe_outputs,)
            self.set_outputs(*po)

    def define_graph(self):
        self.input_node = self.input()
        self.input_node_name = self.input_node.name

        input_node = self.input_node
        for op in self.op_list:
            if isinstance(op, ToTensor) and op != self.op_list[-1]:
                raise NotImplementedError("ToTensor can only be the last operation in the pipeline")
            input_node = op(input_node)
        return input_node

    def _convert_tensor_to_image(self, in_tensor: torch.Tensor, layout: str):

        if layout == "HWC":
            channels = -1
        elif layout == "CHW":
            channels = -3
        else:
            raise ValueError(f"Unsupported layout: {layout}")
        # TODO: consider when to convert to PIL.Image - e.g. if it make sense for channels < 3
        if in_tensor.shape[channels] == 1:
            mode = "L"
            in_tensor = in_tensor.squeeze(-1)
        elif in_tensor.shape[channels] == 3:
            mode = "RGB"
        else:
            raise ValueError(f"Unsupported channels count: {channels}")
        # We need to convert tensor to CPU, otherwise it will be unusable
        return Image.fromarray(in_tensor.cpu().numpy(), mode=mode)

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

        layout = "CHW"
        _input = data_input
        if isinstance(data_input, Image.Image):
            _input = torch.as_tensor(np.array(data_input, copy=True)).unsqueeze(0)
            layout = "HWC"
        elif isinstance(data_input, torch.Tensor) and data_input.ndim == 3:
            # DALI requires batch size to be present
            _input = _input.unsqueeze(0)

        self.build()
        self.feed_input(data_node=self.input_node_name, layout=layout, data=_input)
        output = self.run()

        if output is None:
            return output

        output = to_torch_tensor(output)
        # ToTensor
        if isinstance(self.op_list[-1], ToTensor):
            if output.shape[-4] > 1:
                raise NotImplementedError("ToTensor does not currently work for batches")

        # Convert to PIL.Image
        elif isinstance(data_input, Image.Image):
            if isinstance(output, tuple):
                output = self._convert_tensor_to_image(output[0], layout)
            else:
                # batches
                if output.shape[0] > 1:
                    output_list = []
                    for i in range(output.shape[0]):
                        output_list.append(self._convert_tensor_to_image(output[i], layout))
                    output = output_list
                else:
                    output = self._convert_tensor_to_image(output[0], layout)

        elif isinstance(data_input, torch.Tensor):
            if data_input.ndim == 3:
                # DALI requires batch size to be present
                output = output.squeeze(0)

        return output
