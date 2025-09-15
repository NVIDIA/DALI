# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from PIL import Image
import torchvision.transforms as transforms


class Compose(Pipeline):
    """
    Compose several DALI transforms together.

    This class inherits from nvidia.dali.Pipeline and allows chaining multiple
    DALI operations in a sequential manner, similar to torchvision.transforms.Compose.
    The Compose runs implements a callable which runst a pipeline.

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
        # TODO: the below is a hack but it follows the pipeline_def decorator logic:
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
        finally:
            Pipeline.pop_current()

    def define_graph(self):
        self.input_node = self.input()
        self.input_node_name = self.input_node.name

        input_node = self.input_node
        for op in self.op_list:
            input_node = op(input_node)
        return input_node

    def __call__(self, data_input):
        """
        Runs a pipeline

        The Pipeline class builds a graph based on the operations list passed in the constructor.
        Next, whenever the Compose object is called it starts the pipeline and returns results.

        Args:
            data_input: Input tensor or PIL Image.
                In case of PIL image it will be converted to tensor before sending to pipeline
        """
        if isinstance(data_input, Image.Image):
            data_input = transforms.ToTensor()(data_input)

        self.build()
        self.feed_input(data_node=self.input_node_name, data=data_input)
        return self.run()
