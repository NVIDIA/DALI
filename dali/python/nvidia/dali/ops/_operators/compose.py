# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.data_node import DataNode as _DataNode


class _CompoundOp:
    def __init__(self, op_list):
        self._ops = []
        for op in op_list:
            if isinstance(op, _CompoundOp):
                self._ops += op._ops
            else:
                self._ops.append(op)

    def __call__(self, *inputs: _DataNode, **kwargs) -> Union[Sequence[_DataNode], _DataNode, None]:
        inputs = list(inputs)
        for op in self._ops:
            for i in range(len(inputs)):
                if (
                    inputs[i].device == "cpu"
                    and op.device == "gpu"
                    and op.schema.GetInputDevice(i, inputs[i].device, op.device) != "cpu"
                ):
                    inputs[i] = inputs[i].gpu()
            inputs = op(*inputs, **kwargs)
            kwargs = {}
            if isinstance(inputs, tuple):
                inputs = list(inputs)
            if isinstance(inputs, _DataNode):
                inputs = [inputs]

        return inputs[0] if len(inputs) == 1 else inputs


def Compose(op_list: List[Callable[..., Union[Sequence[_DataNode], _DataNode]]]) -> _CompoundOp:
    """Returns a meta-operator that chains the operations in op_list.

    The return value is a callable object which, when called, performs::

        op_list[n-1](op_list([n-2](...  op_list[0](args))))

    Operators can be composed only when all outputs of the previous operator can be processed
    directly by the next operator in the list.

    The example below chains an image decoder and a Resize operation with random square size.
    The  ``decode_and_resize`` object can be called as if it was an operator::

        decode_and_resize = ops.Compose([
            ops.decoders.Image(device="cpu"),
            ops.Resize(size=fn.random.uniform(range=400,500)), device="gpu")
        ])

        files, labels = fn.readers.caffe(path=caffe_db_folder, seed=1)
        pipe.set_outputs(decode_and_resize(files), labels)

    If there's a transition from CPU to GPU in the middle of the `op_list`, as is the case in this
    example, ``Compose`` automatically arranges copying the data to GPU memory.


    .. note::
        This is an experimental feature, subject to change without notice."""
    return op_list[0] if len(op_list) == 1 else _CompoundOp(op_list)
