# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import random
from nvidia.dali import fn, types, ops, pipeline_def
from nvidia.dali.pipeline import Pipeline
from test_utils import RandomlyShapedDataIterator
from test_utils import compare_pipelines
from nose2.tools import params


class LookupTablePipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        iterator,
        data_shape,
        data_layout,
        dtype,
        num_threads=1,
        device_id=0,
        dictionary={},
        default_value=0.0,
    ):
        super().__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_shape = data_shape
        self.data_layout = data_layout

        if dictionary:
            keys = [k for k in dictionary.keys()]
            values = [dictionary[k] for k in keys]
            self.lookup = ops.LookupTable(
                device=self.device,
                dtype=dtype,
                default_value=default_value,
                keys=keys,
                values=values,
            )
        else:
            self.lookup = ops.LookupTable(
                device=self.device, dtype=dtype, default_value=default_value
            )

    def define_graph(self):
        self.data = self.inputs()
        input_data = self.data.gpu() if self.device == "gpu" else self.data
        out = self.lookup(input_data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)


class LookupTablePythonOpPipeline(Pipeline):
    def __init__(
        self,
        function,
        batch_size,
        iterator,
        data_shape,
        data_layout,
        dtype,
        num_threads=1,
        device_id=0,
        dictionary={},
        default_value=0.0,
    ):
        super().__init__(batch_size, num_threads, device_id, exec_async=False, exec_pipelined=False)
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_shape = data_shape
        self.data_layout = data_layout

        def lookup_table_func(input_data):
            return function(input_data, dictionary=dictionary, default_value=default_value)

        self.lookup = ops.PythonFunction(
            function=lookup_table_func, output_layouts=data_layout, batch_processing=False
        )
        self.cast = ops.Cast(dtype=dtype)

    def define_graph(self):
        self.data = self.inputs()
        out = self.lookup(self.data)
        out = self.cast(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)


def lookup_func(image, dictionary, default_value):
    arr = [default_value for k in range(0x1000)]
    for k in dictionary.keys():
        arr[k] = dictionary[k]
    lut = np.array(arr)
    return lut[image]


def check_lookup_table_vs_python_op(
    device, batch_size, layout, shape, dtype, dictionary_type, default_value
):
    eii1 = RandomlyShapedDataIterator(batch_size, max_shape=shape)
    eii2 = RandomlyShapedDataIterator(batch_size, max_shape=shape)
    if dictionary_type == "empty":
        dictionary = {}
    elif dictionary_type == "random":
        dictionary = {k: random.random() for k in range(256)}
    elif dictionary_type == "small":
        dictionary = {0: 0.1, 200: 0.99}
    else:
        assert False
    compare_pipelines(
        LookupTablePipeline(
            device,
            batch_size,
            iter(eii1),
            data_shape=shape,
            data_layout=layout,
            dtype=dtype,
            dictionary=dictionary,
            default_value=default_value,
        ),
        LookupTablePythonOpPipeline(
            lookup_func,
            batch_size,
            iter(eii2),
            data_shape=shape,
            data_layout=layout,
            dtype=dtype,
            dictionary=dictionary,
            default_value=default_value,
        ),
        batch_size=batch_size,
        N_iterations=3,
    )


def test_lookup_table_vs_python_op():
    layout = types.NHWC
    for device in {"cpu", "gpu"}:
        for dtype in {types.FLOAT, types.FLOAT16, types.INT64}:
            for batch_size, shape, dictionary_type, default_value in [
                (1, (300, 300, 3), "random", 0.0),
                (1, (300, 300, 3), "empty", 0.33),
                (10, (300, 300, 3), "random", 0.9),
                (3, (300, 300, 3), "small", 0.4),
            ]:
                yield (
                    check_lookup_table_vs_python_op,
                    device,
                    batch_size,
                    layout,
                    shape,
                    dtype,
                    dictionary_type,
                    default_value,
                )


@params("cpu", "gpu")
def test_scalar(device):
    @pipeline_def(batch_size=64, num_threads=2, device_id=0)
    def pipe():
        raw = np.array([[0, 1, 2, 3]])  # single batch of 4 scalars
        ids = fn.external_source(source=raw, device=device)
        scale_keys = [0, 1]
        scale_values = [100, 200]
        scale_mat = fn.lookup_table(
            ids,
            keys=scale_keys,
            values=scale_values,
            device=device,
            dtype=types.INT64,
        )
        return scale_mat, ids

    p = pipe()
    scaled, _ = p.run()
    if device == "gpu":
        scaled = scaled.as_cpu()
    assert (scaled.as_array() == [100, 200, 0, 0]).all()
