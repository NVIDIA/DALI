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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali import fn, pipeline_def
import math
from test_utils import compare_pipelines, as_array, RandomDataIterator, RandomlyShapedDataIterator
import itertools
from nose2.tools import params
import numpy as np


def transpose_func(image, permutation=(1, 0, 2)):
    return image.transpose(permutation)


class TransposePipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        layout,
        iterator,
        num_threads=1,
        device_id=0,
        permutation=(1, 0, 2),
        transpose_layout=False,
        out_layout_arg=None,
    ):
        super(TransposePipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        if out_layout_arg:
            self.transpose = ops.Transpose(
                device=self.device,
                perm=permutation,
                transpose_layout=transpose_layout,
                output_layout=out_layout_arg,
            )
        else:
            self.transpose = ops.Transpose(
                device=self.device, perm=permutation, transpose_layout=transpose_layout
            )

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == "gpu" else self.data
        out = self.transpose(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


class PythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(PythonOpPipeline, self).__init__(
            batch_size, num_threads, device_id, exec_async=False, exec_pipelined=False
        )
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.oper = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.oper(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def check_transpose_vs_numpy(device, batch_size, dim, total_volume, permutation):
    max_shape = [int(math.pow(total_volume / batch_size, 1 / dim))] * dim
    print("Testing", device, "backend with batch of", batch_size, "max size", max_shape)
    print("permutation ", permutation)
    eii1 = RandomlyShapedDataIterator(batch_size, max_shape=max_shape)
    eii2 = RandomlyShapedDataIterator(batch_size, max_shape=max_shape)
    compare_pipelines(
        TransposePipeline(device, batch_size, "", iter(eii1), permutation=permutation),
        PythonOpPipeline(lambda x: transpose_func(x, permutation), batch_size, "", iter(eii2)),
        batch_size=batch_size,
        N_iterations=3,
    )


def all_permutations(n):
    return itertools.permutations(range(n))


def test_transpose_vs_numpy():
    for device in ["cpu", "gpu"]:
        for batch_size in [1, 3, 10, 100]:
            for dim in range(2, 5):
                for permutation in all_permutations(dim):
                    yield check_transpose_vs_numpy, device, batch_size, dim, 1000000, permutation


def check_transpose_layout(
    device, batch_size, shape, in_layout, permutation, transpose_layout, out_layout_arg
):
    eii = RandomDataIterator(batch_size, shape=shape)
    pipe = TransposePipeline(
        device,
        batch_size,
        in_layout,
        iter(eii),
        permutation=permutation,
        transpose_layout=transpose_layout,
        out_layout_arg=out_layout_arg,
    )
    out = pipe.run()

    expected_out_layout = in_layout
    if out_layout_arg:
        expected_out_layout = out_layout_arg
    elif transpose_layout:
        expected_out_layout = "".join([list(in_layout)[d] for d in permutation])
    else:
        expected_out_layout = "" if in_layout is None else in_layout

    assert out[0].layout() == expected_out_layout


def test_transpose_layout():
    batch_size = 3
    for device in {"cpu", "gpu"}:
        for batch_size in (1, 3):
            for shape in [(600, 400, 3), (600, 400, 1)]:
                for permutation, in_layout, transpose_layout, out_layout_arg in [
                    ((2, 0, 1), "HWC", True, None),
                    ((2, 0, 1), "HWC", True, "CHW"),
                    ((2, 0, 1), "HWC", False, "CHW"),
                    ((1, 0, 2), None, False, None),
                    ((1, 0, 2), "XYZ", True, None),
                    ((1, 0, 2), None, None, "ABC"),
                ]:
                    yield (
                        check_transpose_layout,
                        device,
                        batch_size,
                        shape,
                        in_layout,
                        permutation,
                        transpose_layout,
                        out_layout_arg,
                    )


@params(*itertools.product(("cpu", "gpu"), ((10, 20, 3), (10, 20), (1,), (), (3, 3, 2, 2, 3))))
def test_transpose_default(device, shape):
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        data = fn.random.uniform(range=[0, 255], shape=shape, device=device)
        ndim = len(shape) or 0
        perm = [d - 1 for d in range(ndim, 0, -1)]
        return fn.transpose(data), fn.transpose(data, perm=perm)

    p = pipe()
    out_default, out_explicit = [as_array(o[0]) for o in p.run()]
    np.testing.assert_array_equal(out_explicit, out_default)
