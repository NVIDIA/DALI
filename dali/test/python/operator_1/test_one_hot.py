# Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from test_utils import check_batch
from nose_utils import raises
import numpy as np


num_classes = 20
batch_size = 10


def insert_as_axis(target, value, axis, total_axes):
    return target[0:axis] + (value,) + target[axis:total_axes]


def get_initial_layout(sample_dim=0, base="ABCD"):
    return base[0:sample_dim]


def modify_layout(layout, output_dim, axis=None, axis_name=None):
    if not axis_name or (not layout and output_dim > 1):
        return ""
    if output_dim == 1:
        return axis_name
    layout = layout or ""
    if axis < 0:
        axis = len(layout)
    return layout[:axis] + axis_name + layout[axis:]


def random_3d_tensors_batch():
    return [
        np.random.randint(0, num_classes, size=np.random.randint(2, 8, size=(3,)), dtype=np.int32)
        for _ in range(batch_size)
    ]


def random_scalars_batch():
    return np.random.randint(0, num_classes, size=batch_size, dtype=np.int32)


def random_scalar_like_tensors_batch(nested_level):
    return [
        np.array([np.random.randint(0, num_classes)], dtype=np.int32).reshape((1,) * nested_level)
        for x in range(batch_size)
    ]


class OneHotPipeline(Pipeline):
    def __init__(
        self, num_classes, source, axis=-1, num_threads=1, layout=None, axis_name=None, device="cpu"
    ):
        super(OneHotPipeline, self).__init__(batch_size, num_threads, 0)
        self.is_gpu = device == "gpu"
        self.ext_src = ops.ExternalSource(source=source, layout=layout)
        self.one_hot = ops.OneHot(
            num_classes=num_classes,
            axis=axis,
            dtype=types.INT32,
            device=device,
            axis_name=axis_name,
        )

    def define_graph(self):
        self.data = self.ext_src()
        if self.is_gpu:
            self.data = self.data.gpu()
        return self.one_hot(self.data), self.data


def one_hot_3_axes(input, axis):
    total_axes = len(input[0].shape)
    assert total_axes == 3
    axis = axis if axis >= 0 else total_axes
    shapes = []
    results = []
    for i in range(batch_size):
        shape = insert_as_axis(input[i].shape, num_classes, axis, total_axes)
        result = np.zeros(shape, dtype=np.int32)
        shapes.append(shape)
        for i0 in range(input[i].shape[0]):
            for i1 in range(input[i].shape[1]):
                for i2 in range(input[i].shape[2]):
                    in_coord = (i0, i1, i2)
                    out_coord = insert_as_axis(in_coord, input[i][in_coord], axis, total_axes)
                    result[out_coord] = 1
        results.append(result)
    return results


def one_hot(input):
    outp = np.zeros([batch_size, num_classes], dtype=np.int32)
    for i in range(batch_size):
        outp[i, int(input[i])] = 1
    return outp


def check_one_hot_operator(
    source, device="cpu", axis=-1, expected_output_dim=None, axis_name=None, initial_layout=None
):
    pipeline = OneHotPipeline(
        num_classes=num_classes,
        source=source,
        axis=axis,
        layout=initial_layout,
        axis_name=axis_name,
        device=device,
    )
    (outputs, input_batch) = pipeline.run()
    if device == "gpu":
        input_batch = input_batch.as_cpu()
    input_batch = list(map(np.array, input_batch))
    expected_output_dim = expected_output_dim or len(input_batch[0].shape) + 1
    reference = (
        one_hot_3_axes(input_batch, axis) if expected_output_dim == 4 else one_hot(input_batch)
    )
    expected_layout = modify_layout(initial_layout, expected_output_dim, axis, axis_name)
    check_batch(
        outputs, reference, batch_size, max_allowed_error=0, expected_layout=expected_layout
    )


def test_one_hot_scalar():
    np.random.seed(42)
    for device in ["cpu", "gpu"]:
        for i in range(10):
            yield partial(check_one_hot_operator, axis_name="O"), random_scalars_batch, device


def test_one_hot_legacy():
    np.random.seed(42)
    for device in ["cpu", "gpu"]:
        for j in range(1, 5):  # test 1..4 levels of nested 'multi-dimensional' scalars
            layout = get_initial_layout(j)

            class RandomScalarLikeTensors:
                def __init__(self, i):
                    self.i = i

                def __call__(self):
                    return random_scalar_like_tensors_batch(self.i)

            for i in range(5):
                yield partial(
                    check_one_hot_operator,
                    axis=None,
                    axis_name="O",
                    expected_output_dim=1,
                    initial_layout=layout,
                ), RandomScalarLikeTensors(j), device


def test_one_hot():
    np.random.seed(42)
    for device in ["cpu", "gpu"]:
        layout = get_initial_layout(3)
        for i in range(10):
            for axis in [-1, 0, 1, 2, 3]:
                yield partial(
                    check_one_hot_operator, axis_name="O", initial_layout=layout
                ), random_3d_tensors_batch, device, axis


def test_multi_dim_one_hot_no_initial_layout():
    np.random.seed(42)
    for axis in [-1, 0, 1, 2, 3]:
        yield partial(
            check_one_hot_operator, initial_layout=None
        ), random_3d_tensors_batch, "cpu", axis


def test_one_hot_reset_layout():
    np.random.seed(42)
    layout = get_initial_layout(3)
    for axis in [-1, 0, 1, 2, 3]:
        yield partial(
            check_one_hot_operator, initial_layout=layout
        ), random_3d_tensors_batch, "cpu", axis

    yield check_one_hot_operator, random_scalars_batch

    def random_scalar_like_tensors():
        return random_scalar_like_tensors_batch(3)

    yield partial(
        check_one_hot_operator, axis=None, expected_output_dim=1, initial_layout=layout
    ), random_scalar_like_tensors, "cpu"


def test_one_hot_custom_layout_axis_name():
    np.random.seed(42)
    layout = get_initial_layout(3)
    for axis_name in "Xx01":
        yield partial(
            check_one_hot_operator, axis=-1, initial_layout=layout, axis_name=axis_name
        ), random_3d_tensors_batch


@raises(RuntimeError, glob="Unsupported axis_name value")
def test_too_long_axis_name():
    np.random.seed(42)
    check_one_hot_operator(random_3d_tensors_batch, axis=-1, initial_layout="ABC", axis_name="CD")


@raises(RuntimeError, glob="Unsupported axis_name value")
def test_empty_string_axis_name():
    np.random.seed(42)
    check_one_hot_operator(random_3d_tensors_batch, axis=-1, initial_layout="ABC", axis_name="")


@raises(RuntimeError, glob="Input layout mismatch")
def test_axis_name_no_initial_layout_multi_dim():
    np.random.seed(42)
    check_one_hot_operator(random_3d_tensors_batch, axis=-1, axis_name="O")
