# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.fn as fn
import numpy as np
from nose_utils import raises


def get_sequence(shape, offset=0):
    assert len(shape) > 1
    elem_shape = shape.copy()
    seq_length = elem_shape[0]
    elem_shape[0] = 1
    elems = []
    for i in range(seq_length):
        elems.append(np.full(elem_shape, offset + i))
    return np.concatenate(elems, axis=0)


def get_sequences(batch_size, shape):
    batch = []
    for i in range(batch_size):
        batch.append(get_sequence(shape, i * shape[0]))
    return batch


def reorder_sample(sample, seq_len, order):
    """
    Reorder sequence in one sample according to order parameter
    """
    split = np.split(sample, seq_len)
    reordered = []
    for i in range(len(order)):
        reordered.append(split[order[i]])
    return np.concatenate(reordered, axis=0)


def reorder(input, seq_len, reorders, persample_reorder=True):
    """
    Reorder the whole batch of sequences according to `reorders`
    reorders is one list with new order or list of new_orders depending on `persample_reorder`
    """
    result = []
    for i, sample in enumerate(input):
        order = reorders[i] if persample_reorder else reorders
        result.append(reorder_sample(sample, seq_len, order))
    return result


def to_batch(tl, batch_size):
    return [np.array(tl[i]) for i in range(batch_size)]


def check_sequence_rearrange(
    batch_size, shape, reorders, persample_reorder=True, op_type="cpu", layout=""
):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    with pipe:
        input = fn.external_source(lambda: get_sequences(batch_size, shape), layout=layout)
        frames = input.gpu() if op_type == "gpu" else input
        order = fn.external_source(lambda: reorders) if persample_reorder else reorders
        rearranged = fn.sequence_rearrange(frames, new_order=order, device=op_type)
        pipe.set_outputs(rearranged, input)
    result, input = pipe.run()
    if op_type == "gpu":
        result = result.as_cpu()
    input = to_batch(input, batch_size)
    baseline = reorder(input, shape[0], reorders, persample_reorder)
    for i in range(batch_size):
        np.testing.assert_array_equal(result[i], baseline[i])


order_0 = ([3, 2, 1, 0], False)

order_1 = (
    [np.int32([3, 0]), np.int32([2, 1]), np.int32([1, 1]), np.int32([0, 1, 2]), np.int32([3])],
    True,
)

order_2 = (
    [np.int32([0]), np.int32([1]), np.int32([2]), np.int32([3]), np.int32([0, 1, 2, 3])],
    True,
)


def test_sequence_rearrange():
    for dev in ["cpu", "gpu"]:
        for shape in [[4, 3, 2], [5, 1]]:
            for new_order, per_sample in [order_0, order_1, order_2]:
                for layout in ["FHW"[: len(shape)], ""]:
                    yield check_sequence_rearrange, 5, shape, new_order, per_sample, dev, layout


def check_fail_sequence_rearrange(
    batch_size, shape, reorders, persample_reorder=True, op_type="cpu", layout=""
):
    check_sequence_rearrange(batch_size, shape, reorders, persample_reorder, op_type, layout)


def test_fail_sequence_rearrange():
    shape = [5, 1]
    orders = [
        ([6, 7], False),
        ([-1], False),
        ([], False),
        ([np.int32([0]), np.int32([])], True),
        ([np.int32([6, 7]), np.int32([0])], True),
        ([np.int32([-1]), np.int32([0])], True),
        ([np.int32([[1], [2]]), np.int32([[1], [2]])], True),
    ]
    error_msgs = [
        "new_order[[]*[]] must be between * and input_sequence_length = * for sample *, but it is: *",  # noqa:E501
        "new_order[[]*[]] must be between * and input_sequence_length = * for sample *, but it is: *",  # noqa:E501
        "Empty result sequences are not allowed",
        "Empty `new_order` for sample * is not allowed",
        "new_order[[]*[]] must be between * and input_sequence_length = * for sample *, but it is: *",  # noqa:E501
        "new_order[[]*[]] must be between * and input_sequence_length = * for sample *, but it is: *",  # noqa:E501
        "Input with dimension * cannot be converted to dimension *",
    ]

    assert len(orders) == len(error_msgs)

    for dev in ["cpu", "gpu"]:
        for [new_order, per_sample], error_msg in zip(orders, error_msgs):
            yield raises(RuntimeError, glob=error_msg)(
                check_fail_sequence_rearrange
            ), 2, shape, new_order, per_sample, dev


def test_wrong_layouts_sequence_rearrange():
    shape = [5, 1]
    new_order = [0, 2, 1, 3, 4]
    per_sample = False
    for dev in ["cpu", "gpu"]:
        for layout in ["HF", "HW"]:
            yield raises(
                RuntimeError,
                glob=(
                    "Expected sequence as the input, where outermost dimension represents"
                    ' frames dimension `F`, got data with layout = "H[WF]"'
                ),
            )(check_fail_sequence_rearrange), 5, shape, new_order, per_sample, dev, layout
