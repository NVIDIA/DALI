# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

from nose_utils import raises


def get_data(shapes):
    return [np.empty(shape, dtype=np.uint8) for shape in shapes]


@pipeline_def
def squeeze_pipe(shapes, axes=None, axis_names=None, layout=None):
    data = fn.external_source(lambda: get_data(shapes), layout=layout, batch=True, device="cpu")
    return fn.squeeze(data, axes=axes, axis_names=axis_names)


def _testimpl_squeeze(axes, axis_names, layout, shapes, expected_out_shapes, expected_layout):
    batch_size = len(shapes)
    pipe = squeeze_pipe(
        batch_size=batch_size,
        num_threads=1,
        device_id=0,
        shapes=shapes,
        axes=axes,
        axis_names=axis_names,
        layout=layout,
    )
    for _ in range(3):
        outs = pipe.run()
        assert outs[0].layout() == expected_layout
        for i in range(batch_size):
            out_arr = np.array(outs[0][i])
            assert out_arr.shape == expected_out_shapes[i]


def test_squeeze():
    # axes, axis_names, layout, shapes, expected_out_shapes, expected_layout
    args = [
        ([1], None, "XYZ", [(300, 1, 200), (10, 1, 10)], [(300, 200), (10, 10)], "XZ"),
        ([1, 2], None, "XYZ", [(300, 1, 1), (10, 1, 1)], [(300,), (10,)], "X"),
        ([0, 2], None, "XYZ", [(1, 300, 1), (1, 10, 1)], [(300,), (10,)], "Y"),
        (
            [0, 2],
            None,
            "ABCD",
            [(1, 1, 1, 1), (1, 1, 1, 1)],
            [
                (
                    1,
                    1,
                ),
                (1, 1),
            ],
            "BD",
        ),
        (None, "Z", "XYZ", [(300, 1, 1), (10, 1, 1)], [(300, 1), (10, 1)], "XY"),
        (None, "ZY", "XYZ", [(300, 1, 1), (10, 1, 1)], [(300,), (10,)], "X"),
        ([0], None, "X", [(1,)], [()], ""),
        ([1], None, "XYZ", [(100, 0, 0)], [(100, 0)], "XZ"),
        (None, "Z", "XYZ", [(100, 0, 0)], [(100, 0)], "XY"),
        (None, "X", "XYZ", [(100, 0, 0)], [(0, 0)], "YZ"),
    ]
    for axes, axis_names, layout, shapes, expected_out_shapes, expected_layout in args:
        yield (
            _testimpl_squeeze,
            axes,
            axis_names,
            layout,
            shapes,
            expected_out_shapes,
            expected_layout,
        )


def _test_squeeze_throw_error(axes, axis_names, layout, shapes):
    pipe = squeeze_pipe(
        batch_size=len(shapes),
        num_threads=1,
        device_id=0,
        shapes=shapes,
        axes=axes,
        axis_names=axis_names,
        layout=layout,
    )
    pipe.run()


def test_squeeze_throw_error():
    args_list = [
        ([1], None, None, [(300, 1, 200), (10, 10, 10)]),
        (None, "C", "XYZ", [(2, 3, 4), (4, 2, 3)]),
        (None, "Z", "XYZ", [(1, 1, 10)]),
        ([2], "Z", "XYZ", [[1, 1, 10]]),
        ([2, 1], None, "XYZ", [(100, 0, 0)]),
        ([1, 1], None, "XYZ", [(300, 1, 200), (10, 1, 10)]),
    ]
    expected_errors = [
        "Requested a shape with 100 elements but the original shape has 1000 elements.",
        "Axis 'C' is not present in the input layout",
        "Requested a shape with 1 elements but the original shape has 10 elements.",
        "Provided both ``axes`` and ``axis_names`` arguments",
        "Requested a shape with 100 elements but the original shape has 0 elements.",
        "Specified at least twice same dimension to remove.",
    ]
    assert len(expected_errors) == len(args_list)
    for (axes, axis_names, layout, shapes), error_msg in zip(args_list, expected_errors):
        yield raises(RuntimeError, error_msg)(
            _test_squeeze_throw_error
        ), axes, axis_names, layout, shapes
