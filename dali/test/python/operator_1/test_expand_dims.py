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


from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import numpy as np
from nose_utils import assert_raises


def get_data(shapes):
    return [np.empty(shape, dtype=np.uint8) for shape in shapes]


@pipeline_def
def expand_dims_pipe(shapes, axes=None, new_axis_names=None, layout=None):
    data = fn.external_source(lambda: get_data(shapes), layout=layout, batch=True, device="cpu")
    return fn.expand_dims(data, axes=axes, new_axis_names=new_axis_names)


def _testimpl_expand_dims(
    axes, new_axis_names, layout, shapes, expected_out_shapes, expected_layout
):
    batch_size = len(shapes)
    pipe = expand_dims_pipe(
        batch_size=batch_size,
        num_threads=1,
        device_id=0,
        shapes=shapes,
        axes=axes,
        new_axis_names=new_axis_names,
        layout=layout,
    )
    for _ in range(3):
        outs = pipe.run()
        assert outs[0].layout() == expected_layout
        for i in range(batch_size):
            out_arr = np.array(outs[0][i])
            assert out_arr.shape == expected_out_shapes[i]


def test_expand_dims():
    # axes, new_axis_names, layout, shapes, expected_shapes, expected_layout
    args = [
        ([0, 2], "AB", "XYZ", [(10, 20, 30)], [(1, 10, 1, 20, 30)], "AXBYZ"),
        ([0, 3], None, "XYZ", [(10, 20, 30)], [(1, 10, 20, 1, 30)], ""),
        (
            [3],
            None,
            "XYZ",
            [(10, 20, 30), (100, 200, 300)],
            [(10, 20, 30, 1), (100, 200, 300, 1)],
            "",
        ),
        (
            [4, 3],
            None,
            "XYZ",
            [(10, 20, 30), (100, 200, 300)],
            [(10, 20, 30, 1, 1), (100, 200, 300, 1, 1)],
            "",
        ),
        (
            [0, 1, 3, 5, 7],
            "ABCDE",
            "XYZ",
            [(11, 22, 33)],
            [(1, 1, 11, 1, 22, 1, 33, 1)],
            "ABXCYDZE",
        ),
        ([], "", "HW", [(10, 20)], [(10, 20)], "HW"),
        ([0, 1], "", "", [()], [(1, 1)], ""),
        ([0], "", "HW", [(10, 20)], [(1, 10, 20)], ""),
        ([4, 3], "AB", "XYZ", [(10, 20, 30)], [(10, 20, 30, 1, 1)], "XYZBA"),
        ([0], "X", "", [()], [(1,)], "X"),
    ]
    for axes, new_axis_names, layout, shapes, expected_out_shapes, expected_layout in args:
        yield (
            _testimpl_expand_dims,
            axes,
            new_axis_names,
            layout,
            shapes,
            expected_out_shapes,
            expected_layout,
        )


def test_expand_dims_throw_error():
    args = [
        (
            [4],
            None,
            None,
            [(10, 20, 30)],
            r"Data has not enough dimensions to add new axes at specified indices.",
        ),
        ([0, -1], None, None, [(10, 20, 30)], r"Axis value can't be negative"),
        (
            [2, 0, 2],
            "AB",
            "XYZ",
            [(10, 20, 30)],
            r"Specified [\d]+ new dimensions, but layout contains only [\d]+ new dimension names",
        ),
        (
            [2],
            "C",
            None,
            [(10, 20, 30)],
            r"Specifying ``new_axis_names`` requires an input with a proper layout.",
        ),
    ]
    for axes, new_axis_names, layout, shapes, err_msg in args:
        pipe = expand_dims_pipe(
            batch_size=len(shapes),
            num_threads=1,
            device_id=0,
            shapes=shapes,
            axes=axes,
            new_axis_names=new_axis_names,
            layout=layout,
        )
        with assert_raises(RuntimeError, regex=err_msg):
            pipe.run()
