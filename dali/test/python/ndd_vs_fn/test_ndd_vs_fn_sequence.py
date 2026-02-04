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


import numpy as np
from nose2.tools import params
import nvidia.dali.types as types
from ndd_vs_fn_test_utils import (
    OperatorTestConfig,
    run_operator_test,
    custom_shape_generator,
    flatten_operator_configs,
    generate_data,
)


SEQUENCE_OPERATORS = [
    OperatorTestConfig("cast", {"dtype": types.INT32}),
    OperatorTestConfig("copy"),
    OperatorTestConfig("crop", {"crop": (5, 5)}),
    OperatorTestConfig("crop_mirror_normalize", {"mirror": 1, "output_layout": "FCHW"}),
    OperatorTestConfig(
        "erase",
        {
            "anchor": [0.3],
            "axis_names": "H",
            "normalized_anchor": True,
            "shape": [0.1],
            "normalized_shape": True,
        },
    ),
    OperatorTestConfig("flip", {"horizontal": True}),
    OperatorTestConfig("gaussian_blur", {"window_size": 5}),
    OperatorTestConfig("normalize", {"batch": True}),
    OperatorTestConfig("resize", {"resize_x": 50, "resize_y": 50}),
    OperatorTestConfig("per_frame", devices=["cpu"]),
    OperatorTestConfig("reinterpret", {"rel_shape": [1, -1]}),
]

sequence_ops_test_configuration = flatten_operator_configs(SEQUENCE_OPERATORS)


@params(*sequence_ops_test_configuration)
def test_sequence_operators(device, fn_operator, ndd_operator, operator_args):
    data = generate_data(
        custom_shape_generator(3, 7, 160, 200, 80, 100, 3, 3),
        lo=0,
        hi=255,
        dtype=np.uint8,
    )
    run_operator_test(
        input_epoch=data,
        fn_operator=fn_operator,
        ndd_operator=ndd_operator,
        device=device,
        operator_args=operator_args,
        input_layout="FHWC",
    )


tested_operators = [
    "cast",
    "copy",
    "crop",
    "crop_mirror_normalize",
    "erase",
    "flip",
    "gaussian_blur",
    "normalize",
    "per_frame",
    "resize",
    "reinterpret",
]
