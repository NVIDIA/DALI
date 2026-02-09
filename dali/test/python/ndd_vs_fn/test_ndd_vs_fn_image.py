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
    flatten_operator_configs,
    generate_image_like_data,
)
from test_ndd_vs_fn_coverage import register_operator_test


IMAGE_LIKE_OPERATORS = [
    # Operators with default arguments (tested on both CPU and GPU)
    OperatorTestConfig("brightness"),
    OperatorTestConfig("brightness_contrast"),
    OperatorTestConfig("color_twist"),
    OperatorTestConfig("contrast"),
    OperatorTestConfig("copy"),
    OperatorTestConfig("hsv"),
    OperatorTestConfig("hue"),
    OperatorTestConfig("reductions.mean"),
    OperatorTestConfig("reductions.mean_square"),
    OperatorTestConfig("reductions.rms"),
    OperatorTestConfig("reductions.min"),
    OperatorTestConfig("reductions.max"),
    OperatorTestConfig("reductions.sum"),
    OperatorTestConfig("saturation"),
    OperatorTestConfig("sphere"),
    OperatorTestConfig("water"),
    OperatorTestConfig("dump_image"),
    OperatorTestConfig("jpeg_compression_distortion"),
    OperatorTestConfig("crop_mirror_normalize"),
    OperatorTestConfig("stack"),
    OperatorTestConfig("cat"),
    # Operators with custom arguments:
    OperatorTestConfig("cast", {"dtype": types.INT32}),
    OperatorTestConfig("coord_transform", {"M": 0.5, "T": 2}),
    OperatorTestConfig("coord_transform", {"T": 2}),
    OperatorTestConfig("coord_transform", {"M": 0.5}),
    OperatorTestConfig("crop", {"crop": (5, 5)}),
    OperatorTestConfig("gaussian_blur", {"window_size": 5}),
    OperatorTestConfig("get_property", {"key": "layout"}),
    OperatorTestConfig("laplacian", {"window_size": 3}),
    OperatorTestConfig("laplacian", {"window_size": 3, "smoothing_size": 1}),
    OperatorTestConfig("laplacian", {"window_size": 3, "normalized_kernel": True}),
    OperatorTestConfig("normalize", {"batch": True}),
    OperatorTestConfig("rotate", {"angle": 25}),
    OperatorTestConfig("transpose", {"perm": [2, 0, 1]}),
    OperatorTestConfig("grid_mask", {"angle": 2.6810782, "ratio": 0.38158387, "tile": 51}),
    OperatorTestConfig(
        "multi_paste",
        {"in_ids": np.zeros([31], dtype=np.int32), "output_size": [300, 300, 3]},
    ),
    OperatorTestConfig("pad", {"fill_value": -1, "axes": (0,), "shape": (10,)}),
    OperatorTestConfig("paste", {"fill_value": 42, "ratio": 1}),
    OperatorTestConfig("reshape", {"shape": (1, 1, -1)}),
    OperatorTestConfig(
        "color_space_conversion", {"image_type": types.BGR, "output_type": types.RGB}
    ),
    OperatorTestConfig("experimental.warp_perspective", {"matrix": np.eye(3)}),
    OperatorTestConfig("flip", {"horizontal": True}),
    OperatorTestConfig("resize", {"resize_x": 50, "resize_y": 50}),
    OperatorTestConfig("tensor_resize", {"sizes": [50, 50], "axes": [0, 1]}),
    OperatorTestConfig("reinterpret", {"rel_shape": [-1]}),
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
    OperatorTestConfig("pad", {"fill_value": -1, "axes": (0,), "align": 16}),
    OperatorTestConfig("expand_dims", {"axes": 1, "new_axis_names": "Z"}),
    # CPU-only operators:
    OperatorTestConfig("zeros_like", devices=["cpu"]),
    OperatorTestConfig("ones_like", devices=["cpu"]),
    OperatorTestConfig("per_frame", {"replace": True}, devices=["cpu"]),
    OperatorTestConfig(
        "resize_crop_mirror", {"crop": [5, 5], "resize_shorter": 10}, devices=["cpu"]
    ),
    # GPU-only operators:
    OperatorTestConfig("clahe", {"tiles_x": 4, "tiles_y": 4, "clip_limit": 2.0}, devices=["gpu"]),
    OperatorTestConfig("equalize", devices=["gpu"]),
    OperatorTestConfig("slice", {"rel_start": 0.1, "rel_end": 0.5}),
    OperatorTestConfig("experimental.median_blur", devices=["gpu"]),
    OperatorTestConfig("experimental.dilate", devices=["gpu"]),
    OperatorTestConfig("experimental.erode", devices=["gpu"]),
    OperatorTestConfig("experimental.resize", {"resize_x": 50, "resize_y": 50}, devices=["gpu"]),
]


ops_image_like_test_configuration = flatten_operator_configs(IMAGE_LIKE_OPERATORS)


@params(*ops_image_like_test_configuration)
def test_operators_with_image_like_input(
    device, operator_name, fn_operator, ndd_operator, operator_args
):
    """Example: run_operator_test with image-like data and HWC layout.

    Test config comes from flatten_operator_configs(IMAGE_LIKE_OPERATORS), which
    yields (device, operator_name, fn_operator, ndd_operator, operator_args).
    """
    register_operator_test(operator_name)
    data = generate_image_like_data()
    run_operator_test(
        input_epoch=data,
        fn_operator=fn_operator,
        ndd_operator=ndd_operator,
        device=device,
        operator_args=operator_args,
        input_layout="HWC",
    )
