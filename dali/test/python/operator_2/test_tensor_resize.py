# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.types as types
from nvidia.dali import pipeline_def, fn
from nose2.tools import params
import numpy as np
from test_utils import as_array

devices = ("cpu", "gpu")

data_1x1x2x2 = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
data_1x1x2x4 = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], dtype=np.float32)

data_1x1x4x4 = np.array(
    [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32
)


def run_and_compare(expected, data, device, resize_fn):
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        input_data = types.Constant(data, device=device)
        return resize_fn(input_data)

    p = pipe()
    out = p.run()
    np.testing.assert_allclose(expected, as_array(out[0][0]), rtol=1e-3, atol=1e-7)


@params("cpu", "gpu")
def test_resize_upsample_scales_nearest(device):
    data = data_1x1x2x2
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

    # from onnx.backend.test.case.node.resize import interpolate_nd, nearest_coeffs
    # expected = interpolate_nd(data, lambda x, _: nearest_coeffs(x), scale_factors=scales)
    expected = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                ]
            ]
        ],
        dtype=np.float32,
    )

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, scales=scales, alignment=0, interp_type=types.INTERP_NN
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_downsample_scales_nearest(device):
    data = data_1x1x2x4
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

    # from onnx.backend.test.case.node.resize import interpolate_nd, nearest_coeffs
    # expected = interpolate_nd(data, lambda x, _: nearest_coeffs(x), scale_factors=scales)
    expected = np.array([[[[1.0, 3.0]]]], dtype=np.float32)

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, scales=scales, alignment=0, interp_type=types.INTERP_NN
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_upsample_sizes_nearest(device):
    data = data_1x1x2x2
    sizes = np.array([1.0, 1.0, 7.0, 8.0], dtype=np.float32)

    # from onnx.backend.test.case.node.resize import interpolate_nd, nearest_coeffs
    # expected = interpolate_nd(data, lambda x, _: nearest_coeffs(x, mode='round_prefer_ceil'),
    #                           output_size=sizes.astype(np.int64))
    expected = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                ]
            ]
        ],
        dtype=np.float32,
    )

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, sizes=sizes, alignment=0, interp_type=types.INTERP_NN
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_upsample_scales_linear(device):
    data = data_1x1x2x2
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # from onnx.backend.test.case.node.resize import interpolate_nd, linear_coeffs
    # expected = interpolate_nd(data, lambda x, _: linear_coeffs(x),
    #                           scale_factors=scales)
    expected = np.array(
        [
            [
                [
                    [1.0, 1.25, 1.75, 2.0],
                    [1.5, 1.75, 2.25, 2.5],
                    [2.5, 2.75, 3.25, 3.5],
                    [3.0, 3.25, 3.75, 4.0],
                ]
            ]
        ],
        dtype=np.float32,
    )

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, scales=scales, alignment=0, interp_type=types.INTERP_LINEAR, antialias=False
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_downsample_scales_linear(device):
    data = data_1x1x2x4
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

    # from onnx.backend.test.case.node.resize import interpolate_nd, linear_coeffs
    # expected = interpolate_nd(data, lambda x, _: linear_coeffs(x), scale_factors=scales)
    expected = np.array([[[[2.6666665, 4.3333331]]]], dtype=np.float32)

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, scales=scales, alignment=0, interp_type=types.INTERP_LINEAR, antialias=False
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_alignment(device):
    data = data_1x1x2x4
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

    def resize_fn(input_data, alignment=0):
        return fn.experimental.tensor_resize(
            input_data,
            scales=scales,
            alignment=alignment,
            interp_type=types.INTERP_LINEAR,
            antialias=False,
        )

    run_and_compare(
        np.array([[[[2.6666665, 4.3333331]]]], dtype=np.float32),
        data,
        device,
        lambda in_data: resize_fn(in_data, alignment=0),
    )
    run_and_compare(
        np.array([[[[3.6666665, 5.3333331]]]], dtype=np.float32),
        data,
        device,
        lambda in_data: resize_fn(in_data, alignment=0.5),
    )
    run_and_compare(
        np.array([[[[4.6666665, 6.3333331]]]], dtype=np.float32),
        data,
        device,
        lambda in_data: resize_fn(in_data, alignment=1),
    )


@params("cpu", "gpu")
def test_resize_upsample_scales_cubic(device):
    data = data_1x1x4x4
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # from onnx.backend.test.case.node.resize import interpolate_nd, cubic_coeffs
    # expected = interpolate_nd(data, lambda x, _: cubic_coeffs(x, A=-0.5), scale_factors=scales)
    expected = np.array(
        [
            [
                [
                    [
                        0.6484375,
                        0.8984375,
                        1.4453125,
                        1.96875,
                        2.46875,
                        2.9921875,
                        3.5390625,
                        3.7890625,
                    ],
                    [
                        1.6484375,
                        1.8984375,
                        2.4453125,
                        2.96875,
                        3.46875,
                        3.9921875,
                        4.5390625,
                        4.7890625,
                    ],
                    [
                        3.8359375,
                        4.0859375,
                        4.6328125,
                        5.15625,
                        5.65625,
                        6.1796875,
                        6.7265625,
                        6.9765625,
                    ],
                    [5.9296875, 6.1796875, 6.7265625, 7.25, 7.75, 8.2734375, 8.8203125, 9.0703125],
                    [
                        7.9296875,
                        8.1796875,
                        8.7265625,
                        9.25,
                        9.75,
                        10.2734375,
                        10.8203125,
                        11.0703125,
                    ],
                    [
                        10.0234375,
                        10.2734375,
                        10.8203125,
                        11.34375,
                        11.84375,
                        12.3671875,
                        12.9140625,
                        13.1640625,
                    ],
                    [
                        12.2109375,
                        12.4609375,
                        13.0078125,
                        13.53125,
                        14.03125,
                        14.5546875,
                        15.1015625,
                        15.3515625,
                    ],
                    [
                        13.2109375,
                        13.4609375,
                        14.0078125,
                        14.53125,
                        15.03125,
                        15.5546875,
                        16.1015625,
                        16.3515625,
                    ],
                ]
            ]
        ],
        dtype=np.float32,
    )

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, scales=scales, alignment=0, interp_type=types.INTERP_CUBIC, antialias=False
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_downsample_scales_cubic(device):
    data = data_1x1x4x4
    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

    # from onnx.backend.test.case.node.resize import interpolate_nd, cubic_coeffs
    # expected = interpolate_nd(data, lambda x, _: cubic_coeffs(x, A=-0.5), scale_factors=scales)
    expected = np.array(
        [
            [
                [
                    [1.38574215, 2.68359369, 4.00683586],
                    [6.57714832, 7.87499986, 9.19824203],
                    [11.87011699, 13.16796853, 14.4912107],
                ]
            ]
        ],
        dtype=np.float32,
    )

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, scales=scales, alignment=0, interp_type=types.INTERP_CUBIC, antialias=False
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_upsample_sizes_cubic(device):
    data = data_1x1x4x4
    sizes = np.array([1.0, 1.0, 9.0, 10.0], dtype=np.float32)

    # from onnx.backend.test.case.node.resize import interpolate_nd, cubic_coeffs
    # expected = interpolate_nd(data, lambda x, _: cubic_coeffs(x, A=-0.5),
    #                           output_size=sizes.astype(np.int64))
    expected = np.array(
        [
            [
                [
                    [
                        0.63671948,
                        0.76971948,
                        1.14771948,
                        1.60571948,
                        2.01021948,
                        2.41021948,
                        2.81471948,
                        3.27271948,
                        3.65071948,
                        3.78371948,
                    ],
                    [
                        1.36168519,
                        1.49468519,
                        1.87268519,
                        2.33068519,
                        2.73518519,
                        3.13518519,
                        3.53968519,
                        3.99768519,
                        4.37568519,
                        4.50868519,
                    ],
                    [
                        3.18610219,
                        3.31910219,
                        3.69710219,
                        4.15510219,
                        4.55960219,
                        4.95960219,
                        5.36410219,
                        5.82210219,
                        6.20010219,
                        6.33310219,
                    ],
                    [
                        5.14872222,
                        5.28172222,
                        5.65972222,
                        6.11772222,
                        6.52222222,
                        6.92222222,
                        7.32672222,
                        7.78472222,
                        8.16272222,
                        8.29572222,
                    ],
                    [6.9265, 7.0595, 7.4375, 7.8955, 8.3, 8.7, 9.1045, 9.5625, 9.9405, 10.0735],
                    [
                        8.70427778,
                        8.83727778,
                        9.21527778,
                        9.67327778,
                        10.07777778,
                        10.47777778,
                        10.88227778,
                        11.34027778,
                        11.71827778,
                        11.85127778,
                    ],
                    [
                        10.66689781,
                        10.79989781,
                        11.17789781,
                        11.63589781,
                        12.04039781,
                        12.44039781,
                        12.84489781,
                        13.30289781,
                        13.68089781,
                        13.81389781,
                    ],
                    [
                        12.49131481,
                        12.62431481,
                        13.00231481,
                        13.46031481,
                        13.86481481,
                        14.26481481,
                        14.66931481,
                        15.12731481,
                        15.50531481,
                        15.63831481,
                    ],
                    [
                        13.21628052,
                        13.34928052,
                        13.72728052,
                        14.18528052,
                        14.58978052,
                        14.98978052,
                        15.39428052,
                        15.85228052,
                        16.23028052,
                        16.36328052,
                    ],
                ]
            ]
        ],
        dtype=np.float32,
    )

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, sizes=sizes, alignment=0, interp_type=types.INTERP_CUBIC, antialias=False
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_downsample_sizes_cubic(device):
    data = data_1x1x4x4
    sizes = np.array([1.0, 1.0, 3.0, 3.0], dtype=np.float32)

    # from onnx.backend.test.case.node.resize import interpolate_nd, cubic_coeffs
    # expected = interpolate_nd(data, lambda x, _: cubic_coeffs(x, A=-0.5),
    #                           output_size=sizes.astype(np.int64))
    expected = np.array(
        [
            [
                [
                    [1.54398148, 2.93518519, 4.32638889],
                    [7.1087963, 8.5, 9.8912037],
                    [12.67361111, 14.06481481, 15.45601852],
                ]
            ]
        ],
        dtype=np.float32,
    )

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, sizes=sizes, alignment=0, interp_type=types.INTERP_CUBIC, antialias=False
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_upsample_resize_only_1d(device):
    data = data_1x1x2x2
    scales = np.array([1.0, 1.0, 1.0, 3.0], dtype=np.float32)
    expected = np.array(
        [[[[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 4.0, 4.0, 4.0]]]], dtype=np.float32
    )

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, scales=scales, alignment=0, interp_type=types.INTERP_NN
        )

    run_and_compare(expected, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_upsample_resize_only_noop(device):
    data = data_1x1x2x2
    scales = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, scales=scales, alignment=0, interp_type=types.INTERP_NN
        )

    run_and_compare(data, data, device, resize_fn)


@params("cpu", "gpu")
def test_resize_upsample_1d(device):
    data = np.array([1.0, 2.0], dtype=np.float32)
    scales = np.array([3.0], dtype=np.float32)
    expected = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)

    def resize_fn(input_data):
        return fn.experimental.tensor_resize(
            input_data, scales=scales, alignment=0, interp_type=types.INTERP_NN
        )

    run_and_compare(expected, data, device, resize_fn)
