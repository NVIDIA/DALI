# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali as dali
import nvidia.dali.fn as fn
import cv2
import numpy as np
import multiprocessing as mp
import test_utils
from nose2.tools import params

NUM_THREADS = mp.cpu_count()
DEV_ID = 0
SEED = 1313


def ocv_border_mode(border_mode):
    if border_mode == "constant":
        return cv2.BORDER_CONSTANT
    elif border_mode == "replicate":
        return cv2.BORDER_REPLICATE
    elif border_mode == "reflect":
        return cv2.BORDER_REFLECT
    elif border_mode == "reflect_101":
        return cv2.BORDER_REFLECT_101
    elif border_mode == "wrap":
        return cv2.BORDER_WRAP
    else:
        raise ValueError("Invalid border mode")


def ocv_interp_type(interp_type):
    if interp_type == "nearest":
        return cv2.INTER_NEAREST
    elif interp_type == "linear":
        return cv2.INTER_LINEAR
    elif interp_type == "cubic":
        return cv2.INTER_CUBIC
    else:
        raise ValueError("Invalid interpolation type")


def cv2_warp_perspective(
    dst, img, matrix, layout, border_mode, interp_type, inverse_map, fill_value=None
):
    border_mode = ocv_border_mode(border_mode)
    interp_type = ocv_interp_type(interp_type)
    flags = interp_type

    if fill_value is None:
        fill_value = 0

    if inverse_map:
        flags = flags | cv2.WARP_INVERSE_MAP
    if layout[-1] == "C":
        dsize = (dst.shape[1], dst.shape[0])
        if not isinstance(fill_value, tuple):
            fill_value = tuple([fill_value for c in range(dst.shape[2])])
        dst[:, :, :] = cv2.warpPerspective(
            img, M=matrix, dsize=dsize, flags=flags, borderMode=border_mode, borderValue=fill_value
        ).reshape(dst.shape)
    else:
        dsize = (dst.shape[2], dst.shape[1])
        for c in range(img.shape[0]):
            dst[c, :, :] = cv2.warpPerspective(
                img[c, :, :],
                M=matrix,
                dsize=dsize,
                flags=flags,
                borderMode=border_mode,
                borderValue=fill_value,
            ).reshape(dst.shape[1:])


def ref_func(img, matrix, size, layout, border_mode, interp_type, inverse_map, fill_value):
    out_shape = list(img.shape)
    if size is not None:
        out_shape[layout.find("H")] = size[0]
        out_shape[layout.find("W")] = size[1]
    dst = np.zeros(out_shape, dtype=img.dtype)
    if layout[0] == "F":
        for f in range(0, img.shape[0]):
            cv2_warp_perspective(
                dst[f, :, :, :],
                img[f, :, :, :],
                matrix,
                layout,
                border_mode,
                interp_type,
                inverse_map,
                fill_value,
            )
    else:
        cv2_warp_perspective(
            dst, img, matrix, layout, border_mode, interp_type, inverse_map, fill_value
        )
    return dst


@dali.pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def reference_pipe(
    data_src, matrix_src, size, layout, border_mode, interp_type, inverse_map, fill_value
):
    img = fn.external_source(source=data_src, batch=True, layout=layout)
    matrix = fn.external_source(source=matrix_src)
    return fn.python_function(
        img,
        matrix,
        function=lambda im, mx: ref_func(
            im,
            mx,
            size=size,
            layout=layout,
            border_mode=border_mode,
            interp_type=interp_type,
            inverse_map=inverse_map,
            fill_value=fill_value,
        ),
        batch_processing=False,
        output_layouts=layout,
    )


def matrix_source(batch_size, seed, constant=False):
    np_rng = np.random.default_rng(seed=seed)

    def gen_matrix():
        dst_pts = np.array([[0, 0], [0, 100], [100, 0], [100, 100]], dtype=np.float32)
        src_offsets = np_rng.random((4, 2), dtype=np.float32) * 50 - 25
        src_pts = dst_pts - src_offsets
        return cv2.getPerspectiveTransform(src_pts, dst_pts).astype(np.float32)

    if constant:
        m = gen_matrix()
        while True:
            yield [m for _ in range(batch_size)]
    else:
        while True:
            yield [gen_matrix() for _ in range(batch_size)]


@dali.pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def warp_perspective_pipe_const_matrix(
    data_src, layout, matrix, size, border_mode, interp_type, inverse_map, fill_value
):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device="gpu")
    return fn.experimental.warp_perspective(
        img,
        matrix=matrix.reshape((-1)),
        size=size,
        border_mode=border_mode,
        interp_type=interp_type,
        inverse_map=inverse_map,
        fill_value=fill_value,
    )


@dali.pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def warp_perspective_pipe_arg_inp_matrix(
    data_src, matrix_src, layout, size, border_mode, interp_type, inverse_map, fill_value
):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device="gpu")
    matrix = fn.external_source(source=matrix_src, batch=True)
    matrix = fn.reshape(matrix, shape=(-1))
    return fn.experimental.warp_perspective(
        img,
        matrix=matrix,
        size=size,
        border_mode=border_mode,
        interp_type=interp_type,
        inverse_map=inverse_map,
        fill_value=fill_value,
    )


@dali.pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def warp_perspective_pipe_gpu_inp_matrix(
    data_src, matrix_src, layout, size, border_mode, interp_type, inverse_map, fill_value
):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device="gpu")
    matrix = fn.external_source(source=matrix_src, batch=True, device="gpu")
    matrix = fn.reshape(matrix, shape=(-1))
    return fn.experimental.warp_perspective(
        img,
        matrix,
        size=size,
        border_mode=border_mode,
        interp_type=interp_type,
        inverse_map=inverse_map,
        fill_value=fill_value,
    )


def input_iterator(bs, layout, dtype, channels):
    cdim = layout.find("C")
    min_shape = [64 for d in layout]
    min_shape[cdim] = channels
    max_shape = [256 for d in layout]
    max_shape[cdim] = channels
    if layout[0] == "F":
        min_shape[0] = 8
        max_shape[0] = 32

    return test_utils.RandomlyShapedDataIterator(
        batch_size=bs, min_shape=min_shape, max_shape=max_shape, dtype=dtype, seed=SEED
    )


def compare_pipelines(pipe1, pipe2, bs, dtype):
    if dtype == np.float32:
        eps = 0.05
    elif dtype == np.uint8:
        eps = 1
    elif dtype == np.int16 or dtype == np.uint16:
        eps = 5

    test_utils.compare_pipelines(pipe1, pipe2, batch_size=bs, N_iterations=10, eps=eps)


counter = 1


@params(
    (32, "HWC", np.uint8, 3, None, "constant", "nearest", False, (100, 50, 25)),
    (32, "HWC", np.uint8, 3, None, "constant", "linear", False, 77),
    (4, "FCHW", np.uint8, 1, (200, 300), "reflect", "nearest", True, None),
    (32, "CHW", np.float32, 1, (300, 300), "replicate", "cubic", False, None),
    (32, "CHW", np.float32, 4, (150, 300), "constant", "nearest", False, 55),
    (8, "FHWC", np.int16, 3, None, "reflect_101", "linear", True, None),
    (32, "HWC", np.uint16, 4, None, "constant", "cubic", False, None),
)
def test_warp_perspective_const_matrix_vs_ocv(
    bs, layout, dtype, channels, size, border_mode, interp_type, inverse_map, fill_value
):
    data1 = input_iterator(bs, layout, dtype, channels)
    data2 = input_iterator(bs, layout, dtype, channels)

    global counter
    matrix_src = matrix_source(bs, SEED + counter, True)
    matrix = matrix_src.__next__()[0]
    pipe1 = warp_perspective_pipe_const_matrix(
        data1,
        layout,
        matrix,
        size,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )

    pipe2 = reference_pipe(
        data2,
        matrix_src,
        size,
        layout,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )
    compare_pipelines(pipe1, pipe2, bs, dtype)
    counter = counter + 1


@params(
    (32, "HWC", np.uint8, 3, None, "constant", "nearest", False, (100, 50, 25)),
    (32, "HWC", np.uint8, 3, None, "constant", "linear", False, 77),
    (32, "CHW", np.float32, 1, (300, 300), "replicate", "cubic", False, None),
    (4, "FCHW", np.uint8, 1, (150, 150), "reflect", "linear", True, None),
    (32, "CHW", np.float32, 4, (150, 300), "constant", "nearest", False, 55),
    (8, "FHWC", np.int16, 3, None, "reflect_101", "linear", True, None),
    (32, "HWC", np.uint16, 4, None, "constant", "cubic", False, None),
)
def test_warp_perspective_arg_inp_matrix_vs_ocv(
    bs, layout, dtype, channels, size, border_mode, interp_type, inverse_map, fill_value
):
    data1 = input_iterator(bs, layout, dtype, channels)
    data2 = input_iterator(bs, layout, dtype, channels)

    global counter
    matrix_src1 = matrix_source(bs, SEED + counter, False)
    pipe1 = warp_perspective_pipe_arg_inp_matrix(
        data1,
        matrix_src1,
        layout,
        size,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )

    matrix_src2 = matrix_source(bs, SEED + counter, False)
    pipe2 = reference_pipe(
        data2,
        matrix_src2,
        size,
        layout,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )
    compare_pipelines(pipe1, pipe2, bs, dtype)
    counter = counter + 1


@params(
    (32, "HWC", np.uint8, 3, None, "constant", "nearest", False, (100, 50, 25)),
    (32, "HWC", np.uint8, 3, None, "constant", "linear", False, 77),
    (32, "CHW", np.float32, 1, (300, 300), "replicate", "cubic", False, None),
    (4, "FCHW", np.uint8, 1, (150, 150), "reflect", "linear", True, None),
    (32, "CHW", np.float32, 4, (150, 300), "constant", "nearest", False, 55),
    (8, "FHWC", np.int16, 3, None, "reflect_101", "linear", True, None),
    (32, "HWC", np.uint16, 4, None, "constant", "cubic", False, None),
)
def test_warp_perspective_gpu_inp_matrix_vs_ocv(
    bs, layout, dtype, channels, size, border_mode, interp_type, inverse_map, fill_value
):
    data1 = input_iterator(bs, layout, dtype, channels)
    data2 = input_iterator(bs, layout, dtype, channels)

    global counter
    matrix_src1 = matrix_source(bs, SEED + counter, False)
    pipe1 = warp_perspective_pipe_gpu_inp_matrix(
        data1,
        matrix_src1,
        layout,
        size,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )

    matrix_src2 = matrix_source(bs, SEED + counter, False)
    pipe2 = reference_pipe(
        data2,
        matrix_src2,
        size,
        layout,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )
    compare_pipelines(pipe1, pipe2, bs, dtype)
    counter = counter + 1
