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

import multiprocessing as mp

import cv2
import numpy as np
import test_utils
from nose2.tools import cartesian_params
from nose_utils import raises
from nvidia.dali import fn, types
from nvidia.dali.pipeline import Pipeline, pipeline_def
from nvidia.dali.types import DALIInterpType

NUM_THREADS = mp.cpu_count()
DEV_ID = 0
SEED = 1316


def ocv_border_mode(border_mode: str):
    try:
        return {
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP,
        }[border_mode]
    except KeyError as err:
        raise ValueError("Invalid border mode") from err


def ocv_interp_type(interp_type: DALIInterpType):
    try:
        return {
            DALIInterpType.INTERP_NN: cv2.INTER_NEAREST,
            DALIInterpType.INTERP_LINEAR: cv2.INTER_LINEAR,
            DALIInterpType.INTERP_CUBIC: cv2.INTER_CUBIC,
        }[interp_type]
    except KeyError as err:
        raise ValueError("Invalid interpolation type") from err


def ToCVMatrix(matrix):
    shift = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float32)
    shift_back = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]], dtype=np.float32)
    return np.matmul(shift_back, np.matmul(matrix, shift))


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
            fill_value = tuple(fill_value for c in range(dst.shape[2]))
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
    if size[0] > 0 and size[1] > 0:
        out_shape[layout.find("H")] = round(size[0])
        out_shape[layout.find("W")] = round(size[1])
    matrix = ToCVMatrix(matrix)
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


@pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def reference_pipe(
    data_src, matrix_src, size_src, layout, border_mode, interp_type, inverse_map, fill_value
):
    img = fn.external_source(source=data_src, batch=True, layout=layout)
    matrix = fn.external_source(source=matrix_src)
    size = fn.external_source(source=size_src, batch=True)
    return fn.python_function(
        img,
        matrix,
        size,
        function=lambda im, mx, sz: ref_func(
            im,
            mx,
            sz,
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
    np_rng = np.random.default_rng(seed)

    def gen_matrix():
        dst_pts = np.array([[0, 0], [0, 100], [100, 0], [100, 100]], dtype=np.float32)
        src_offsets = np_rng.random((4, 2), dtype=np.float32) * 20 - 10
        src_pts = dst_pts - src_offsets
        return cv2.getPerspectiveTransform(src_pts, dst_pts).astype(np.float32)

    if constant:
        m = gen_matrix()
        while True:
            yield [m for _ in range(batch_size)]
    else:
        while True:
            yield [gen_matrix() for _ in range(batch_size)]


def size_source(batch_size, seed, constant=False):
    np_rng = np.random.default_rng(seed)

    def gen_size():
        return np_rng.integers(100, 200, size=2).astype(np.float32)

    if constant:
        s = gen_size()
        while True:
            yield [s for _ in range(batch_size)]
    else:
        while True:
            yield [gen_size() for _ in range(batch_size)]


@pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def warp_perspective_pipe_const_matrix(
    data_src,
    layout,
    matrix,
    size_src,
    border_mode,
    interp_type,
    inverse_map,
    fill_value,
    device="gpu",
):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device=device)
    size = fn.external_source(source=size_src, batch=True)
    return fn.experimental.warp_perspective(
        img,
        matrix=matrix,
        size=size,
        border_mode=border_mode,
        interp_type=interp_type,
        inverse_map=inverse_map,
        fill_value=fill_value,
    )


@pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def warp_perspective_pipe_arg_inp_matrix(
    data_src,
    matrix_src,
    size_src,
    layout,
    border_mode,
    interp_type,
    inverse_map,
    fill_value,
    device="gpu",
):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device=device)
    matrix = fn.external_source(source=matrix_src, batch=True)
    if isinstance(size_src, tuple) or size_src is None:
        size = size_src
    else:
        size = fn.external_source(source=size_src, batch=True)

    return fn.experimental.warp_perspective(
        img,
        matrix=matrix,
        size=size,
        border_mode=border_mode,
        interp_type=interp_type,
        inverse_map=inverse_map,
        fill_value=fill_value,
    )


@pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def warp_perspective_pipe_arg1_matrix(
    data_src,
    matrix_src,
    layout,
    size_src,
    border_mode,
    interp_type,
    inverse_map,
    fill_value,
    device="gpu",
):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device=device)
    matrix = fn.external_source(source=matrix_src, batch=True, device=device)
    size = fn.external_source(source=size_src, batch=True)
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
    else:  # int16, uint16
        eps = 5

    test_utils.compare_pipelines(pipe1, pipe2, batch_size=bs, N_iterations=10, eps=eps)


@cartesian_params(
    ("cpu", "gpu"),
    (
        (32, "HWC", np.uint8, 3, "constant", DALIInterpType.INTERP_NN, False, (100, 50, 25)),
        (32, "HWC", np.uint8, 3, "constant", DALIInterpType.INTERP_LINEAR, False, 77),
        (32, "CHW", np.float32, 1, "replicate", DALIInterpType.INTERP_CUBIC, False, None),
        (4, "FCHW", np.uint8, 1, "reflect", DALIInterpType.INTERP_LINEAR, True, None),
        (32, "CHW", np.float32, 4, "constant", DALIInterpType.INTERP_NN, False, 55),
        (8, "FHWC", np.int16, 3, "reflect_101", DALIInterpType.INTERP_LINEAR, True, None),
        (32, "HWC", np.uint16, 4, "constant", DALIInterpType.INTERP_CUBIC, False, None),
    ),
)
def test_warp_perspective_const_matrix_vs_ocv(device, args):
    bs, layout, dtype, channels, border_mode, interp_type, inverse_map, fill_value = args
    data1 = input_iterator(bs, layout, dtype, channels)
    data2 = input_iterator(bs, layout, dtype, channels)

    matrix_src = matrix_source(bs, SEED + 1, True)
    matrix = matrix_src.__next__()[0]
    size_src1 = size_source(bs, SEED + 1, False)
    pipe1 = warp_perspective_pipe_const_matrix(
        data1,
        layout,
        matrix,
        size_src1,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
        device=device,
    )

    size_src2 = size_source(bs, SEED + 1, False)
    pipe2 = reference_pipe(
        data2,
        matrix_src,
        size_src2,
        layout,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )
    compare_pipelines(pipe1, pipe2, bs, dtype)


@cartesian_params(
    ("cpu", "gpu"),
    (
        (32, "HWC", np.uint8, 3, "constant", DALIInterpType.INTERP_NN, False, (100, 50, 25)),
        (32, "HWC", np.uint8, 3, "constant", DALIInterpType.INTERP_LINEAR, False, 77),
        (32, "CHW", np.float32, 1, "replicate", DALIInterpType.INTERP_CUBIC, False, None),
        (4, "FCHW", np.uint8, 1, "reflect", DALIInterpType.INTERP_LINEAR, True, None),
        (32, "CHW", np.float32, 4, "constant", DALIInterpType.INTERP_NN, False, 55),
        (8, "FHWC", np.int16, 3, "reflect_101", DALIInterpType.INTERP_LINEAR, True, None),
        (32, "HWC", np.uint16, 4, "constant", DALIInterpType.INTERP_CUBIC, False, None),
    ),
)
def test_warp_perspective_arg_inp_matrix_vs_ocv(device, args):
    bs, layout, dtype, channels, border_mode, interp_type, inverse_map, fill_value = args
    data1 = input_iterator(bs, layout, dtype, channels)
    data2 = input_iterator(bs, layout, dtype, channels)

    matrix_src1 = matrix_source(bs, SEED + 2, False)
    size_src1 = size_source(bs, SEED + 2, False)
    pipe1 = warp_perspective_pipe_arg_inp_matrix(
        data1,
        matrix_src1,
        size_src1,
        layout,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
        device=device,
    )

    matrix_src2 = matrix_source(bs, SEED + 2, False)
    size_src2 = size_source(bs, SEED + 2, False)
    pipe2 = reference_pipe(
        data2,
        matrix_src2,
        size_src2,
        layout,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )
    compare_pipelines(pipe1, pipe2, bs, dtype)


@cartesian_params(
    ("cpu", "gpu"),
    (
        (32, "HWC", np.uint8, 3, None, "constant", DALIInterpType.INTERP_NN, False, (100, 50, 25)),
        (32, "HWC", np.uint8, 3, (200, 300), "constant", DALIInterpType.INTERP_LINEAR, False, 77),
        (
            32,
            "CHW",
            np.float32,
            1,
            (150, 150),
            "replicate",
            DALIInterpType.INTERP_CUBIC,
            False,
            None,
        ),
        (4, "FCHW", np.uint8, 1, None, "reflect", DALIInterpType.INTERP_LINEAR, True, None),
        (32, "CHW", np.float32, 4, None, "constant", DALIInterpType.INTERP_NN, False, 55),
        (8, "FHWC", np.int16, 3, (20, 30), "reflect_101", DALIInterpType.INTERP_LINEAR, True, None),
        (32, "HWC", np.uint16, 4, None, "constant", DALIInterpType.INTERP_CUBIC, False, None),
    ),
)
def test_warp_perspective_const_size_vs_ocv(device, args):
    bs, layout, dtype, channels, size, border_mode, interp_type, inverse_map, fill_value = args
    data1 = input_iterator(bs, layout, dtype, channels)
    data2 = input_iterator(bs, layout, dtype, channels)

    matrix_src1 = matrix_source(bs, SEED + 2, False)
    pipe1 = warp_perspective_pipe_arg_inp_matrix(
        data1,
        matrix_src1,
        size,
        layout,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
        device=device,
    )

    matrix_src2 = matrix_source(bs, SEED + 2, False)
    if size is not None:

        def size_src():
            return [np.array(size) for _ in range(bs)]

    else:

        def size_src():
            return [np.ones(2) * -1 for _ in range(bs)]

    pipe2 = reference_pipe(
        data2,
        matrix_src2,
        size_src,
        layout,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )
    compare_pipelines(pipe1, pipe2, bs, dtype)


@cartesian_params(
    ("cpu", "gpu"),
    (
        (32, "HWC", np.uint8, 3, "constant", DALIInterpType.INTERP_NN, False, (100, 50, 25)),
        (32, "HWC", np.uint8, 3, "constant", DALIInterpType.INTERP_LINEAR, False, 77),
        (32, "CHW", np.float32, 1, "replicate", DALIInterpType.INTERP_CUBIC, False, None),
        (4, "FCHW", np.uint8, 1, "reflect", DALIInterpType.INTERP_LINEAR, True, None),
        (32, "CHW", np.float32, 4, "constant", DALIInterpType.INTERP_NN, False, 55),
        (8, "FHWC", np.int16, 3, "reflect_101", DALIInterpType.INTERP_LINEAR, True, None),
        (32, "HWC", np.uint16, 4, "constant", DALIInterpType.INTERP_CUBIC, False, None),
    ),
)
def test_warp_perspective_gpu_inp_matrix_vs_ocv(device, args):
    bs, layout, dtype, channels, border_mode, interp_type, inverse_map, fill_value = args
    data1 = input_iterator(bs, layout, dtype, channels)
    data2 = input_iterator(bs, layout, dtype, channels)

    matrix_src1 = matrix_source(bs, SEED + 3, False)
    size_src1 = size_source(bs, SEED + 3, False)
    pipe1 = warp_perspective_pipe_arg1_matrix(
        data1,
        matrix_src1,
        layout,
        size_src1,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
        device=device,
    )

    matrix_src2 = matrix_source(bs, SEED + 3, False)
    size_src2 = size_source(bs, SEED + 3, False)
    pipe2 = reference_pipe(
        data2,
        matrix_src2,
        size_src2,
        layout,
        border_mode,
        interp_type,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )
    compare_pipelines(pipe1, pipe2, bs, dtype)


@pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def warp_affine_pipe(data_src, matrix_src, layout, size, inverse_map, fill_value):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device="gpu")
    matrix = fn.external_source(source=matrix_src, batch=True)[0:2, :]
    return fn.warp_affine(
        img,
        matrix=matrix,
        size=size,
        interp_type=DALIInterpType.INTERP_LINEAR,
        inverse_map=inverse_map,
        fill_value=fill_value,
    )


def affine_matrix_src(bs, seed):
    np_rng = np.random.default_rng(seed)

    def gen_matrix():
        angle = np_rng.random() * 0.5 * np.pi
        rotation = np.array(
            [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        scale = np.diag(
            np.array(
                [np_rng.random() * 0.2 + 0.9, np_rng.random() * 0.2 + 0.9, 1.0], dtype=np.float32
            )
        )
        matrix = np.matmul(rotation, scale)
        matrix[0:2, 2] = np_rng.random((1, 2)) * 50 - 25
        assert np.allclose(matrix[2], [0, 0, 1])
        return matrix

    while True:
        yield [gen_matrix() for _ in range(bs)]


@cartesian_params(
    ("cpu", "gpu"),
    (
        (32, "HWC", np.float32, 3, None, False, 0),
        (32, "HWC", np.uint8, 3, (400, 400), True, 77),
        (4, "FHWC", np.uint8, 1, (200, 300), True, None),
    ),
)
def test_warp_perspective_const_matrix_vs_warp_affine(device, args):
    bs, layout, dtype, channels, size, inverse_map, fill_value = args
    data1 = input_iterator(bs, layout, dtype, channels)
    data2 = input_iterator(bs, layout, dtype, channels)

    matrix_src = affine_matrix_src(bs, SEED)
    if fill_value is None:
        border_mode = "replicate"
    else:
        border_mode = "constant"
    pipe1 = warp_perspective_pipe_arg_inp_matrix(
        data1,
        matrix_src,
        size,
        layout,
        border_mode,
        DALIInterpType.INTERP_LINEAR,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
        device=device,
    )

    matrix_src = affine_matrix_src(bs, SEED)
    pipe2 = warp_affine_pipe(
        data2,
        matrix_src,
        layout,
        size,
        inverse_map,
        fill_value,
        batch_size=bs,
        prefetch_queue_depth=1,
    )
    compare_pipelines(pipe1, pipe2, bs, dtype)


@raises(
    RuntimeError, glob="*Expected a uniform list of 3x3 matrices. Instead got data with shape: *"
)
def test_invalid_shape():
    pipe1 = Pipeline(1, 1, 0)
    with pipe1:
        pipe1.set_outputs(
            fn.experimental.warp_perspective(
                types.Constant(np.ones((10, 10, 3), dtype=np.float32), device="gpu"),
                types.Constant(np.ones((3, 4), dtype=np.float32), device="gpu"),
            )
        )
    pipe1.run()


@raises(
    RuntimeError,
    glob="*Matrix input and `matrix` argument should not be provided at the same time*",
)
def test_clashing_args():
    pipe1 = Pipeline(1, 1, 0)
    with pipe1:
        pipe1.set_outputs(
            fn.experimental.warp_perspective(
                types.Constant(np.ones((10, 10, 3), dtype=np.float32), device="gpu"),
                types.Constant(np.ones((3, 3), dtype=np.float32), device="gpu"),
                matrix=np.ones((3, 3), dtype=np.float32),
            )
        )
    pipe1.run()


@raises(RuntimeError, glob="*Transformation matrix can be provided only as float32 values.*")
def test_invalid_matrix_type():
    pipe1 = Pipeline(1, 1, 0)
    with pipe1:
        pipe1.set_outputs(
            fn.experimental.warp_perspective(
                types.Constant(np.ones((10, 10, 3), dtype=np.float32), device="gpu"),
                types.Constant(np.ones((3, 3), dtype=np.int16), device="gpu"),
            )
        )
    pipe1.run()


@raises(RuntimeError, glob="*Unknown border mode: foo*")
def test_invalid_border_mode():
    pipe1 = Pipeline(1, 1, 0)
    with pipe1:
        pipe1.set_outputs(
            fn.experimental.warp_perspective(
                types.Constant(np.ones((10, 10, 3), dtype=np.float32), device="gpu"),
                matrix=np.ones((3, 3), dtype=np.float32),
                border_mode="foo",
            )
        )
    pipe1.run()
