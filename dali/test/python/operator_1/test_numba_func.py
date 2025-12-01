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
import os
from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import do_not_convert
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as dali_types
from nose_utils import with_setup, attr
from test_utils import (
    get_dali_extra_path,
    to_array,
    check_numba_compatibility_cpu,
    check_numba_compatibility_gpu,
)
from nvidia.dali.plugin.numba.fn.experimental import numba_function
from numba import cuda

test_data_root = get_dali_extra_path()
lmdb_folder = os.path.join(test_data_root, "db", "lmdb")


def set_all_values_to_1_batch(out0, in0):
    out0[0][:] = 1


def set_all_values_to_255_batch(out0, in0):
    out0[0][:] = 255


def set_all_values_to_255_sample(out0, in0):
    out0[:] = 255


def set_all_values_to_1_sample_gpu(out0, in0):
    tx, ty, tz = cuda.grid(3)
    x_s, y_s, z_s = cuda.gridsize(3)

    out0[tz::z_s, ty::y_s, tx::x_s] = 1


def set_all_values_to_255_sample_gpu(out0, in0):
    tx, ty, tz = cuda.grid(3)
    x_s, y_s, z_s = cuda.gridsize(3)

    out0[tz::z_s, ty::y_s, tx::x_s] = 255


def set_all_values_to_float_batch(out0, in0):
    out0[0][:] = 0.5


def set_all_values_to_float_sample(out0, in0):
    out0[:] = 0.5


def set_all_values_to_float_sample_gpu(out0, in0):
    tx, ty, tz = cuda.grid(3)
    x_s, y_s, z_s = cuda.gridsize(3)

    out0[tz::z_s, ty::y_s, tx::x_s] = 0.5


def setup_change_out_shape(out_shape, in_shape):
    out0_shape = out_shape[0]
    in0_shape = in_shape[0]
    perm = [1, 2, 0]
    for sample_idx in range(len(out0_shape)):
        for d in range(len(perm)):
            out0_shape[sample_idx][d] = in0_shape[sample_idx][perm[d]]


def change_out_shape_batch(out0, in0):
    for sample_id in range(len(out0)):
        out0[sample_id][:] = 42


def change_out_shape_sample(out0, in0):
    out0[:] = 42


def change_out_shape_sample_gpu(out0, in0):
    tx, ty, tz = cuda.grid(3)
    x_s, y_s, z_s = cuda.gridsize(3)

    out0[tz::z_s, ty::y_s, tx::x_s] = 42


# in shape [x] -> out shape [2, 2, 2, x]
def change_ndim_setup(outs_shape, ins_shapes):
    out_shape = outs_shape[0]
    in_shape = ins_shapes[0]
    for sample_id in range(len(out_shape)):
        out_shape[sample_id][:] = [2, 2, 2, in_shape[sample_id][0]]


def change_ndim_gpu(out0, in0):
    tx, ty = cuda.grid(2)
    x_s, y_s = cuda.gridsize(2)
    tid = ty * x_s + tx
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for x in range(tid, out0.shape[3], x_s * y_s):
                    out0[i][j][k][x] = x


def change_dim_expected_out(d):
    return np.array(list(range(d)) * 8).reshape(2, 2, 2, d)


def get_data(shapes, dtype):
    return [np.empty(shape, dtype=dtype) for shape in shapes]


def get_data_zeros(shapes, dtype):
    return [np.zeros(shape, dtype=dtype) for shape in shapes]


@attr("sanitizer_skip")
def _testimpl_numba_func(
    device,
    shapes,
    dtype,
    run_fn,
    out_types,
    in_types,
    outs_ndim,
    ins_ndim,
    setup_fn,
    batch_processing,
    expected_out,
    blocks=None,
    threads_per_block=None,
    enable_conditionals=False,
):
    @pipeline_def(enable_conditionals=enable_conditionals)
    def numba_func_pipe(
        shapes,
        dtype,
        device="cpu",
        run_fn=None,
        out_types=None,
        in_types=None,
        outs_ndim=None,
        ins_ndim=None,
        setup_fn=None,
        batch_processing=None,
        blocks=None,
        threads_per_block=None,
    ):
        data = fn.external_source(lambda: get_data(shapes, dtype), batch=True, device=device)
        return numba_function(
            data,
            run_fn=run_fn,
            out_types=out_types,
            in_types=in_types,
            outs_ndim=outs_ndim,
            ins_ndim=ins_ndim,
            setup_fn=setup_fn,
            batch_processing=batch_processing,
            device=device,
            blocks=blocks,
            threads_per_block=threads_per_block,
        )

    batch_size = len(shapes)
    pipe = numba_func_pipe(
        batch_size=batch_size,
        num_threads=1,
        device_id=0,
        shapes=shapes,
        dtype=dtype,
        device=device,
        run_fn=run_fn,
        setup_fn=setup_fn,
        out_types=out_types,
        in_types=in_types,
        outs_ndim=outs_ndim,
        ins_ndim=ins_ndim,
        batch_processing=batch_processing,
        blocks=blocks,
        threads_per_block=threads_per_block,
    )
    for it in range(3):
        outs = pipe.run()
        for i in range(batch_size):
            out_arr = to_array(outs[0][i])
            assert np.array_equal(out_arr, expected_out[i])


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_cpu)
def test_numba_func():
    # shape, dtype, run_fn, out_types,
    # in_types, out_ndim, in_ndim, setup_fn, batch_processing,
    # expected_out
    args = [
        (
            [(10, 10, 10)],
            np.bool_,
            set_all_values_to_1_batch,
            [dali_types.BOOL],
            [dali_types.BOOL],
            [3],
            [3],
            None,
            True,
            [np.full((10, 10, 10), 1, dtype=np.bool_)],
        ),
        (
            [(10, 10, 10)],
            np.uint8,
            set_all_values_to_255_batch,
            [dali_types.UINT8],
            [dali_types.UINT8],
            [3],
            [3],
            None,
            True,
            [np.full((10, 10, 10), 255, dtype=np.uint8)],
        ),
        (
            [(10, 10, 10)],
            np.uint8,
            set_all_values_to_255_sample,
            [dali_types.UINT8],
            [dali_types.UINT8],
            [3],
            [3],
            None,
            None,
            [np.full((10, 10, 10), 255, dtype=np.uint8)],
        ),
        (
            [(10, 10, 10)],
            np.float32,
            set_all_values_to_float_batch,
            [dali_types.FLOAT],
            [dali_types.FLOAT],
            [3],
            [3],
            None,
            True,
            [np.full((10, 10, 10), 0.5, dtype=np.float32)],
        ),
        (
            [(10, 10, 10)],
            np.float32,
            set_all_values_to_float_sample,
            [dali_types.FLOAT],
            [dali_types.FLOAT],
            [3],
            [3],
            None,
            None,
            [np.full((10, 10, 10), 0.5, dtype=np.float32)],
        ),
        (
            [(10, 20, 30), (20, 10, 30)],
            np.int64,
            change_out_shape_batch,
            [dali_types.INT64],
            [dali_types.INT64],
            [3],
            [3],
            setup_change_out_shape,
            True,
            [np.full((20, 30, 10), 42, dtype=np.int32), np.full((10, 30, 20), 42, dtype=np.int32)],
        ),
        (
            [(10, 20, 30), (20, 10, 30)],
            np.int64,
            change_out_shape_sample,
            [dali_types.INT64],
            [dali_types.INT64],
            [3],
            [3],
            setup_change_out_shape,
            None,
            [np.full((20, 30, 10), 42, dtype=np.int32), np.full((10, 30, 20), 42, dtype=np.int32)],
        ),
    ]

    device = "cpu"
    for (
        shape,
        dtype,
        run_fn,
        out_types,
        in_types,
        outs_ndim,
        ins_ndim,
        setup_fn,
        batch_processing,
        expected_out,
    ) in args:
        yield (
            _testimpl_numba_func,
            device,
            shape,
            dtype,
            run_fn,
            out_types,
            in_types,
            outs_ndim,
            ins_ndim,
            setup_fn,
            batch_processing,
            expected_out,
        )


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_cpu)
def test_numba_func_with_cond():
    # When the function is not converted, the numba still works with no issues.
    # AG conversion or using a complex enough decorator would break this.
    # TODO(klecki): Can we add any additional safeguards?
    _testimpl_numba_func(
        device="cpu",
        shapes=[(10, 10, 10)],
        dtype=np.uint8,
        run_fn=set_all_values_to_255_batch,
        out_types=[dali_types.UINT8],
        in_types=[dali_types.UINT8],
        outs_ndim=[3],
        ins_ndim=[3],
        setup_fn=None,
        batch_processing=True,
        expected_out=[np.full((10, 10, 10), 255, dtype=np.uint8)],
        enable_conditionals=True,
    )


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_cpu)
def test_numba_func_with_cond_do_not_convert():
    # Test if do_not_convert decorated functions still work.
    _testimpl_numba_func(
        device="cpu",
        shapes=[(10, 10, 10)],
        dtype=np.uint8,
        run_fn=do_not_convert(set_all_values_to_255_batch),
        out_types=[dali_types.UINT8],
        in_types=[dali_types.UINT8],
        outs_ndim=[3],
        ins_ndim=[3],
        setup_fn=None,
        batch_processing=True,
        expected_out=[np.full((10, 10, 10), 255, dtype=np.uint8)],
        enable_conditionals=True,
    )


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_gpu)
def test_numba_func_gpu():
    # shape, dtype, run_fn, out_types,
    # in_types, out_ndim, in_ndim, setup_fn, batch_processing,
    # expected_out
    args = [
        (
            [(10, 10, 10)],
            np.bool_,
            set_all_values_to_1_sample_gpu,
            [dali_types.BOOL],
            [dali_types.BOOL],
            [3],
            [3],
            None,
            None,
            [np.full((10, 10, 10), 1, dtype=np.bool_)],
        ),
        (
            [(10, 10, 10)],
            np.uint8,
            set_all_values_to_255_sample_gpu,
            [dali_types.UINT8],
            [dali_types.UINT8],
            [3],
            [3],
            None,
            None,
            [np.full((10, 10, 10), 255, dtype=np.uint8)],
        ),
        (
            [(10, 10, 10)],
            np.float32,
            set_all_values_to_float_sample_gpu,
            [dali_types.FLOAT],
            [dali_types.FLOAT],
            [3],
            [3],
            None,
            None,
            [np.full((10, 10, 10), 0.5, dtype=np.float32)],
        ),
        (
            [(100, 20, 30), (20, 100, 30)],
            np.int64,
            change_out_shape_sample_gpu,
            [dali_types.INT64],
            [dali_types.INT64],
            [3],
            [3],
            setup_change_out_shape,
            None,
            [
                np.full((20, 30, 100), 42, dtype=np.int32),
                np.full((100, 30, 20), 42, dtype=np.int32),
            ],
        ),
        (
            [(20), (30)],
            np.int32,
            change_ndim_gpu,
            [dali_types.INT32],
            [dali_types.INT32],
            [4],
            [1],
            change_ndim_setup,
            None,
            [change_dim_expected_out(20), change_dim_expected_out(30)],
        ),
    ]

    device = "gpu"
    blocks = [32, 32, 1]
    threads_per_block = [32, 16, 1]
    for (
        shape,
        dtype,
        run_fn,
        out_types,
        in_types,
        outs_ndim,
        ins_ndim,
        setup_fn,
        batch_processing,
        expected_out,
    ) in args:
        yield (
            _testimpl_numba_func,
            device,
            shape,
            dtype,
            run_fn,
            out_types,
            in_types,
            outs_ndim,
            ins_ndim,
            setup_fn,
            batch_processing,
            expected_out,
            blocks,
            threads_per_block,
        )


@pipeline_def
def numba_func_image_pipe(
    device="cpu",
    run_fn=None,
    out_types=None,
    in_types=None,
    outs_ndim=None,
    ins_ndim=None,
    setup_fn=None,
    batch_processing=None,
    blocks=None,
    threads_per_block=None,
):
    files, _ = dali.fn.readers.caffe(path=lmdb_folder, random_shuffle=True)
    dec_device = "cpu" if device == "cpu" else "mixed"
    images_in = dali.fn.decoders.image(files, device=dec_device)
    images_out = numba_function(
        images_in,
        run_fn=run_fn,
        out_types=out_types,
        in_types=in_types,
        outs_ndim=outs_ndim,
        ins_ndim=ins_ndim,
        setup_fn=setup_fn,
        batch_processing=batch_processing,
        device=device,
        blocks=blocks,
        threads_per_block=threads_per_block,
    )
    return images_in, images_out


@attr("sanitizer_skip")
def _testimpl_numba_func_image(
    device,
    run_fn,
    out_types,
    in_types,
    outs_ndim,
    ins_ndim,
    setup_fn,
    batch_processing,
    transform,
    blocks=None,
    threads_per_block=None,
):
    pipe = numba_func_image_pipe(
        device=device,
        batch_size=8,
        num_threads=3,
        device_id=0,
        run_fn=run_fn,
        setup_fn=setup_fn,
        out_types=out_types,
        in_types=in_types,
        outs_ndim=outs_ndim,
        ins_ndim=ins_ndim,
        batch_processing=batch_processing,
        blocks=blocks,
        threads_per_block=threads_per_block,
    )
    for _ in range(3):
        images_in, images_out = pipe.run()
        for i in range(len(images_in)):
            image_in_transformed = transform(to_array(images_in[i]))
            assert np.array_equal(image_in_transformed, to_array(images_out[i]))


def reverse_col_batch(out0, in0):
    for sample_id in range(len(out0)):
        out0[sample_id][:] = 255 - in0[sample_id][:]


def reverse_col_sample(out0, in0):
    out0[:] = 255 - in0[:]


def reverse_col_sample_gpu(out0, in0):
    tx, ty, tz = cuda.grid(3)
    x_s, y_s, z_s = cuda.gridsize(3)

    for z in range(tz, out0.shape[0], z_s):
        for y in range(ty, out0.shape[1], y_s):
            for x in range(tx, out0.shape[2], x_s):
                out0[z][y][x] = 255 - in0[z][y][x]


def rot_image_batch(out0, in0):
    for out_sample, in_sample in zip(out0, in0):
        for i in range(out_sample.shape[0]):
            for j in range(out_sample.shape[1]):
                out_sample[i][j] = in_sample[j][out_sample.shape[0] - i - 1]


def rot_image_sample(out0, in0):
    for i in range(out0.shape[0]):
        for j in range(out0.shape[1]):
            out0[i][j] = in0[j][out0.shape[0] - i - 1]


def rot_image_sample_gpu(out0, in0):
    tx, ty, tz = cuda.grid(3)
    x_s, y_s, z_s = cuda.gridsize(3)

    for z in range(tz, out0.shape[0], z_s):
        for y in range(ty, out0.shape[1], y_s):
            for x in range(tx, out0.shape[2], x_s):
                out0[z][y][x] = in0[y][out0.shape[0] - z - 1][x]


def rot_image_setup(outs, ins):
    out0 = outs[0]
    in0 = ins[0]
    for sample_id in range(len(out0)):
        out0[sample_id][0] = in0[sample_id][1]
        out0[sample_id][1] = in0[sample_id][0]
        out0[sample_id][2] = in0[sample_id][2]


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_cpu)
def test_numba_func_image():
    args = [
        (
            reverse_col_batch,
            [dali_types.UINT8],
            [dali_types.UINT8],
            [3],
            [3],
            None,
            True,
            lambda x: 255 - x,
        ),
        (
            reverse_col_sample,
            [dali_types.UINT8],
            [dali_types.UINT8],
            [3],
            [3],
            None,
            None,
            lambda x: 255 - x,
        ),
        (
            rot_image_batch,
            [dali_types.UINT8],
            [dali_types.UINT8],
            [3],
            [3],
            rot_image_setup,
            True,
            lambda x: np.rot90(x),
        ),
        (
            rot_image_sample,
            [dali_types.UINT8],
            [dali_types.UINT8],
            [3],
            [3],
            rot_image_setup,
            None,
            lambda x: np.rot90(x),
        ),
    ]
    device = "cpu"
    for (
        run_fn,
        out_types,
        in_types,
        outs_ndim,
        ins_ndim,
        setup_fn,
        batch_processing,
        transform,
    ) in args:
        yield (
            _testimpl_numba_func_image,
            device,
            run_fn,
            out_types,
            in_types,
            outs_ndim,
            ins_ndim,
            setup_fn,
            batch_processing,
            transform,
        )


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_gpu)
def test_numba_func_image_gpu():
    args = [
        (
            reverse_col_sample_gpu,
            [dali_types.UINT8],
            [dali_types.UINT8],
            [3],
            [3],
            None,
            None,
            lambda x: 255 - x,
        ),
        (
            rot_image_sample_gpu,
            [dali_types.UINT8],
            [dali_types.UINT8],
            [3],
            [3],
            rot_image_setup,
            None,
            np.rot90,
        ),
    ]
    device = "gpu"
    blocks = [32, 32, 1]
    threads_per_block = [32, 8, 1]
    for (
        run_fn,
        out_types,
        in_types,
        outs_ndim,
        ins_ndim,
        setup_fn,
        batch_processing,
        transform,
    ) in args:
        yield (
            _testimpl_numba_func_image,
            device,
            run_fn,
            out_types,
            in_types,
            outs_ndim,
            ins_ndim,
            setup_fn,
            batch_processing,
            transform,
            blocks,
            threads_per_block,
        )


def split_images_col_sample(out0, out1, out2, in0):
    for i in range(in0.shape[0]):
        for j in range(in0.shape[1]):
            out0[i][j] = in0[i][j][0]
            out1[i][j] = in0[i][j][1]
            out2[i][j] = in0[i][j][2]


def split_images_col_sample_gpu(out0, out1, out2, in0):
    tx, ty = cuda.grid(2)
    x_s, y_s = cuda.gridsize(2)

    for y in range(ty, out0.shape[0], y_s):
        for x in range(tx, out0.shape[1], x_s):
            out0[y][x] = in0[y][x][0]
            out1[y][x] = in0[y][x][1]
            out2[y][x] = in0[y][x][2]


def setup_split_images_col(outs, ins):
    out0 = outs[0]
    out1 = outs[1]
    out2 = outs[2]
    for sample_id in range(len(out0)):
        out0[sample_id][0] = ins[0][sample_id][0]
        out0[sample_id][1] = ins[0][sample_id][1]
        out1[sample_id][0] = ins[0][sample_id][0]
        out1[sample_id][1] = ins[0][sample_id][1]
        out2[sample_id][0] = ins[0][sample_id][0]
        out2[sample_id][1] = ins[0][sample_id][1]


@pipeline_def
def numba_func_split_image_pipe(
    run_fn=None,
    out_types=None,
    in_types=None,
    outs_ndim=None,
    ins_ndim=None,
    setup_fn=None,
    batch_processing=None,
    device="cpu",
    blocks=None,
    threads_per_block=None,
):
    files, _ = dali.fn.readers.caffe(path=lmdb_folder)
    dec_device = "cpu" if device == "cpu" else "mixed"
    images_in = dali.fn.decoders.image(files, device=dec_device)
    out0, out1, out2 = numba_function(
        images_in,
        run_fn=run_fn,
        out_types=out_types,
        in_types=in_types,
        outs_ndim=outs_ndim,
        ins_ndim=ins_ndim,
        setup_fn=setup_fn,
        batch_processing=batch_processing,
        device=device,
        blocks=blocks,
        threads_per_block=threads_per_block,
    )
    return images_in, out0, out1, out2


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_cpu)
def test_split_images_col():
    pipe = numba_func_split_image_pipe(
        batch_size=8,
        num_threads=1,
        device_id=0,
        run_fn=split_images_col_sample,
        setup_fn=setup_split_images_col,
        out_types=[dali_types.UINT8 for i in range(3)],
        in_types=[dali_types.UINT8],
        outs_ndim=[2, 2, 2],
        ins_ndim=[3],
        device="cpu",
    )
    for _ in range(3):
        images_in, R, G, B = pipe.run()
        for i in range(len(images_in)):
            assert np.array_equal(images_in.at(i), np.stack([R.at(i), G.at(i), B.at(i)], axis=2))


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_gpu)
def test_split_images_col_gpu():
    blocks = [32, 32, 1]
    threads_per_block = [32, 8, 1]
    pipe = numba_func_split_image_pipe(
        batch_size=8,
        num_threads=1,
        device_id=0,
        run_fn=split_images_col_sample_gpu,
        setup_fn=setup_split_images_col,
        out_types=[dali_types.UINT8 for i in range(3)],
        in_types=[dali_types.UINT8],
        outs_ndim=[2, 2, 2],
        ins_ndim=[3],
        device="gpu",
        blocks=blocks,
        threads_per_block=threads_per_block,
    )
    for _ in range(3):
        images_in, R, G, B = pipe.run()
        for i in range(len(images_in)):
            assert np.array_equal(
                to_array(images_in[i]),
                np.stack([to_array(R[i]), to_array(G[i]), to_array(B[i])], axis=2),
            )


def multiple_ins_setup(outs, ins):
    out0 = outs[0]
    in0 = ins[0]
    for sample_id in range(len(out0)):
        out0[sample_id][0] = in0[sample_id][0]
        out0[sample_id][1] = in0[sample_id][1]
        out0[sample_id][2] = 3


def multiple_ins_run(out0, in0, in1, in2):
    for i in range(out0.shape[0]):
        for j in range(out0.shape[1]):
            out0[i][j][0] = in0[i][j]
            out0[i][j][1] = in1[i][j]
            out0[i][j][2] = in2[i][j]


def multiple_ins_run_gpu(out0, in0, in1, in2):
    tx, ty = cuda.grid(2)
    x_s, y_s = cuda.gridsize(2)

    for y in range(ty, out0.shape[0], y_s):
        for x in range(tx, out0.shape[1], x_s):
            out0[y][x][0] = in0[y][x]
            out0[y][x][1] = in1[y][x]
            out0[y][x][2] = in2[y][x]


@attr("sanitizer_skip")
@pipeline_def
def numba_multiple_ins_pipe(
    shapes,
    dtype,
    run_fn=None,
    out_types=None,
    in_types=None,
    outs_ndim=None,
    ins_ndim=None,
    setup_fn=None,
    batch_processing=None,
    device="cpu",
    blocks=None,
    threads_per_block=None,
):
    data0 = fn.external_source(lambda: get_data_zeros(shapes, dtype), batch=True, device=device)
    data1 = fn.external_source(lambda: get_data_zeros(shapes, dtype), batch=True, device=device)
    data2 = fn.external_source(lambda: get_data_zeros(shapes, dtype), batch=True, device=device)
    return numba_function(
        data0,
        data1,
        data2,
        run_fn=run_fn,
        out_types=out_types,
        in_types=in_types,
        outs_ndim=outs_ndim,
        ins_ndim=ins_ndim,
        setup_fn=setup_fn,
        batch_processing=batch_processing,
        device=device,
        blocks=blocks,
        threads_per_block=threads_per_block,
    )


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_cpu)
def test_multiple_ins():
    pipe = numba_multiple_ins_pipe(
        shapes=[(10, 10)],
        dtype=np.uint8,
        batch_size=8,
        num_threads=1,
        device_id=0,
        run_fn=multiple_ins_run,
        setup_fn=multiple_ins_setup,
        out_types=[dali_types.UINT8],
        in_types=[dali_types.UINT8 for i in range(3)],
        outs_ndim=[3],
        ins_ndim=[2, 2, 2],
        device="cpu",
    )
    for _ in range(3):
        outs = pipe.run()
        out_arr = np.array(outs[0][0])
        assert np.array_equal(out_arr, np.zeros((10, 10, 3), dtype=np.uint8))


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_gpu)
def test_multiple_ins_gpu():
    blocks = [32, 32, 1]
    threads_per_block = [32, 8, 1]
    pipe = numba_multiple_ins_pipe(
        shapes=[(10, 10)],
        dtype=np.uint8,
        batch_size=8,
        num_threads=1,
        device_id=0,
        run_fn=multiple_ins_run_gpu,
        setup_fn=multiple_ins_setup,
        out_types=[dali_types.UINT8],
        in_types=[dali_types.UINT8 for i in range(3)],
        outs_ndim=[3],
        ins_ndim=[2, 2, 2],
        device="gpu",
        blocks=blocks,
        threads_per_block=threads_per_block,
    )
    for _ in range(3):
        outs = pipe.run()
        out_arr = to_array(outs[0][0])
        assert np.array_equal(out_arr, np.zeros((10, 10, 3), dtype=np.uint8))


def nonuniform_types_setup(outs, ins):
    out0 = outs[0]
    out1 = outs[1]
    in0 = ins[0]
    for sample_id in range(len(out0)):
        out0[sample_id][0] = in0[sample_id][0]
        out0[sample_id][1] = in0[sample_id][1]
        out0[sample_id][2] = in0[sample_id][2]
        out1[sample_id][0] = 3


def nonuniform_types_run_cpu(out_img, out_shape, in_img):
    out_img[:] = 255 - in_img[:]
    out_shape[:] = out_img.shape


def nonuniform_types_run_gpu(out0, out_shape, in0):
    tx, ty, tz = cuda.grid(3)
    x_s, y_s, z_s = cuda.gridsize(3)

    if tx + ty + tz == 0:
        out_shape[:] = out0.shape

    for z in range(tz, out0.shape[0], z_s):
        for y in range(ty, out0.shape[1], y_s):
            for x in range(tx, out0.shape[2], x_s):
                out0[z][y][x] = 255 - in0[z][y][x]


@pipeline_def
def nonuniform_types_pipe(
    run_fn=None,
    out_types=None,
    in_types=None,
    outs_ndim=None,
    ins_ndim=None,
    setup_fn=nonuniform_types_setup,
    batch_processing=False,
    device="cpu",
    blocks=None,
    threads_per_block=None,
):
    files, _ = dali.fn.readers.caffe(path=lmdb_folder)
    dec_device = "cpu" if device == "cpu" else "mixed"
    images_in = dali.fn.decoders.image(files, device=dec_device)
    out_img, out_shape = numba_function(
        images_in,
        run_fn=run_fn,
        out_types=out_types,
        in_types=in_types,
        outs_ndim=outs_ndim,
        ins_ndim=ins_ndim,
        setup_fn=setup_fn,
        batch_processing=batch_processing,
        device=device,
        blocks=blocks,
        threads_per_block=threads_per_block,
    )
    return images_in, out_img, out_shape


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_cpu)
def test_nonuniform_types_cpu():
    pipe = nonuniform_types_pipe(
        batch_size=8,
        num_threads=1,
        device_id=0,
        run_fn=nonuniform_types_run_cpu,
        out_types=[dali_types.UINT8, dali_types.INT64],
        in_types=[dali_types.UINT8],
        outs_ndim=[3, 1],
        ins_ndim=[3],
        device="cpu",
    )
    for _ in range(3):
        images_in, images_out, img_shape = pipe.run()
        for i in range(len(images_in)):
            assert np.array_equal(255 - images_in.at(i), images_out.at(i))
            assert np.array_equal(images_out.at(i).shape, img_shape.at(i))


@attr("sanitizer_skip")
@with_setup(check_numba_compatibility_gpu)
def test_nonuniform_types_gpu():
    blocks = [16, 16, 1]
    threads_per_block = [32, 16, 1]
    pipe = nonuniform_types_pipe(
        batch_size=8,
        num_threads=1,
        device_id=0,
        run_fn=nonuniform_types_run_gpu,
        out_types=[dali_types.UINT8, dali_types.INT64],
        in_types=[dali_types.UINT8],
        outs_ndim=[3, 1],
        ins_ndim=[3],
        device="gpu",
        blocks=blocks,
        threads_per_block=threads_per_block,
    )
    for _ in range(3):
        images_in, images_out, img_shape = pipe.run()
        images_in, images_out, img_shape = (
            images_in.as_cpu(),
            images_out.as_cpu(),
            img_shape.as_cpu(),
        )
        for i in range(len(images_in)):
            assert np.array_equal(255 - images_in.at(i), images_out.at(i))
            assert np.array_equal(images_out.at(i).shape, img_shape.at(i))
