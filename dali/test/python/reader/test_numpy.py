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

from nvidia.dali import Pipeline, pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.backend as dali_b
import numpy as np
from numpy.testing import assert_array_equal
import os
import platform
import random
import tempfile
from nose_utils import assert_raises, SkipTest
from nose2.tools import params, cartesian_params
from test_utils import compare_pipelines, to_array


gds_data_root = "/scratch/"
if not os.path.isdir(gds_data_root):
    gds_data_root = os.getcwd() + "/scratch/"
    if not os.path.isdir(gds_data_root):
        os.mkdir(gds_data_root)
        assert os.path.isdir(gds_data_root)


# GDS beta is supported only on x86_64 and compute cap 6.0 >=0
is_gds_supported_var = None


def is_gds_supported(device_id=0):
    global is_gds_supported_var
    if is_gds_supported_var is not None:
        return is_gds_supported_var

    compute_cap = 0
    cuda_drv_ver = 0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_cap = compute_cap[0] + compute_cap[1] / 10.0
        cuda_drv_ver = pynvml.nvmlSystemGetCudaDriverVersion()
    except ModuleNotFoundError:
        print("Python bindings for NVML not found")

    # for CUDA < 12.2 only x86 platform is supported, above aarch64 is supported as well
    is_gds_supported_var = (
        platform.processor() == "x86_64"
        or (dali_b.__cuda_version__ >= 12200 and cuda_drv_ver >= 12020)
    ) and compute_cap >= 6.0
    return is_gds_supported_var


def create_numpy_file(filename, shape, typ, fortran_order):
    # generate random array
    arr = rng.random_sample(shape) * 10.0
    arr = arr.astype(typ)
    if fortran_order:
        arr = np.asfortranarray(arr)
    np.save(filename, arr)


def delete_numpy_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)


def NumpyReaderPipeline(
    path,
    batch_size,
    device="cpu",
    file_list=None,
    files=None,
    file_filter="*.npy",
    num_threads=1,
    device_id=0,
    cache_header_information=False,
    pad_last_batch=False,
    dont_use_mmap=False,
    enable_o_direct=False,
    shard_id=0,
    num_shards=1,
):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    data = fn.readers.numpy(
        device=device,
        file_list=file_list,
        files=files,
        file_root=path,
        file_filter=file_filter,
        shard_id=shard_id,
        num_shards=num_shards,
        cache_header_information=cache_header_information,
        pad_last_batch=pad_last_batch,
        dont_use_mmap=dont_use_mmap,
        use_o_direct=enable_o_direct,
    )
    pipe.set_outputs(data)
    return pipe


all_numpy_types = set(
    [
        np.bool_,
        np.byte,
        np.ubyte,
        np.short,
        np.ushort,
        np.intc,
        np.uintc,
        np.int_,
        np.uint,
        np.longlong,
        np.ulonglong,
        np.half,
        np.float16,
        np.single,
        np.double,
        np.longdouble,
        np.csingle,
        np.cdouble,
        np.clongdouble,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.intp,
        np.uintp,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        complex,
    ]
)
unsupported_numpy_types = set(
    [
        np.csingle,
        np.cdouble,
        np.clongdouble,
        np.complex64,
        np.complex128,
        np.longdouble,
        complex,
    ]
)
rng = np.random.RandomState(12345)

# Test shapes, for each number of dims
test_shapes = {
    0: [(), (), (), (), (), (), (), ()],
    1: [(10,), (12,), (10,), (20,), (10,), (12,), (13,), (19,)],
    2: [(10, 10), (12, 10), (10, 12), (20, 15), (10, 11), (12, 11), (13, 11), (19, 10)],
    3: [(6, 2, 5), (5, 6, 2), (3, 3, 3), (10, 1, 8), (8, 8, 3), (2, 2, 3), (8, 4, 3), (1, 10, 1)],
    4: [
        (2, 6, 2, 5),
        (5, 1, 6, 2),
        (3, 2, 3, 3),
        (1, 10, 1, 8),
        (2, 8, 2, 3),
        (2, 3, 2, 3),
        (1, 8, 4, 3),
        (1, 3, 10, 1),
    ],
}


def _testimpl_types_and_shapes(
    device,
    shapes,
    type,
    batch_size,
    num_threads,
    fortran_order_arg,
    file_arg_type,
    cache_header_information,
    dont_use_mmap=False,
    enable_o_direct=False,
):
    """compare reader with numpy, with different batch_size and num_threads"""
    nsamples = len(shapes)

    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        # setup file
        filenames = ["test_{:02d}.npy".format(i) for i in range(nsamples)]
        full_paths = [os.path.join(test_data_root, fname) for fname in filenames]
        for i in range(nsamples):
            fortran_order = fortran_order_arg
            if fortran_order is None:
                fortran_order = random.choice([False, True])
            create_numpy_file(full_paths[i], shapes[i], type, fortran_order)

        # load manually
        arrays = [np.load(filename) for filename in full_paths]

        # load with numpy reader
        file_list_arg = None
        files_arg = None
        file_filter_arg = None
        if file_arg_type == "file_list":
            file_list_arg = os.path.join(test_data_root, "input.lst")
            with open(file_list_arg, "w") as f:
                f.writelines("\n".join(filenames))
        elif file_arg_type == "files":
            files_arg = filenames
        elif file_arg_type == "file_filter":
            file_filter_arg = "*.npy"
        else:
            assert False

        pipe = NumpyReaderPipeline(
            path=test_data_root,
            files=files_arg,
            file_list=file_list_arg,
            file_filter=file_filter_arg,
            cache_header_information=cache_header_information,
            device=device,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=0,
            dont_use_mmap=dont_use_mmap,
            enable_o_direct=enable_o_direct,
        )
        try:
            i = 0
            while i < nsamples:
                pipe_out = pipe.run()
                for s in range(batch_size):
                    if i == nsamples:
                        break
                    pipe_arr = to_array(pipe_out[0][s])
                    ref_arr = arrays[i]
                    assert_array_equal(pipe_arr, ref_arr)
                    i += 1
        finally:
            del pipe


def _get_type_and_shape_params():
    rng = np.random.default_rng(1902)
    for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
        for fortran_order in [False, True, None]:
            for dtype in all_numpy_types - unsupported_numpy_types:
                for ndim in [0, 1, 2, rng.choice([3, 4])]:
                    if ndim <= 1 and fortran_order is not False:
                        continue
                    shapes = test_shapes[ndim]
                    file_arg_type = rng.choice(["file_list", "files", "file_filter"])
                    num_threads = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])
                    batch_size = rng.choice([1, 3, 4, 8, 16])
                    yield (
                        device,
                        fortran_order,
                        dtype,
                        shapes,
                        file_arg_type,
                        num_threads,
                        batch_size,
                    )


@params(
    (False,),
    (True,),
)
def test_header_parse(use_o_direct):
    # Test different ndims to see how well we handle headers of different lengths and padding.
    # The NPY token (header meta-data) + header is padded to the size aligned up to 64 bytes.
    # In particular the `np.full((1,) * 21, 1., dtype=float32)` and
    # `np.full((1,) * 22, 1., dtype=float32)` are the boundary between 128 and 192 bytes header.
    # This make a good case for testing the bounds we use for extracting the header.
    # The 32 is the max dimensionality handled by the numpy
    ndims = list(range(33))
    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        names = [f"numpy_ndim_{ndim}.npy" for ndim in ndims]
        paths = [os.path.join(test_data_root, name) for name in names]
        assert len(paths) == len(ndims)
        for ndim, path in zip(ndims, paths):
            np.save(path, np.full((1,) * ndim, 1.0, dtype=np.float32))

        reader_kwargs = {} if not use_o_direct else {"use_o_direct": True, "dont_use_mmap": True}

        @pipeline_def(batch_size=1, device_id=0, num_threads=4)
        def pipeline(test_filename):
            arr = fn.readers.numpy(files=[test_filename], **reader_kwargs)
            return arr

        for ndim, path in zip(ndims, paths):
            p = pipeline(test_filename=path)
            (out,) = p.run()
            shapes = out.shape()
            assert len(shapes) == 1, f"{len(shapes)}"
            shape = shapes[0]
            assert shape == (1,) * ndim, f"{ndim} {shape}"


@params(*list(_get_type_and_shape_params()))
def test_types_and_shapes(
    device, fortran_order, dtype, shapes, file_arg_type, num_threads, batch_size
):
    cache_header_information = False
    _testimpl_types_and_shapes(
        device,
        shapes,
        dtype,
        batch_size,
        num_threads,
        fortran_order,
        file_arg_type,
        cache_header_information,
    )


@cartesian_params(
    (0, 1, 2, random.choice([3, 4])),
    (True, False),
    (random.choice(["file_list", "files", "file_filter"]),),
    (random.choice([1, 2, 3, 4, 5, 6, 7, 8]),),
    (random.choice([1, 3, 4, 8, 16]),),
    (random.choice(list(all_numpy_types - unsupported_numpy_types)),),
)
def test_o_direct(
    ndim,
    o_direct,
    file_arg_type,
    num_threads,
    batch_size,
    type,
):
    cache_header_information = False
    device = "cpu"
    fortran_order = False
    shapes = test_shapes[ndim]
    _testimpl_types_and_shapes(
        device,
        shapes,
        type,
        batch_size,
        num_threads,
        fortran_order,
        file_arg_type,
        cache_header_information,
        True,
        o_direct,
    )


def _get_unsupported_param():
    for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
        for dtype in unsupported_numpy_types:
            yield device, dtype


@params(*list(_get_unsupported_param()))
def test_unsupported_types(device, dtype):
    fortran_order = False
    cache_header_information = False
    file_arg_type = "files"
    ndim = 1
    shapes = test_shapes[ndim]
    num_threads = 3
    batch_size = 3
    with assert_raises(RuntimeError, glob="Unknown Numpy type string"):
        _testimpl_types_and_shapes(
            device,
            shapes,
            dtype,
            batch_size,
            num_threads,
            fortran_order,
            file_arg_type,
            cache_header_information,
        )


@params(*(["cpu", "gpu"] if is_gds_supported() else ["cpu"]))
def test_cache_headers(device):
    type = np.float32
    ndim = 2
    shapes = test_shapes[ndim]
    num_threads = 3
    batch_size = 3
    cache_header_information = True
    fortran_order = False
    file_arg_type = "files"
    _testimpl_types_and_shapes(
        device,
        shapes,
        type,
        batch_size,
        num_threads,
        fortran_order,
        file_arg_type,
        cache_header_information,
    )


def check_dim_mismatch(device, test_data_root, names):
    pipe = Pipeline(2, 2, 0)
    pipe.set_outputs(fn.readers.numpy(device=device, file_root=test_data_root, files=names))
    err = None
    try:
        pipe.run()
    except RuntimeError as thrown:
        err = thrown
    finally:
        del pipe
    # asserts should not be in except block to avoid printing nested exception on failure
    assert err, "Exception not thrown"
    assert "Inconsistent data" in str(err), "Unexpected error message: {}".format(err)


@params(*(["cpu", "gpu"] if is_gds_supported() else ["cpu"]))
def test_dim_mismatch(device):
    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        names = ["2D.npy", "3D.npy"]
        paths = [os.path.join(test_data_root, name) for name in names]
        create_numpy_file(paths[0], [3, 4], np.float32, False)
        create_numpy_file(paths[1], [2, 3, 4], np.float32, False)
        check_dim_mismatch(device, test_data_root, names)


def check_type_mismatch(device, test_data_root, names):
    err = None
    pipe = Pipeline(2, 2, 0)
    pipe.set_outputs(fn.readers.numpy(device=device, file_root=test_data_root, files=names))

    try:
        pipe.run()
    except RuntimeError as thrown:
        err = thrown
    finally:
        del pipe
    # asserts should not be in except block to avoid printing nested exception on failure
    assert err, "Exception not thrown"
    assert "Inconsistent data" in str(err), "Unexpected error message: {}".format(err)
    assert "int32" in str(err) and "float" in str(err), "Unexpected error message: {}".format(err)


@params(*(["cpu", "gpu"] if is_gds_supported() else ["cpu"]))
def test_type_mismatch(device):
    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        names = ["int.npy", "float.npy"]
        paths = [os.path.join(test_data_root, name) for name in names]
        create_numpy_file(paths[0], [1, 2, 5], np.int32, False)
        create_numpy_file(paths[1], [2, 3, 4], np.float32, False)
        check_type_mismatch(device, test_data_root, names)


batch_size_alias_test = 64


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def numpy_reader_pipe(numpy_op, path, device="cpu", file_filter="*.npy"):
    data = numpy_op(device=device, file_root=path, file_filter=file_filter, seed=1234)
    return data


def check_numpy_reader_alias(test_data_root, device):
    new_pipe = numpy_reader_pipe(
        fn.readers.numpy, path=test_data_root, device=device, file_filter="test_*.npy"
    )
    legacy_pipe = numpy_reader_pipe(
        fn.numpy_reader, path=test_data_root, device=device, file_filter="test_*.npy"
    )
    try:
        compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 50)
    finally:
        del new_pipe
        del legacy_pipe


@params(*(["cpu", "gpu"] if is_gds_supported() else ["cpu"]))
def test_numpy_reader_alias(device):
    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        # create files
        num_samples = 20
        filenames = []
        arr_np_list = []
        for index in range(0, num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)
            arr_np_list.append(np.load(filename))

        check_numpy_reader_alias(test_data_root, device)


@pipeline_def(device_id=0, num_threads=8)
def numpy_reader_roi_pipe(
    file_root,
    device="cpu",
    file_filter="*.npy",
    roi_start=None,
    rel_roi_start=None,
    roi_end=None,
    rel_roi_end=None,
    roi_shape=None,
    rel_roi_shape=None,
    roi_axes=None,
    default_axes=[],
    out_of_bounds_policy=None,
    fill_value=None,
):
    data = fn.readers.numpy(
        device=device,
        file_root=file_root,
        file_filter=file_filter,
        shard_id=0,
        num_shards=1,
        cache_header_information=False,
    )
    roi_data = fn.readers.numpy(
        device=device,
        file_root=file_root,
        file_filter=file_filter,
        roi_start=roi_start,
        rel_roi_start=rel_roi_start,
        roi_end=roi_end,
        rel_roi_end=rel_roi_end,
        roi_shape=roi_shape,
        rel_roi_shape=rel_roi_shape,
        roi_axes=roi_axes,
        out_of_bounds_policy=out_of_bounds_policy,
        fill_value=fill_value,
        shard_id=0,
        num_shards=1,
        cache_header_information=False,
    )
    sliced_data = fn.slice(
        data,
        start=roi_start,
        rel_start=rel_roi_start,
        end=roi_end,
        rel_end=rel_roi_end,
        shape=roi_shape,
        rel_shape=rel_roi_shape,
        axes=roi_axes or default_axes,  # Slice has different default (axis_names="WH")
        out_of_bounds_policy=out_of_bounds_policy,
        fill_values=fill_value,
    )
    return roi_data, sliced_data


def _testimpl_numpy_reader_roi(
    file_root,
    batch_size,
    ndim,
    dtype,
    device,
    fortran_order=False,
    file_filter="*.npy",
    roi_start=None,
    rel_roi_start=None,
    roi_end=None,
    rel_roi_end=None,
    roi_shape=None,
    rel_roi_shape=None,
    roi_axes=None,
    out_of_bounds_policy=None,
    fill_value=None,
):
    default_axes = list(range(ndim))
    pipe = numpy_reader_roi_pipe(
        file_root=file_root,
        file_filter=file_filter,
        device=device,
        roi_start=roi_start,
        rel_roi_start=rel_roi_start,
        roi_end=roi_end,
        rel_roi_end=rel_roi_end,
        roi_shape=roi_shape,
        rel_roi_shape=rel_roi_shape,
        roi_axes=roi_axes,
        default_axes=default_axes,
        out_of_bounds_policy=out_of_bounds_policy,
        fill_value=fill_value,
        batch_size=batch_size,
    )

    try:
        roi_out, sliced_out = pipe.run()
        for i in range(batch_size):
            roi_arr = to_array(roi_out[i])
            sliced_arr = to_array(sliced_out[i])
            assert_array_equal(roi_arr, sliced_arr)
    finally:
        del pipe


def _testimpl_numpy_reader_roi_empty_axes(
    testcase_name, file_root, batch_size, ndim, dtype, device, fortran_order, file_filter="*.npy"
):
    # testcase name used for visibility in the output logs
    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=8)
    def pipe():
        data0 = fn.readers.numpy(
            device=device,
            file_root=file_root,
            file_filter=file_filter,
            shard_id=0,
            num_shards=1,
            cache_header_information=False,
            seed=1234,
        )
        data1 = fn.readers.numpy(
            device=device,
            file_root=file_root,
            file_filter=file_filter,
            roi_start=[],
            roi_end=[],
            roi_axes=[],
            shard_id=0,
            num_shards=1,
            cache_header_information=False,
            seed=1234,
        )
        return data0, data1

    p = pipe()
    try:
        data0, data1 = p.run()
    finally:
        del p
    for i in range(batch_size):
        arr = to_array(data0[i])
        roi_arr = to_array(data1[i])
        assert_array_equal(arr, roi_arr)


def _testimpl_numpy_reader_roi_empty_range(
    testcase_name, file_root, batch_size, ndim, dtype, device, fortran_order, file_filter="*.npy"
):
    # testcase name used for visibility in the output logs
    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=8)
    def pipe():
        data0 = fn.readers.numpy(
            device=device,
            file_root=file_root,
            file_filter=file_filter,
            shard_id=0,
            num_shards=1,
            cache_header_information=False,
            seed=1234,
        )
        data1 = fn.readers.numpy(
            device=device,
            file_root=file_root,
            file_filter=file_filter,
            roi_start=[1],
            roi_end=[1],
            roi_axes=[1],
            shard_id=0,
            num_shards=1,
            cache_header_information=False,
            seed=1234,
        )
        return data0, data1

    p = pipe()
    try:
        data0, data1 = p.run()
        for i in range(batch_size):
            arr = to_array(data0[i])
            roi_arr = to_array(data1[i])
            for d in range(len(arr.shape)):
                if d == 1:
                    assert roi_arr.shape[d] == 0
                else:
                    assert roi_arr.shape[d] == arr.shape[d]
    finally:
        del p


# roi_start, rel_roi_start, roi_end, rel_roi_end, roi_shape,
# rel_roi_shape, roi_axes, out_of_bounds_policy
roi_args = [
    ([1, 2], None, None, None, None, None, None, None),
    (None, [0.1, 0.2], None, None, None, None, None, None),
    (None, None, [8, 7], None, None, None, None, None),
    (None, None, None, [0.5, 0.9], None, None, None, None),
    (None, None, None, None, [4, 5], None, None, None),
    (None, None, None, None, None, [0.4, 0.8], None, None),
    (1, None, 9, None, None, None, [0], None),
    (1, None, 9, None, None, None, [1], None),
    ([1, 2], None, [8, 9], None, None, None, [0, 1], None),
    ([1, 2], None, [8, 9], None, None, None, [0, 1], None),
    ([1, 2], None, None, [0.5, 0.4], None, None, [0, 1], None),
    (None, [0.1, 0.2], [8, 9], None, None, None, [0, 1], None),
    ([1, 2], None, [20, 9], None, None, None, [0, 1], "pad"),
    ([-10, 2], None, [8, 9], None, None, None, [0, 1], "pad"),
    ([1, 2], None, [20, 9], None, None, None, [0, 1], "trim_to_shape"),
    ([-10, 2], None, [8, 9], None, None, None, [0, 1], "trim_to_shape"),
    (
        fn.random.uniform(range=(0, 2), shape=(2,), dtype=types.INT32),
        None,
        fn.random.uniform(range=(7, 10), shape=(2,), dtype=types.INT32),
        None,
        None,
        None,
        (0, 1),
        None,
    ),
    (
        fn.random.uniform(range=(0, 2), shape=(1,), dtype=types.INT32),
        None,
        fn.random.uniform(range=(7, 10), shape=(1,), dtype=types.INT32),
        None,
        None,
        None,
        (1,),
        None,
    ),
    (
        None,
        fn.random.uniform(range=(0.0, 0.2), shape=(1,)),
        None,
        fn.random.uniform(range=(0.8, 1.0), shape=(1,)),
        None,
        None,
        (1,),
        None,
    ),
]


def _get_roi_suite_params():
    i = 0
    rng = np.random.default_rng(1902)
    for roi_params in roi_args:
        for fortran_order in [False, True, None]:
            for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
                fill_value = rng.choice([None, 10.0])
                yield (i,) + roi_params + (fortran_order, device, fill_value)
                i += 1


@params(*list(_get_roi_suite_params()))
def test_numpy_reader_roi(
    i,
    roi_start,
    rel_roi_start,
    roi_end,
    rel_roi_end,
    roi_shape,
    rel_roi_shape,
    roi_axes,
    out_of_bounds_policy,
    fortran_order,
    device,
    fill_value,
):
    # setup file
    shapes = [(10, 10), (12, 10), (10, 12), (20, 15), (10, 11), (12, 11), (13, 11), (19, 10)]
    ndim = 2
    dtype = np.uint8
    batch_size = 8
    file_filter = "*.npy"
    rng = np.random.default_rng(4242 + i)

    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        index = 0
        for sh in shapes:
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            index += 1
            if fortran_order is not None:
                actual_fortran_order = fortran_order
            else:
                actual_fortran_order = rng.choice([False, True])
            create_numpy_file(filename, sh, dtype, actual_fortran_order)

        _testimpl_numpy_reader_roi(
            test_data_root,
            batch_size,
            ndim,
            dtype,
            device,
            fortran_order,
            file_filter,
            roi_start,
            rel_roi_start,
            roi_end,
            rel_roi_end,
            roi_shape,
            rel_roi_shape,
            roi_axes,
            out_of_bounds_policy,
            fill_value,
        )


def _get_roi_empty_axes_params():
    i = 0
    for fortran_order in [False, True, None]:
        for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
            for axes_or_range in ["axes", "range"]:
                yield i, fortran_order, device, axes_or_range
                i += 1


@params(*list(_get_roi_empty_axes_params()))
def test_numpy_reader_roi_empty_axes(i, fortran_order, device, axes_or_range):
    # setup file
    shapes = [(10, 10), (12, 10), (10, 12), (20, 15), (10, 11), (12, 11), (13, 11), (19, 10)]
    ndim = 2
    dtype = np.uint8
    batch_size = 8
    file_filter = "*.npy"
    rng = np.random.default_rng(4242 + i)

    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        index = 0
        for sh in shapes:
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            index += 1
            if fortran_order is not None:
                actual_fortran_order = fortran_order
            else:
                actual_fortran_order = rng.choice([False, True])
            create_numpy_file(filename, sh, dtype, actual_fortran_order)

        if axes_or_range == "axes":
            _testimpl_numpy_reader_roi_empty_axes(
                "empty axes",
                test_data_root,
                batch_size,
                ndim,
                dtype,
                device,
                fortran_order,
                file_filter,
            )
        else:
            assert axes_or_range == "range"
            _testimpl_numpy_reader_roi_empty_range(
                "empty range",
                test_data_root,
                batch_size,
                ndim,
                dtype,
                device,
                fortran_order,
                file_filter,
            )


def _testimpl_numpy_reader_roi_error(
    file_root,
    batch_size,
    ndim,
    dtype,
    device,
    fortran_order=False,
    file_filter="*.npy",
    roi_start=None,
    rel_roi_start=None,
    roi_end=None,
    rel_roi_end=None,
    roi_shape=None,
    rel_roi_shape=None,
    roi_axes=None,
    out_of_bounds_policy=None,
    fill_value=None,
):
    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=8)
    def pipe():
        data = fn.readers.numpy(
            device=device,
            file_root=file_root,
            file_filter=file_filter,
            roi_start=roi_start,
            rel_roi_start=rel_roi_start,
            roi_end=roi_end,
            rel_roi_end=rel_roi_end,
            roi_shape=roi_shape,
            rel_roi_shape=rel_roi_shape,
            roi_axes=roi_axes,
            out_of_bounds_policy=out_of_bounds_policy,
            fill_value=fill_value,
            shard_id=0,
            num_shards=1,
            cache_header_information=False,
        )
        return data

    p = pipe()
    err = None
    try:
        p.run()
    except RuntimeError as thrown:
        err = thrown
    # asserts should not be in except block to avoid printing nested exception on failure
    assert err, "Exception not thrown"


def _get_roi_error_params():
    # roi_start, rel_roi_start, roi_end, rel_roi_end, roi_shape, rel_roi_shape,
    # roi_axes, out_of_bounds_policy
    roi_args = [
        # Both roi_start and rel_roi_start
        ([1, 2], [0.1, 0.2], None, None, None, None, None, None),
        (None, None, [8, 7], [0.4, 0.5], None, None, None, None),  # Both roi_end and rel_roi_end
        (None, None, [8, 7], None, [8, 7], None, None, None),  # Both roi_end and roi_shape
        (None, None, [8, 7], None, None, [0.4, 0.5], None, None),  # Both roi_end and rel_roi_shape
        (None, None, None, [0.5, 0.4], [8, 7], None, None, None),  # Both rel_roi_end and roi_shape
        ([-1, 2], None, None, None, None, None, None, None),  # Out of bounds anchor
        (None, None, [100, 8], None, None, None, None, None),  # Out of bounds end
        (None, None, None, None, [100, 8], None, None, None),  # Out of bounds shape
    ]

    for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
        for roi_params in roi_args:
            fill_value = rng.choice([None, 10.0])
            yield (device,) + roi_params + (fill_value,)


@params(*list(_get_roi_error_params()))
def test_numpy_reader_roi_error(
    device,
    roi_start,
    rel_roi_start,
    roi_end,
    rel_roi_end,
    roi_shape,
    rel_roi_shape,
    roi_axes,
    out_of_bounds_policy,
    fill_value,
):
    # setup file
    shapes = [(10, 10), (12, 10), (10, 12), (20, 15), (10, 11), (12, 11), (13, 11), (19, 10)]
    ndim = 2
    dtype = np.uint8
    batch_size = 8
    file_filter = "*.npy"
    fortran_order = False

    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        index = 0
        for sh in shapes:
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            index += 1
            create_numpy_file(filename, sh, dtype, fortran_order=fortran_order)

        _testimpl_numpy_reader_roi_error(
            test_data_root,
            batch_size,
            ndim,
            dtype,
            device,
            fortran_order,
            file_filter,
            roi_start,
            rel_roi_start,
            roi_end,
            rel_roi_end,
            roi_shape,
            rel_roi_shape,
            roi_axes,
            out_of_bounds_policy,
            fill_value,
        )


@cartesian_params(("cpu", "gpu"), ((1, 2, 1), (3, 1, 2)), (True, False), (True, False))
def test_pad_last_sample(device, batch_description, dont_use_mmap, use_o_direct):
    if not is_gds_supported() and device == "gpu":
        raise SkipTest("GDS is not supported in this platform")
    if not dont_use_mmap and use_o_direct:
        raise SkipTest("Cannot use O_DIRECT with mmap")
    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        # create files
        num_samples, batch_size, num_shards = batch_description
        filenames = []
        ref_filenames = []
        arr_np_list = []
        last_file_name = None
        for index in range(0, num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            last_file_name = filename
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)
            arr_np_list.append(np.load(filename))
            ref_filenames.append(filename)
        while len(arr_np_list) < batch_size:
            arr_np_list.append(np.load(last_file_name))
            ref_filenames.append(last_file_name)
        pipe = NumpyReaderPipeline(
            path=test_data_root,
            files=filenames,
            file_list=None,
            file_filter=None,
            device=device,
            batch_size=batch_size,
            num_threads=4,
            device_id=0,
            pad_last_batch=True,
            num_shards=num_shards,
            dont_use_mmap=dont_use_mmap,
            enable_o_direct=use_o_direct,
        )

        try:
            for _ in range(2):
                pipe_out = pipe.run()
                for i in range(batch_size):
                    out_arr = to_array(pipe_out[0][i])
                    out_prop = pipe_out[0][i].source_info()
                    ref_arr = arr_np_list[i]
                    assert out_prop == ref_filenames[i]
                    assert_array_equal(out_arr, ref_arr)
        finally:
            del pipe


@cartesian_params(("global", "local", "none"), (True, False))
def test_shuffling(shuffling, pad_last_batch):
    if not is_gds_supported():
        raise SkipTest("GDS is not supported in this platform")

    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        # create files
        num_samples = 10
        batch_size = 3
        filenames = []
        for index in range(0, num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (3, 2, 1), np.int8, False)
        random_shuffle = False
        shuffle_after_epoch = False
        stick_to_shard = False
        if shuffling == "global":
            shuffle_after_epoch = True
        elif shuffle_after_epoch == "local":
            random_shuffle = True
            stick_to_shard = True

        pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
        with pipe:
            data_cpu = fn.readers.numpy(
                device="cpu",
                files=filenames,
                file_root=test_data_root,
                shard_id=0,
                num_shards=2,
                pad_last_batch=pad_last_batch,
                random_shuffle=random_shuffle,
                shuffle_after_epoch=shuffle_after_epoch,
                stick_to_shard=stick_to_shard,
            )
            data_gpu = fn.readers.numpy(
                device="gpu",
                files=filenames,
                file_root=test_data_root,
                shard_id=0,
                num_shards=2,
                pad_last_batch=pad_last_batch,
                random_shuffle=random_shuffle,
                shuffle_after_epoch=shuffle_after_epoch,
                stick_to_shard=stick_to_shard,
            )
            pipe.set_outputs(data_cpu, data_gpu)

        for _ in range(num_samples // batch_size * 2):
            (cpu_arr, gpu_arr) = pipe.run()
            assert_array_equal(to_array(cpu_arr), to_array(gpu_arr))
