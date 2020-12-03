# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import nvidia.dali.types as types
import nvidia.dali as dali
import nvidia.dali.fn as fn
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os
import platform
import tempfile
import nose.tools

gds_data_root = '/scratch/'
if not os.path.isdir(gds_data_root):
    gds_data_root = os.getcwd() + "/"

# GDS beta is supported only on x86_64 and compute cap 6.0 >=0
is_gds_supported_var = None
def is_gds_supported(device_id=0):
    global is_gds_supported_var
    if is_gds_supported_var is not None:
        return is_gds_supported_var

    compute_cap = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_cap = compute_cap[0] + compute_cap[1] / 10.
    except ModuleNotFoundError:
        pass

    is_gds_supported_var = platform.processor() == "x86_64" and compute_cap >= 6.0
    return is_gds_supported_var

def NumpyReaderPipeline(path, batch_size, device="cpu", file_list=None, files=None, path_filter="*.npy",
                        num_threads=1, device_id=0, num_gpus=1, cache_header_information=False):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=0)
    data = fn.numpy_reader(device = device,
                           file_list = file_list,
                           files = files,
                           file_root = path,
                           file_filter = path_filter,
                           shard_id = 0,
                           num_shards = 1,
                           cache_header_information = cache_header_information)
    pipe.set_outputs(data)
    return pipe

all_numpy_types = set(
    [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc, np.int_, np.uint,
     np.longlong, np.ulonglong, np.half, np.float16, np.single, np.double, np.longdouble,
     np.csingle, np.cdouble, np.clongdouble, np.int8, np.int16, np.int32, np.int64, np.uint8,
     np.uint16, np.uint32, np.uint64, np.intp, np.uintp, np.float32, np.float64, np.float_,
     np.complex64, np.complex128, np.complex_])
unsupported_numpy_types = set(
    [np.bool_, np.csingle, np.cdouble, np.clongdouble, np.complex64, np.complex128, np.longdouble,
     np.complex_])
test_np_shapes = [(), (11), (4, 7), (6, 2, 5), (1, 2, 7, 4)]
rng = np.random.RandomState(12345)

# test: compare reader with numpy
# with different batch_size and num_threads
def test_types_and_shapes():
    with tempfile.TemporaryDirectory(prefix = gds_data_root) as test_data_root:
        index = 0
        for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
            for fortran_order in [False, True]:
                for type in all_numpy_types - unsupported_numpy_types:
                    for shape in test_np_shapes:
                        filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
                        index += 1
                        yield check_array, filename, shape, type, device, fortran_order

def test_unsupported_types():
    with tempfile.TemporaryDirectory(prefix = gds_data_root) as test_data_root:
        index = 0
        filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
        shape = test_np_shapes[1]
        fortran_order = True
        for type in unsupported_numpy_types:
            nose.tools.assert_raises(RuntimeError, check_array, filename, shape, type,
                                     fortran_order)

def check_batch(test_data_root, batch_size, num_samples, device, arr_np_all, file_list=None, files=None, cache_header_information=False):
    for num_threads in [1, 2, 4, 8]:
        pipe = NumpyReaderPipeline(path = test_data_root,
                                   file_list = file_list,
                                   files=files,
                                   device = device,
                                   path_filter = "test_*.npy",
                                   batch_size = batch_size,
                                   num_threads = num_threads,
                                   device_id = 0,
                                   cache_header_information = False)
        pipe.build()

        for batch in range(0, num_samples, batch_size):
            pipe_out = pipe.run()
            if device == "cpu":
                arr_rd = pipe_out[0].as_array()
            else:
                arr_rd = pipe_out[0].as_cpu().as_array()
            arr_np = arr_np_all[batch:batch+batch_size, ...]

            # compare
            assert_array_equal(arr_rd, arr_np)

# test batch_size > 1
def test_batch():
    with tempfile.TemporaryDirectory(prefix = gds_data_root) as test_data_root:
        # create files
        num_samples = 20
        batch_size = 4
        filenames = []
        arr_np_list = []
        for index in range(0,num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)
            arr_np_list.append(np.load(filename))
        arr_np_all = np.stack(arr_np_list, axis=0)

        for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
            yield check_batch, test_data_root, batch_size, num_samples, device, arr_np_all

def test_batch_file_list():
    with tempfile.TemporaryDirectory(prefix = gds_data_root) as test_data_root:
        # create files
        num_samples = 20
        batch_size = 4
        filenames = []
        arr_np_list = []
        for index in range(0,num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)
            arr_np_list.append(np.load(filename))
        arr_np_all = np.stack(arr_np_list, axis=0)

        # save filenames
        file_list_path = os.path.join(test_data_root, "input.lst")
        with open(file_list_path, "w") as f:
            f.writelines("\n".join(filenames))

        for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
            yield check_batch, "", batch_size, num_samples, device , arr_np_all, file_list_path


def test_batch_files():
    with tempfile.TemporaryDirectory(prefix = gds_data_root) as test_data_root:
        # create files
        num_samples = 20
        batch_size = 4
        filenames = []
        arr_np_list = []
        for index in range(0,num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)
            arr_np_list.append(np.load(filename))
        arr_np_all = np.stack(arr_np_list, axis=0)

        for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
            yield check_batch, None, batch_size, num_samples, device , arr_np_all, None, filenames

def test_batch_files_cache_headers():
    with tempfile.TemporaryDirectory(prefix = gds_data_root) as test_data_root:
        # create files
        num_samples = 20
        batch_size = 4
        filenames = []
        arr_np_list = []
        for index in range(0,num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)
            arr_np_list.append(np.load(filename))
        arr_np_all = np.stack(arr_np_list, axis=0)

        for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
            yield check_batch, None, batch_size, num_samples, device , arr_np_all, None, filenames, True


def test_batch_files_arg():
    with tempfile.TemporaryDirectory(prefix = gds_data_root) as test_data_root:
        # create files
        num_samples = 20
        batch_size = 4
        filenames = []
        arr_np_list = []
        for index in range(0,num_samples):
            rel_name = "test_{:02d}.npy".format(index)
            filename = os.path.join(test_data_root, rel_name)
            filenames.append(rel_name)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)
            arr_np_list.append(np.load(filename))
        arr_np_all = np.stack(arr_np_list, axis=0)

        # create pipe
        for num_threads in [1, 2, 4, 8]:
            pipe = NumpyReaderPipeline(path = test_data_root,
                                       files = filenames,
                                       batch_size = batch_size,
                                       num_threads = num_threads,
                                       device_id = 0)
            pipe.build()

            for batch in range(0, num_samples, batch_size):
                pipe_out = pipe.run()
                arr_rd = pipe_out[0].as_array()
                arr_np = arr_np_all[batch:batch+batch_size, ...]

                # compare
                assert_array_equal(arr_rd, arr_np)

def check_dim_mismatch(device, test_data_root, names):
    pipe = Pipeline(2, 2, 0)
    pipe.set_outputs(fn.numpy_reader(device=device, file_root=test_data_root, files=names))
    pipe.build()
    err = None
    try:
        pipe.run()
    except RuntimeError as thrown:
        err = thrown
    # asserts should not be in except block to avoid printing nested exception on failure
    assert err, "Exception not thrown"
    assert "Inconsistent data" in str(err), "Unexpected error message: {}".format(err)

def test_dim_mismatch():
    with tempfile.TemporaryDirectory(prefix = gds_data_root) as test_data_root:
        names = ["2D.npy", "3D.npy"]
        paths = [os.path.join(test_data_root, name) for name in names]
        create_numpy_file(paths[0], [3,4], np.float32, False)
        create_numpy_file(paths[1], [2,3,4], np.float32, False)

        for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
            yield check_dim_mismatch, device, test_data_root, names

def check_type_mismatch(device, test_data_root, names):
    pipe = Pipeline(2, 2, 0)
    pipe.set_outputs(fn.numpy_reader(device=device, file_root=test_data_root, files=names))
    pipe.build()

    err = None
    try:
        pipe.run()
    except RuntimeError as thrown:
        err = thrown
    # asserts should not be in except block to avoid printing nested exception on failure
    assert err, "Exception not thrown"
    assert "Inconsistent data" in str(err), "Unexpected error message: {}".format(err)
    assert "int32" in str(err) and "float" in str(err), "Unexpected error message: {}".format(err)

def test_type_mismatch():
    with tempfile.TemporaryDirectory(prefix = gds_data_root) as test_data_root:
        names = ["int.npy", "float.npy"]
        paths = [os.path.join(test_data_root, name) for name in names]
        create_numpy_file(paths[0], [1,2,5], np.int32, False)
        create_numpy_file(paths[1], [2,3,4], np.float32, False)

        for device in ["cpu", "gpu"] if is_gds_supported() else ["cpu"]:
            yield check_type_mismatch, device, test_data_root, names

def create_numpy_file(filename, shape, typ, fortran_order):
    # generate random array
    arr = rng.random_sample(shape) * 10.
    arr = arr.astype(typ)
    if fortran_order:
        arr = np.asfortranarray(arr)
    np.save(filename, arr)

def delete_numpy_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)

def check_array(filename, shape, typ, device, fortran_order=False):
    # setup file
    create_numpy_file(filename, shape, typ, fortran_order)

    # load manually
    arr_np = np.load(filename)

    for num_threads in [1, 2, 4, 8]:
        # load with numpy reader
        pipe = NumpyReaderPipeline(path = os.path.dirname(filename),
                                   device = device,
                                   batch_size = 1,
                                   num_threads = num_threads,
                                   device_id = 0)
        pipe.build()
        pipe_out = pipe.run()
        if device == "cpu":
            arr_rd = np.squeeze(pipe_out[0].as_array(), axis=0)
        else:
            arr_rd = np.squeeze(pipe_out[0].as_cpu().as_array(), axis=0)

        # compare
        assert_array_equal(arr_rd, arr_np)

    # delete temp files
    delete_numpy_file(filename)
