# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.types as dali_types
from nvidia.dali.backend_impl import TensorListCPU
from nvidia.dali import plugin_manager

import functools
import inspect
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from packaging.version import Version
from nose_utils import SkipTest


is_of_supported_var = None


def get_arch(device_id=0):
    compute_cap = 0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_cap = compute_cap[0] + compute_cap[1] / 10.0
    except ModuleNotFoundError:
        print("NVML not found")
    return compute_cap


def is_mulit_gpu():
    try:
        import pynvml

        pynvml.nvmlInit()
        is_mulit_gpu_var = pynvml.nvmlDeviceGetCount() != 1
    except ModuleNotFoundError:
        print("Python bindings for NVML not found")

    return is_mulit_gpu_var


def get_device_memory_info(device_id=0):
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetMemoryInfo(handle)
    except ModuleNotFoundError:
        print("Python bindings for NVML not found")
        return None
    except pynvml.NVMLError_NotSupported:
        print("nvmlDeviceGetMemoryInfo not supported on this system")
        return None


def get_gpu_name_from_nvml():
    try:
        nvml_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        )
        return nvml_output.decode("utf-8").strip().split("\n")
    except subprocess.CalledProcessError:
        return None


def skip_if_m60():
    """Skip the test if the GPU is M60. The video decoder is not supported in full on M60."""

    gpus = get_gpu_name_from_nvml()
    if "Tesla M60" in gpus:
        raise SkipTest()


def get_dali_extra_path():
    try:
        dali_extra_path = os.environ["DALI_EXTRA_PATH"]
    except KeyError:
        print("WARNING: DALI_EXTRA_PATH not initialized.", file=sys.stderr)
        dali_extra_path = "."
    return dali_extra_path


# those functions import modules on demand to no impose additional dependency on numpy or matplot
# to test that are using these utilities
np = None
assert_array_equal = None
assert_allclose = None
cp = None
absdiff_checked = False


def import_numpy():
    global np
    global assert_array_equal
    global assert_allclose
    import numpy as np
    from numpy.testing import assert_array_equal, assert_allclose


def import_cupy():
    global cp
    import cupy as cp


Image = None


def import_pil():
    global Image
    from PIL import Image


def save_image(image, file_name):
    import_numpy()
    import_pil()
    if image.dtype == np.float32:
        min = np.min(image)
        max = np.max(image)
        if min >= 0 and max <= 1:
            image = image * 256
        elif min >= -1 and max <= 1:
            image = (image + 1) * 128
        elif min >= -128 and max <= 127:
            image = image + 128
    else:
        lo = np.iinfo(image.dtype).min
        hi = np.iinfo(image.dtype).max
        image = (image - lo) * (255.0 / (hi - lo))
    image = image.astype(np.uint8)
    Image.fromarray(image).save(file_name)


def get_gpu_num():
    sp = subprocess.Popen(
        ["nvidia-smi", "-L"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    out_str = sp.communicate()
    out_list = out_str[0].split("\n")
    out_list = [elm for elm in out_list if len(elm) > 0]
    return len(out_list)


def _get_absdiff(left, right):
    def make_unsigned(dtype):
        if not np.issubdtype(dtype, np.signedinteger):
            return dtype
        return {
            np.dtype(np.int8): np.uint8,
            np.dtype(np.int16): np.uint16,
            np.dtype(np.int32): np.uint32,
            np.dtype(np.int64): np.uint64,
        }[dtype]

    # np.abs of diff doesn't handle overflow for unsigned types
    absdiff = np.maximum(left, right) - np.minimum(left, right)
    # max - min can overflow for signed types, wrap them up
    absdiff = absdiff.astype(make_unsigned(absdiff.dtype))
    return absdiff


def _check_absdiff():
    """
    In principle, overflow on signed int is UB (that we relied on so far anyway).
    The following one-time check aims to verify the overflow wraps as expected.
    """
    for i in range(-128, 127):
        for j in range(-128, 127):
            left = np.array([i, i], dtype=np.int8)
            right = np.array([j, j], dtype=np.int8)
            diff = _get_absdiff(left, right)
            expected_diff = np.array([abs(i - j), abs(i - j)], dtype=np.uint8)
            assert np.array_equal(diff, expected_diff), f"{diff} {expected_diff} {i} {j}"
    for i in range(0, 255):
        for j in range(0, 255):
            left = np.array([i, i], dtype=np.uint8)
            right = np.array([j, j], dtype=np.uint8)
            diff = _get_absdiff(left, right)
            expected_diff = np.array([abs(i - j), abs(i - j)], dtype=np.uint8)
            assert np.array_equal(diff, expected_diff), f"{diff} {expected_diff} {i} {j}"


def get_absdiff(left, right):
    # Make sanity checks, in particular, if wrapping signed integers works as expected
    global absdiff_checked
    if not absdiff_checked:
        absdiff_checked = True
        _check_absdiff()
    return _get_absdiff(left, right)


def dump_as_core_artifacts(image_info, lhs, rhs, iter=None, sample_idx=None):
    import_numpy()
    import_pil()

    from pathlib import Path

    path = (
        "/opt/dali"
        if os.path.exists("/opt/dali") and os.access("/opt/dali", os.W_OK)
        else os.getcwd()
    )
    Path(f"{path}/core_artifacts").mkdir(parents=True, exist_ok=True)

    image_info = image_info.replace("/", "_")
    image_info = image_info.replace(" ", "_")
    if iter is not None:
        image_info = image_info + f"_iter{iter}"
    if sample_idx is not None:
        image_info = image_info + f"_sample_idx{sample_idx}"

    try:
        save_image(lhs, f"{path}/core_artifacts/{image_info}.lhs.png")
        save_image(rhs, f"{path}/core_artifacts/{image_info}.rhs.png")
    except Exception as e:
        print(f"Tried to save images but got an error: {e}")

    try:
        # save arrays on artifact folder
        import numpy as np

        np.save(f"{path}/core_artifacts/{image_info}.lhs.npy", lhs)
        np.save(f"{path}/core_artifacts/{image_info}.rhs.npy", rhs)
    except Exception as e:
        print(f"Tried to save arrays but got an error: {e}")


# If the `max_allowed_error` is not None, it's checked instead of comparing mean error with `eps`.
def check_batch(
    batch1,
    batch2,
    batch_size=None,
    eps=1e-07,
    max_allowed_error=None,
    expected_layout=None,
    compare_layouts=True,
):
    """Compare two batches of data, be it dali TensorList or list of numpy arrays.

    Args:
        batch1: input batch
        batch2: input batch
        batch_size: reference batch size - if None, only equality is enforced
        eps (float, optional): Used for mean error validation. Defaults to 1e-07.
        max_allowed_error (int or float, optional): If provided the max diff between elements.
        expected_layout (str, optional): If provided, the batches that are DALI types
            will be checked to match this layout. If None, there will be no check
        compare_layouts (bool, optional): Whether to compare layouts between two batches.
            Checked only if both inputs are DALI types. Defaults to True.
    """

    def is_error(mean_err, max_err, eps, max_allowed_error):
        if max_allowed_error is not None:
            if max_err > max_allowed_error:
                return True
        elif mean_err > eps:
            return True
        return False

    import_numpy()
    if isinstance(batch1, dali.backend.TensorListGPU):
        batch1 = batch1.as_cpu()
    if isinstance(batch2, dali.backend.TensorListGPU):
        batch2 = batch2.as_cpu()

    if batch_size is None:
        batch_size = len(batch1)

    def _verify_batch_size(batch):
        if isinstance(batch, dali.backend.TensorListCPU) or isinstance(batch, list):
            tested_batch_size = len(batch)
        else:
            tested_batch_size = batch.shape[0]
        assert (
            tested_batch_size == batch_size
        ), "Incorrect batch size. Expected: {}, actual: {}".format(batch_size, tested_batch_size)

    _verify_batch_size(batch1)
    _verify_batch_size(batch2)

    # Check layouts where possible
    for batch in [batch1, batch2]:
        if expected_layout is not None and isinstance(batch, dali.backend.TensorListCPU):
            assert (
                batch.layout() == expected_layout
            ), 'Unexpected layout, expected "{}", got "{}".'.format(expected_layout, batch.layout())

    if (
        compare_layouts
        and isinstance(batch1, dali.backend.TensorListCPU)
        and isinstance(batch2, dali.backend.TensorListCPU)
    ):
        assert batch1.layout() == batch2.layout(), 'Layout mismatch "{}" != "{}"'.format(
            batch1.layout(), batch2.layout()
        )

    for i in range(batch_size):
        # This allows to handle list of Tensors, list of np arrays and TensorLists
        left = np.array(batch1[i])
        right = np.array(batch2[i])
        err_err = None
        assert left.shape == right.shape, "Shape mismatch {} != {}".format(left.shape, right.shape)
        assert left.size == right.size, "Size mismatch {} != {}".format(left.size, right.size)
        if left.size != 0:
            try:
                # Do the difference calculation on a type that allows subtraction
                if left.dtype == bool:
                    left = left.astype(int)
                if right.dtype == bool:
                    right = right.astype(int)
                absdiff = get_absdiff(left, right)
                err = np.mean(absdiff)
                max_err = np.max(absdiff)
                min_err = np.min(absdiff)
                total_errors = np.sum(absdiff != 0)
            except Exception as e:
                err_err = str(e)
            if err_err or is_error(err, max_err, eps, max_allowed_error):
                if err_err:
                    error_msg = f"Error calculation failed:\n{err_err}!\n"
                else:
                    error_msg = (
                        f"Mean error: [{err}], Min error: [{min_err}], "
                        f"Max error: [{max_err}]\n"
                        f"Total error count: [{total_errors}], "
                        f"Tensor size: [{absdiff.size}]\n"
                        f"Index in batch: {i}\n"
                    )
                if hasattr(batch1[i], "source_info"):
                    error_msg += f"\nLHS data source: {batch1[i].source_info()}"
                if hasattr(batch2[i], "source_info"):
                    error_msg += f"\nRHS data source: {batch2[i].source_info()}"

                filename = (
                    batch1[i].source_info() if hasattr(batch1[i], "source_info") else f"unknown{i}"
                )
                dump_as_core_artifacts(filename, left, right, sample_idx=i)
                assert False, error_msg


def compare_pipelines(
    pipe1,
    pipe2,
    batch_size,
    N_iterations,
    eps=1e-07,
    max_allowed_error=None,
    expected_layout=None,
    compare_layouts=True,
):
    """Compare the outputs of two pipelines across several iterations.

    Args:
        pipe1: input pipeline object.
        pipe2: input pipeline object.
        batch_size (int): batch size
        N_iterations (int): Number of iterations used for comparison
        eps (float, optional): Allowed mean error between samples. Defaults to 1e-07.
        max_allowed_error (int or float, optional): If provided the max diff between elements.
        expected_layout (str or tuple of str, optional): If provided the outputs of both pipelines
            will be matched with provided layouts and error will be raised if there is mismatch.
            Defaults to None.
        compare_layouts (bool, optional): Whether to compare layouts of outputs between pipelines.
            Defaults to True.
    """
    for _ in range(N_iterations):
        out1 = tuple(out.as_cpu() for out in pipe1.run())
        out2 = tuple(out.as_cpu() for out in pipe2.run())
        assert len(out1) == len(out2)
        for i, (out1_data, out2_data) in enumerate(zip(out1, out2)):
            if isinstance(expected_layout, tuple):
                current_expected_layout = expected_layout[i]
            else:
                current_expected_layout = expected_layout
            check_batch(
                out1_data,
                out2_data,
                batch_size,
                eps,
                max_allowed_error,
                expected_layout=current_expected_layout,
                compare_layouts=compare_layouts,
            )


class RandomDataIterator(object):
    def __init__(self, batch_size, shape=(10, 600, 800, 3), dtype=None, seed=0):
        import_numpy()
        # to avoid any numpy reference in the interface
        if dtype is None:
            dtype = np.uint8
        self.batch_size = batch_size
        self.test_data = []
        self.np_rng = np.random.default_rng(seed=seed)
        for _ in range(self.batch_size):
            if dtype == np.float32:
                self.test_data.append(
                    np.array(self.np_rng.random(shape) * (1.0), dtype=dtype) - 0.5
                )
            else:
                self.test_data.append(np.array(self.np_rng.random(shape) * 255, dtype=dtype))

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        batch = self.test_data
        self.i = (self.i + 1) % self.n
        return batch

    next = __next__


class RandomlyShapedDataIterator(object):
    def __init__(
        self,
        batch_size,
        min_shape=None,
        max_shape=(10, 600, 800, 3),
        seed=12345,
        dtype=None,
        val_range=None,
    ):
        import_numpy()
        # to avoid any numpy reference in the interface
        if dtype is None:
            dtype = np.uint8
        self.batch_size = batch_size
        self.test_data = []
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.dtype = dtype
        self.seed = seed
        self.np_rng = np.random.default_rng(seed=seed)
        self.rng = random.Random(seed)
        self.val_range = val_range

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        import_numpy()
        self.test_data = []
        for _ in range(self.batch_size):
            # Scale between 0.5 and 1.0
            if self.min_shape is None:
                shape = [
                    int(self.max_shape[dim] * (0.5 + self.rng.random() * 0.5))
                    for dim in range(len(self.max_shape))
                ]
            else:
                shape = [
                    self.rng.randint(min_s, max_s)
                    for min_s, max_s in zip(self.min_shape, self.max_shape)
                ]
            if self.val_range:
                min_val = self.val_range[0]
                max_val = self.val_range[1]
                self.test_data.append(
                    np.array(self.np_rng.random(shape) * (max_val - min_val), dtype=self.dtype)
                    + min_val
                )
            elif self.dtype == np.float32:
                self.test_data.append(
                    np.array(self.np_rng.random(shape) * (1.0), dtype=self.dtype) - 0.5
                )
            else:
                self.test_data.append(np.array(self.np_rng.random(shape) * 255, dtype=self.dtype))

        batch = self.test_data
        self.i = (self.i + 1) % self.n
        return batch

    next = __next__


class ConstantDataIterator(object):
    def __init__(self, batch_size, sample_data, dtype):
        import_numpy()
        self.batch_size = batch_size
        self.test_data = []
        for _ in range(self.batch_size):
            self.test_data.append(np.array(sample_data, dtype=dtype))

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        batch = self.test_data
        self.i = (self.i + 1) % self.n
        return batch

    next = __next__


def check_output(outputs, ref_out, ref_is_list_of_outputs=None):
    """Checks the outputs of the pipeline.

    `outputs`
        return value from pipeline `run`
    `ref_out`
        a batch or tuple of batches
    `ref_is_list_of_outputs`
        only meaningful when there's just one output - if True, ref_out is a one-element
        list containing a single batch for output 0; otherwise ref_out _is_ a batch
    """
    import_numpy()
    if ref_is_list_of_outputs is None:
        ref_is_list_of_outputs = len(outputs) > 1

    assert ref_is_list_of_outputs or (len(outputs) == 1)

    for idx in range(len(outputs)):
        out = outputs[idx]
        ref = ref_out[idx] if ref_is_list_of_outputs else ref_out
        out = out.as_cpu()
        for i in range(len(out)):
            if not np.array_equal(out[i], ref[i]):
                print("Mismatch at sample", i)
                print("Out: ", out.at(i))
                print("Ref: ", ref[i])
            assert np.array_equal(out[i], ref[i])


def dali_type(t):
    import_numpy()
    if t is None:
        return None
    if t is np.float16:
        return dali_types.FLOAT16
    if t is np.float32:
        return dali_types.FLOAT
    if t is np.uint8:
        return dali_types.UINT8
    if t is np.int8:
        return dali_types.INT8
    if t is np.uint16:
        return dali_types.UINT16
    if t is np.int16:
        return dali_types.INT16
    if t is np.uint32:
        return dali_types.UINT32
    if t is np.int32:
        return dali_types.INT32
    raise TypeError("Unsupported type: " + str(t))


def py_buffer_from_address(address, shape, dtype, gpu=False):
    import_numpy()

    buff = {"data": (address, False), "shape": tuple(shape), "typestr": np.dtype(dtype).str}

    class py_holder(object):
        pass

    holder = py_holder()
    holder.__array_interface__ = buff
    holder.__cuda_array_interface__ = buff
    if not gpu:
        return np.array(holder, copy=False)
    else:
        import_cupy()
        return cp.asanyarray(holder)


class check_output_pattern:
    def __init__(self, pattern, is_regexp=True):
        self.pattern_ = pattern
        self.is_regexp_ = is_regexp

    def __enter__(self):
        self.bucket_out_ = tempfile.TemporaryFile(mode="w+")
        self.bucket_err_ = tempfile.TemporaryFile(mode="w+")
        self.stdout_fileno_ = 1
        self.stderr_fileno_ = 2
        self.old_stdout_ = os.dup(self.stdout_fileno_)
        self.old_stderr_ = os.dup(self.stderr_fileno_)
        os.dup2(self.bucket_out_.fileno(), self.stdout_fileno_)
        os.dup2(self.bucket_err_.fileno(), self.stderr_fileno_)

    def __exit__(self, exception_type, exception_value, traceback):
        self.bucket_out_.seek(0)
        self.bucket_err_.seek(0)
        os.dup2(self.old_stdout_, self.stdout_fileno_)
        os.dup2(self.old_stderr_, self.stderr_fileno_)
        our_data = self.bucket_out_.read()
        err_data = self.bucket_err_.read()

        pattern_found = False
        if self.is_regexp_:
            pattern = re.compile(self.pattern_)
            pattern_found = pattern.search(our_data) or pattern.search(err_data)
        else:
            pattern_found = (self.pattern_ in our_data or self.pattern_ in err_data,)

        assert pattern_found, (
            f"Pattern: ``{self.pattern_}`` \n not found in out: \n"
            f"``{our_data}`` \n and in err: \n ```{err_data}```"
        )


def dali_type_to_np(type):
    import_numpy()

    dali_types_to_np_dict = {
        dali_types.BOOL: np.bool_,
        dali_types.INT8: np.int8,
        dali_types.INT16: np.int16,
        dali_types.INT32: np.int32,
        dali_types.INT64: np.int64,
        dali_types.UINT8: np.uint8,
        dali_types.UINT16: np.uint16,
        dali_types.UINT32: np.uint32,
        dali_types.UINT64: np.uint64,
        dali_types.FLOAT16: np.float16,
        dali_types.FLOAT: np.float32,
        dali_types.FLOAT64: np.float64,
    }
    return dali_types_to_np_dict[type]


def np_type_to_dali(type):
    import_numpy()

    np_types_to_dali_dict = {
        np.bool_: dali_types.BOOL,
        np.int8: dali_types.INT8,
        np.int16: dali_types.INT16,
        np.int32: dali_types.INT32,
        np.int64: dali_types.INT64,
        np.uint8: dali_types.UINT8,
        np.uint16: dali_types.UINT16,
        np.uint32: dali_types.UINT32,
        np.uint64: dali_types.UINT64,
        np.float16: dali_types.FLOAT16,
        np.float32: dali_types.FLOAT,
        np.float64: dali_types.FLOAT64,
        np.longlong: dali_types.INT64,
        np.ulonglong: dali_types.UINT64,
    }
    return np_types_to_dali_dict[type]


def read_file_bin(filename):
    """
    Read file as bytes and insert it into numpy array
    :param filename: path to the file
    :return: numpy array
    """
    import_numpy()
    return np.fromfile(filename, dtype="uint8")


def filter_files(dirpath, suffix, exclude_subdirs=[]):
    """
    Read all file names recursively from a directory and filter those, which end with given suffix
    :param dirpath: Path to directory, from which the file names will be read
    :param suffix: String, which will be used to filter the files
    :return: List of file names
    """
    fnames = []
    for dir_name, subdir_list, file_list in os.walk(dirpath):
        for d in exclude_subdirs:
            if d in subdir_list:
                subdir_list.remove(d)
        flist = filter(lambda fname: fname.endswith(suffix), file_list)
        flist = map(lambda fname: os.path.join(dir_name, fname), flist)
        fnames.extend(flist)
    return fnames


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_last_n = 0
        self.max_val = 0

    def update(self, val, n=1):
        self.val = val
        self.max_val = max(self.max_val, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_array(dali_out):
    import_numpy()
    dali_out = dali_out.as_cpu()
    if isinstance(dali_out, TensorListCPU):
        dali_out = dali_out.as_array()
    return np.array(dali_out)


def module_functions(
    cls, prefix="", remove_prefix="", check_non_module=False, allowed_private_modules=[]
):
    res = []
    if hasattr(cls, "_schema_name"):
        prefix = prefix.replace(remove_prefix, "")
        prefix = prefix.lstrip(".")
        if len(prefix):
            prefix += "."
        else:
            prefix = ""
        res.append(prefix + cls.__name__)
    elif check_non_module or inspect.ismodule(cls):
        for c_name, c in inspect.getmembers(cls):

            def public_or_allowed(c_name):
                return not c_name.startswith("_") or c_name in allowed_private_modules

            if public_or_allowed(c_name) and c_name not in sys.builtin_module_names:
                res += module_functions(
                    c,
                    cls.__name__,
                    remove_prefix=remove_prefix,
                    check_non_module=check_non_module,
                    allowed_private_modules=allowed_private_modules,
                )
    return res


def get_files(path, ext):
    full_path = os.path.join(get_dali_extra_path(), path)
    audio_files = [
        os.path.join(full_path, f)
        for f in os.listdir(full_path)
        if re.match(f".*\\.{ext}", f) is not None
    ]
    return audio_files


def _test_skipped(reason=None):
    print("Test skipped." if reason is None else f"Test skipped: {reason}")


def restrict_python_version(major, minor=None):
    def decorator(test_case):
        version_info = sys.version_info
        if version_info.major > major or (
            version_info.major == major and (minor is None or version_info.minor >= minor)
        ):
            return test_case
        return lambda: _test_skipped(
            f"Insufficient Python version {version_info.major}.{version_info.minor} - "
            f"required {major}.{minor}"
        )

    return decorator


def generator_random_data(
    batch_size, min_sh=(10, 10, 3), max_sh=(100, 100, 3), dtype=None, val_range=[0, 255]
):
    import_numpy()
    if dtype is None:
        dtype = np.uint8
    assert len(min_sh) == len(max_sh)
    ndim = len(min_sh)

    def gen():
        out = []
        for _ in range(batch_size):
            shape = [
                np.random.randint(min_sh[d], max_sh[d] + 1, dtype=np.int32) for d in range(ndim)
            ]
            arr = np.array(np.random.uniform(val_range[0], val_range[1], shape), dtype=dtype)
            out += [arr]
        return out

    return gen


def generator_random_axes_for_3d_input(
    batch_size, use_negative=False, use_empty=False, extra_out_desc=[]
):
    import_numpy()

    def gen():
        ndim = 3
        options = [
            np.array([0, 1], dtype=np.int32),
            np.array([1, 0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
        ]
        if use_negative:
            options += [
                np.array([-2, -3], dtype=np.int32),
                np.array([-2, 0], dtype=np.int32),
                np.array([-3, 1], dtype=np.int32),
                np.array([0, -2], dtype=np.int32),
                np.array([1, -3], dtype=np.int32),
                np.array([-2], dtype=np.int32),
                np.array([-3], dtype=np.int32),
            ]
        if use_empty:
            # Add it 4 times to increase the probability to be chosen
            options += 4 * [np.array([], dtype=np.int32)]

        axes = []
        for _ in range(batch_size):
            axes.append(random.choice(options))

        num_extra_outs = len(extra_out_desc)
        extra_outputs = []
        for out_idx in range(num_extra_outs):
            extra_out = []
            for i in range(batch_size):
                axes_sh = axes[i].shape if axes[i].shape[0] > 0 else [ndim]
                range_start, range_end, dtype = extra_out_desc[out_idx]
                extra_out.append(
                    np.array(np.random.uniform(range_start, range_end, axes_sh), dtype=dtype)
                )
            extra_outputs.append(extra_out)
        return tuple([axes] + extra_outputs)

    return gen


def as_array(tensor):
    import_numpy()
    return np.array(tensor.as_cpu())


def python_function(*inputs, function, **kwargs):
    """
    Convenience wrapper around fn.python_function.
    If you need to pass to the fn.python_function mix of datanodes and parameters
    that are not produced by the pipeline, you probably need to proceed along the lines of:
    `dali.fn.python_function(data_node, function=lambda data:my_fun(data, non_pipeline_data))`.
    This utility separates the data nodes from non data nodes automatically,
    so that you can simply call `python_function(data_node, non_pipeline_data, function=my_fun)`.
    """
    node_inputs = [inp for inp in inputs if isinstance(inp, dali.data_node.DataNode)]
    const_inputs = [inp for inp in inputs if not isinstance(inp, dali.data_node.DataNode)]

    def is_data_node(input):
        return isinstance(input, dali.data_node.DataNode)

    def wrapper(*exec_inputs):
        iter_exec_inputs = (inp for inp in exec_inputs)
        iter_const_inputs = (inp for inp in const_inputs)
        iteration_inputs = [
            next(iter_exec_inputs if is_data_node(inp) else iter_const_inputs) for inp in inputs
        ]
        return function(*iteration_inputs)

    return dali.fn.python_function(*node_inputs, function=wrapper, **kwargs)


def has_operator(operator):
    def get_attr(obj, path):
        attrs = path.split(".")
        for attr in attrs:
            obj = getattr(obj, attr)
        return obj

    def decorator(fun):
        try:
            get_attr(dali.fn, operator)
        except AttributeError:

            @functools.wraps(fun)
            def dummy_case(*args, **kwargs):
                print(f"Omitting test case for unsupported operator: `{operator}`")

            return dummy_case
        else:
            return fun

    return decorator


def restrict_platform(min_compute_cap=None, platforms=None):
    spec = []
    if min_compute_cap is not None:
        compute_cap = get_arch()
        cond = f"compute cap ({compute_cap}) >= {min_compute_cap}"
        spec.append((cond, compute_cap >= min_compute_cap))
    if platforms is not None:
        import platform

        cond = f"platform.machine() ({platform.machine()}) in {platforms}"
        spec.append((cond, platform.machine() in platforms))

    def decorator(fun):
        if all(val for _, val in spec):
            return fun
        else:

            @functools.wraps(fun)
            def dummy_case(*args, **kwargs):
                print(f"Omitting test case in unsupported env: `{spec}`")

            return dummy_case

    return decorator


def check_numba_compatibility_cpu(if_skip=True):
    import numba

    # There's a bug in LLVM JIT linker that makes the tests fail
    # randomly on 64-bit ARM platform for some NUMBA versions.
    #
    # Numba bug:
    # https://github.com/numba/numba/issues/8567
    if platform.processor().lower() in ("arm64", "aarch64", "armv8") and (
        Version(numba.__version__) >= Version("0.57.0")
        and Version(numba.__version__) < Version("0.59.0")
    ):
        if if_skip:
            raise SkipTest()
        else:
            return False
    if not if_skip:
        return True


def check_numba_compatibility_gpu(if_skip=True):
    import nvidia.dali.plugin.numba.experimental as ex

    if not ex.NumbaFunction._check_minimal_numba_version(
        False
    ) or not ex.NumbaFunction._check_cuda_compatibility(False):
        if if_skip:
            raise SkipTest()
        else:
            return False
    if not if_skip:
        return True


def create_sign_off_decorator():
    _tested_ops = []

    class SignOff:
        def __call__(self, *op_names):
            assert all(isinstance(op_name, str) for op_name in op_names)
            assert len(op_names)
            _tested_ops.extend(op_names)

            def dummy(fn):
                return fn

            return dummy

        @property
        def tested_ops(self):
            return set(_tested_ops)

    return SignOff()


def load_test_operator_plugin():
    """Load plugin containing the test operators from: `dali/test/operators`."""
    test_bin_dir = os.path.dirname(dali.__file__) + "/test"
    try:
        plugin_manager.load_library(test_bin_dir + "/libtestoperatorplugin.so")
    except RuntimeError:
        # in conda "libtestoperatorplugin" lands inside lib/ dir
        plugin_manager.load_library("libtestoperatorplugin.so")


def is_of_supported(device_id=0):
    global is_of_supported_var
    if is_of_supported_var is not None:
        return is_of_supported_var

    driver_version_major = 0
    try:
        import pynvml

        pynvml.nvmlInit()
        driver_version = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
        driver_version_major = int(driver_version.split(".")[0])
    except ModuleNotFoundError:
        print("NVML not found")

    # there is an issue with OpticalFlow driver in R495 and newer on aarch64 platform
    is_of_supported_var = get_arch(device_id) >= 7.5 and (
        platform.machine() == "x86_64" or driver_version_major < 495
    )
    return is_of_supported_var
