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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import numpy as np
from test_utils import dali_type
from nvidia.dali import fn, pipeline_def, types
from nose2.tools import params


def normalize(x, axes=None, mean=None, stddev=None, ddof=0, eps=0):
    if type(axes) is list:
        axes = tuple(axes)

    num_reduced = np.prod([x.shape[a] for a in axes]) if axes else x.size

    if mean is None:
        mean = x.mean(axis=axes, keepdims=True)
        if stddev is None and eps == 0 and num_reduced > ddof:
            stddev = np.std(x, axis=axes, ddof=ddof, keepdims=True)

    if stddev is None:
        factor = num_reduced - ddof
        sqr = (x - mean).astype(float) ** 2
        var = np.sum(sqr, axis=axes, keepdims=True)
        if factor > 0:
            var /= factor
        else:
            var *= 0
        stddev = np.sqrt(var + eps)
    elif eps:
        stddev = np.sqrt(stddev**2 + eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (x - mean) / stddev
    return np.nan_to_num(norm, copy=False, nan=0, posinf=0, neginf=0)


def batch_reduced_vol(batch, axes):
    reduced_vol = 0
    if axes is None:
        for x in batch:
            reduced_vol += np.prod(x.shape)
    else:
        for x in batch:
            v = 1
            sh = x.shape
            for a in axes:
                v *= sh[a]
            reduced_vol += v
    return reduced_vol


# calculate mean over whole batch


def batch_mean(batch, axes):
    mean = None
    for x in batch:
        tmp = np.sum(x, axis=axes, keepdims=True)
        if mean is None:
            mean = tmp
        else:
            mean += tmp
    return mean / batch_reduced_vol(batch, axes)


# calculate standard deviation over whole batch
def batch_stddev(batch, axes, mean, ddof=0, eps=0):
    var = None
    for i, x in enumerate(batch):
        tmp = np.sum((x - mean) ** 2, axis=axes, keepdims=True)
        if var is None:
            var = tmp
        else:
            var += tmp
    factor = batch_reduced_vol(batch, axes) - ddof
    if factor > 0:
        var /= factor
    else:
        var *= 0
    return np.sqrt(var + eps)


def batch_norm(in_batch, axes=None, mean=None, stddev=None, ddof=0, eps=0):
    """
    normalize a batch as a whole
    non-reduced dims must have same extent in all batch items
    """
    if type(axes) is list:
        axes = tuple(axes)

    if mean is None:
        mean = batch_mean(in_batch, axes)

    if stddev is None:
        stddev = batch_stddev(in_batch, axes, mean, ddof, eps)
    elif eps:
        stddev = np.sqrt(stddev * stddev + eps)

    out = []
    for x in in_batch:
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = (x - mean) / stddev
        out.append(np.nan_to_num(norm, copy=False, nan=0, posinf=0, neginf=0))
    return out


def generate_data(dims, batch_size, batch_norm, axes, dtype=None):
    """
    Generate random tensors with given dimensionality.
    If batch_norm is True, the extents in non-reduced axe
    If no using batch_norm, axes argument is ignored.
    """
    shapes = np.random.randint(1, 10, [batch_size, dims], dtype=int)
    if batch_norm and axes is not None:
        for i in range(1, batch_size):
            for a in range(dims):
                if a not in axes:
                    shapes[i, a] = shapes[0, a]
    shapes = shapes.tolist()
    scale = 1
    if dtype is None:
        dtype = np.float32
    elif dtype is not np.float32:
        scale = 255
    return [
        (scale * (np.random.rand(*s).astype(np.float32) * (1 + i) - i)).astype(dtype)
        for i, s in enumerate(shapes)
    ]


def custom_mean(batch_norm, axes):
    bias = 0.3  # make the result purposefully slightly off
    if type(axes) is list:
        axes = tuple(axes)
    if batch_norm:

        def whole_batch_mean(batch):
            out = batch_mean(batch, axes) + bias
            return [out.astype(np.float32) for _ in range(len(batch))]

        return whole_batch_mean
    else:

        def per_sample_mean(batch):
            ret = [x.mean(axis=axes, keepdims=True, dtype=np.float32) + bias for x in batch]
            return ret

        return per_sample_mean


def custom_stddev(batch_norm, axes):
    bias = 1.3  # make the result purposefully slightly off
    mean_func = custom_mean(batch_norm, axes)
    if type(axes) is list:
        axes = tuple(axes)
    if batch_norm:

        def whole_batch_stddev(batch):
            mean = mean_func(batch)[0][0]
            out = bias * batch_stddev(batch, axes, mean)
            return [out for _ in range(len(batch))]

        return whole_batch_stddev
    else:

        def per_sample_stddev(batch):
            mean = mean_func(batch)
            out = []
            for i in range(len(batch)):
                stddev = bias * np.sqrt(((batch[i] - mean[i]) ** 2).mean(axis=axes, keepdims=True))
                out.append(stddev)
            return out

        return per_sample_stddev


def normalize_list(whole_batch, data_batch, axes=None, mean=None, stddev=None, ddof=0, eps=0):
    if whole_batch:
        return batch_norm(data_batch, axes, mean, stddev, ddof, eps)
    else:
        if type(mean) is not list:
            mean = [mean] * len(data_batch)
        if type(stddev) is not list:
            stddev = [stddev] * len(data_batch)
        return [
            normalize(data_batch[i].astype(float), axes, mean[i], stddev[i], ddof, eps)
            for i in range(len(data_batch))
        ]


def err(l1, l2):
    return np.max([np.max(np.abs(a[0] - a[1])) for a in zip(l1, l2)])


def check_float(l1, l2, eps=1e-3):
    for i, a in enumerate(zip(l1, l2)):
        assert np.allclose(a[0], a[1], rtol=1e-3, atol=eps)


def check_integer(actual, ref, input=None):
    for i, a in enumerate(zip(actual, ref)):
        t = a[0].dtype
        min = np.iinfo(t).min
        max = np.iinfo(t).max
        a1 = np.clip(a[1], min, max)
        # actual values are saturated, so we must clip the reference, too
        assert np.allclose(a[0], a1, atol=2)


def shift_scale(batch, shift, scale):
    for i in range(len(batch)):
        batch[i] = batch[i] * scale + shift


class NormalizePipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        dims,
        axes,
        axis_names,
        batch=False,
        out_type=None,
        in_type=None,
        shift=None,
        scale=None,
        num_threads=3,
        device_id=0,
        num_gpus=1,
    ):
        super(NormalizePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False
        )
        common_args = {
            "device": device,
            "axes": axes,
            "axis_names": axis_names,
            "batch": batch,
            "dtype": dali_type(out_type),
            "shift": shift,
            "scale": scale,
        }
        self.in_type = in_type
        self.out_type = out_type
        self.device = device
        self.input = ops.ExternalSource()
        self.add_layout = None
        if axis_names is not None:
            layout = ""
            for i in range(dims):
                layout += chr(ord("a") + i)
            self.add_layout = ops.Reshape(layout=layout)
        self.batch = batch
        self.dims = dims
        self.has_axes = axes is not None or axis_names is not None
        self.scale = scale
        self.shift = shift
        self.is_integral = out_type is not None and out_type is not np.float32

        if axis_names is not None:
            axes = []
            for a in axis_names:
                axes.append(ord(a) - ord("a"))

        self.axes = axes
        self.axis_names = axis_names
        self.ddof = 2 if axes is not None and len(axes) > 0 else 0
        self.eps = 0.25

        self.mean = ops.PythonFunction(function=custom_mean(batch, axes), batch_processing=True)
        self.stddev = ops.PythonFunction(function=custom_stddev(batch, axes), batch_processing=True)
        self.normalize = ops.Normalize(**common_args, ddof=self.ddof)
        self.scalar_mean = ops.Normalize(**common_args, mean=1, ddof=self.ddof, epsilon=self.eps)
        self.scalar_stddev = ops.Normalize(**common_args, stddev=2, epsilon=self.eps)
        self.scalar_params = ops.Normalize(**common_args, mean=1, stddev=2)

    def define_graph(self):
        data = self.input_data = self.input()
        if self.add_layout is not None:
            data = self.add_layout(data)
        mean = self.mean(data)
        stddev = self.stddev(data)

        dev_data = data.gpu() if self.device == "gpu" else data
        normalized = self.normalize(dev_data)
        scalar_mean = self.scalar_mean(dev_data)
        scalar_stddev = self.scalar_stddev(dev_data)
        if not self.batch:
            ext_mean = self.normalize(dev_data, mean=mean)
            ext_stddev = self.normalize(dev_data, stddev=stddev)
            ext_all = self.normalize(dev_data, mean=mean, stddev=stddev)
            scalar_mean_ext = self.scalar_mean(dev_data, stddev=stddev)
            scalar_stddev_ext = self.scalar_stddev(dev_data, mean=mean)
        if not self.has_axes:
            scalar_params = self.scalar_params(dev_data)

        out = [data, mean, stddev, normalized, scalar_mean, scalar_stddev]
        if not self.batch:
            out += [ext_mean, ext_stddev, ext_all, scalar_mean_ext, scalar_stddev_ext]
        if not self.has_axes:
            out.append(scalar_params)
        return out

    def check_batch(
        self,
        data,
        mean,
        stddev,
        normalized,
        scalar_mean=None,
        scalar_stddev=None,
        ext_mean=None,
        ext_stddev=None,
        ext_all=None,
        scalar_mean_ext=None,
        scalar_stddev_ext=None,
        scalar_params=None,
    ):
        axes = self.axes
        if type(axes) is list:
            axes = tuple(axes)
        batch = self.batch
        mean_func = custom_mean(batch, axes)
        stddev_func = custom_stddev(batch, axes)
        scale = 1 if self.scale is None else self.scale
        shift = 0 if self.shift is None else self.shift

        def check(l1, l2):
            if self.is_integral:
                check_integer(l1, l2, data)
            else:
                eps = 1e-3 * scale * len(data[0].shape)
                check_float(l1, l2, eps)

        ref = normalize_list(batch, data, axes, ddof=self.ddof)
        ref_scalar_mean = normalize_list(batch, data, axes, mean=1, ddof=self.ddof, eps=self.eps)
        ref_scalar_stddev = normalize_list(batch, data, axes, stddev=2, eps=self.eps)
        shift_scale(ref, shift, scale)
        shift_scale(ref_scalar_mean, shift, scale)
        shift_scale(ref_scalar_stddev, shift, scale)
        mean = mean_func(data)
        stddev = stddev_func(data)

        check(normalized, ref)
        check(scalar_mean, ref_scalar_mean)
        check(scalar_stddev, ref_scalar_stddev)

        if not batch:
            ref_ext_mean = normalize_list(batch, data, axes, mean=mean, ddof=self.ddof)
            ref_ext_stddev = normalize_list(batch, data, axes, stddev=stddev, ddof=self.ddof)
            ref_ext_all = normalize_list(batch, data, axes, mean=mean, stddev=stddev)
            ref_scalar_mean_ext = normalize_list(
                batch, data, axes, mean=1, stddev=stddev, ddof=self.ddof, eps=self.eps
            )
            ref_scalar_stddev_ext = normalize_list(
                batch, data, axes, mean=mean, stddev=2, eps=self.eps
            )

            shift_scale(ref_ext_mean, shift, scale)
            shift_scale(ref_ext_stddev, shift, scale)
            shift_scale(ref_ext_all, shift, scale)
            shift_scale(ref_scalar_mean_ext, shift, scale)
            shift_scale(ref_scalar_stddev_ext, shift, scale)

            check(ext_mean, ref_ext_mean)
            check(ext_stddev, ref_ext_stddev)
            check(ext_all, ref_ext_all)
            check(scalar_mean_ext, ref_scalar_mean_ext)
            check(scalar_stddev_ext, ref_scalar_stddev_ext)

        if scalar_params is not None:
            ref_scalar_params = normalize_list(batch, data, axes, mean=1, stddev=2)
            shift_scale(ref_scalar_params, shift, scale)
            check(scalar_params, ref_scalar_params)

    def iter_setup(self):
        data = generate_data(self.dims, self.batch_size, self.batch, self.axes, dtype=self.in_type)
        self.feed_input(self.input_data, data)


def to_list(tensor_list):
    tensor_list = tensor_list.as_cpu()
    out = []
    for i in range(len(tensor_list)):
        out.append(tensor_list.at(i))
    return out


np.random.seed(seed=1337)


def mask2axes(mask):
    out = []
    a = 0
    while mask:
        if mask & 1:
            out.append(a)
        mask >>= 1
        a += 1
    return out


def all_axes(dim):
    yield None
    for mask in range(1, 1 << dim):
        yield mask2axes(mask)


def _run_test(
    device,
    batch_size,
    dim,
    axes,
    axis_names,
    batch_norm,
    out_type=None,
    in_type=None,
    shift=None,
    scale=None,
):
    kind = "inter-sample" if batch_norm else "per-sample"
    msg = "{0}, {1}, batch = {2}, dim = {3}".format(device, kind, batch_size, dim)
    if axes is not None:
        msg += " axes = {}".format(axes)
    if axis_names is not None:
        msg += " axis_names = '{}'".format(axis_names)
    if out_type is not None:
        msg += " output = {}".format(out_type)
    if in_type is not None:
        msg += " input = {}".format(in_type)
    print(msg)

    pipe = NormalizePipeline(
        device, batch_size, dim, axes, axis_names, batch_norm, out_type, in_type, shift, scale
    )
    for iter in range(2):
        out = pipe.run()
        pipe.check_batch(*[to_list(x) for x in out])


def axes2names(axes, layout="abcdefghijklmnopqrstuvwxyz"):
    return "".join([layout[axis] for axis in axes])


def _test_up_to_5D_all_axis_combinations(device):
    batch_size = 5
    for batch_norm in [False, True]:
        for dim in range(1, 6):
            for axes in all_axes(dim):
                yield _run_test, device, batch_size, dim, axes, None, batch_norm
                if axes is not None and dim < 5:
                    axis_names = axes2names(axes)
                    yield _run_test, device, batch_size, dim, None, axis_names, batch_norm


def test_cpu_up_to_5D_all_axis_combinations():
    for device in ["cpu", "gpu"]:
        for x in _test_up_to_5D_all_axis_combinations(device):
            yield x


def test_types():
    batch_size = 50
    dim = 4
    axes = [1, 2]
    out_type = np.uint8
    in_type = None
    for device in ["cpu", "gpu"]:
        for out_type, scale, shift in [
            (np.uint8, 64, 128),
            (np.int16, 1000, 0),
            (np.float32, 0.5, 0.5),
        ]:
            for in_type in [None, np.uint8, np.int16, np.float32]:
                yield (
                    _run_test,
                    device,
                    batch_size,
                    dim,
                    axes,
                    None,
                    False,
                    out_type,
                    in_type,
                    shift,
                    scale,
                )


@params("cpu", "gpu")
def test_batch_of_empty_samples(device):
    @pipeline_def
    def pipeline():
        empty_sample = types.Constant([])
        if device == "gpu":
            empty_sample = empty_sample.gpu()
        return fn.normalize(empty_sample, mean=5, stddev=1)

    p = pipeline(batch_size=4, device_id=0, num_threads=4)
    p.run()
