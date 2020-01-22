from __future__ import print_function
from __future__ import division
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np

def normalize(x, axes = None, mean = None, stddev = None):
    if type(axes) is list:
        axes = tuple(axes)
    if mean is None and stddev is None:
        mean = x.mean(axis = axes, keepdims = True)
        stddev = np.std(x, axis = axes, keepdims=True)
    elif mean is None:
        mean = x.mean(axis = axes, keepdims = True)
    elif stddev is None:
        stddev = np.sqrt(((x - mean)**2).mean(axis = axes, keepdims = True))

    with np.errstate(divide='ignore', invalid='ignore'):
        norm = (x - mean) / stddev
    return np.nan_to_num(norm, copy = False, nan = 0, posinf = 0, neginf = 0)

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
        tmp = np.sum(x, axis = axes, keepdims = True)
        if mean is None:
            mean = tmp
        else:
            mean += tmp
    return mean / batch_reduced_vol(batch, axes)


# calculate standard deviation over whole batch
def batch_stddev(batch, axes, mean):
    stddev = None
    if type(mean) is not list:
        mean = [mean] * len(batch)
    for i, x in enumerate(batch):
        tmp = np.sum((x - mean[i])**2, axis = axes, keepdims = True)
        if stddev is None:
            stddev = tmp
        else:
            stddev += tmp
    return np.sqrt(stddev / batch_reduced_vol(batch, axes))

# normalize a batch as a whole
# non-reduced dims must have same extent in all batch items
def batch_norm(in_batch, axes = None, mean = None, stddev = None):
    if type(axes) is list:
        axes = tuple(axes)

    if mean is None:
        mean = batch_mean(in_batch, axes)

    if stddev is None:
        stddev = batch_stddev(in_batch, axes, mean)

    out = []
    for x in in_batch:
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = (x - mean) / stddev
        out.append(np.nan_to_num(norm, copy = False, nan = 0, posinf = 0, neginf = 0))
    return out

# Generate random tensors with given dimensionality.
# If batch_norm is True, the extents in non-reduced axes are equal.
# If no using batch_norm, axes argument is ignored.
def generate_data(dims, batch_size, batch_norm, axes):
    shapes = np.random.randint(1, 10, [batch_size, dims], dtype=int)
    if batch_norm and axes is not None:
        for i in range(1, batch_size):
            for a in range(dims):
                if a not in axes:
                    shapes[i, a] = shapes[0, a]
    shapes = shapes.tolist()
    return [np.random.rand(*s).astype(np.float32) for s in shapes]

def custom_mean(batch_norm, axes):
    bias = 0.3  # make the result purposefully slightly off
    if type(axes) is list:
        axes = tuple(axes)
    if batch_norm:
        def whole_batch_mean(batch):
            out = batch_mean(batch, axes) + bias
            return [[out.astype(np.float32) for _ in range(len(batch))]]
        return whole_batch_mean
    else:
        def per_sample_mean(batch):
            return [[x.mean(axis = axes, keepdims = True, dtype=np.float32) + bias for x in batch]]
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
            return [[out for _ in range(len(batch))]]
        return whole_batch_stddev
    else:
        def per_sample_stddev(batch):
            mean = mean_func(batch)[0]
            out = []
            for i in range(len(batch)):
                stddev = bias * np.sqrt(((batch[i] - mean[i])**2).mean(axis = axes, keepdims = True))
                out.append(stddev)
            return [out]
        return per_sample_stddev

def normalize_list(whole_batch, data_batch, axes = None, mean = None, stddev = None):
    if whole_batch:
        return batch_norm(data_batch, axes, mean, stddev)
    else:
        if type(mean) is not list:
            mean = [mean] * len(data_batch)
        if type(stddev) is not list:
            stddev = [stddev] * len(data_batch)
        return [normalize(data_batch[i], axes, mean[i], stddev[i]) for i in range(len(data_batch))]

def err(l1, l2):
    return np.max([np.max(np.abs(a[0] - a[1])) for a in zip(l1, l2)])

def check(l1, l2):
    for a in zip(l1, l2):
        np.allclose(a[0], a[1], rtol=1e-3, atol=1e-3)

class NormalizePipeline(Pipeline):
    def __init__(self, device, batch_size, dims, axis_names, axes, batch = False,
                num_threads=3, device_id=0, num_gpus=1):
        super(NormalizePipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.device = device
        self.input = ops.ExternalSource()
        self.axes = axes
        self.batch = batch
        self.dims = dims
        self.mean = ops.PythonFunction(function = custom_mean(batch, axes), batch_processing=True)
        self.stddev = ops.PythonFunction(function = custom_stddev(batch, axes), batch_processing=True)
        self.normalize = ops.Normalize(axes = axes, axis_names = axis_names)
        self.scalar_mean = ops.Normalize(axes = axes, axis_names = axis_names, mean = 1)
        self.scalar_stddev = ops.Normalize(axes = axes, axis_names = axis_names, stddev = 2)
        self.scalar_params = ops.Normalize(axes = axes, axis_names = axis_names, mean = 1, stddev = 2)

    def define_graph(self):
        data = self.input_data = self.input()
        mean = self.mean(data)
        stddev = self.stddev(data)
        normalized = self.normalize(data)
        scalar_mean = self.scalar_mean(data)
        scalar_stddev = self.scalar_stddev(data)
        ext_mean = self.normalize(data, mean = mean)
        ext_stddev = self.normalize(data, stddev = stddev)
        ext_all = self.normalize(data, mean = mean, stddev = stddev)
        scalar_mean_ext = self.scalar_mean(data, stddev = stddev)
        scalar_stddev_ext = self.scalar_stddev(data, mean = mean)
        if self.axes is None:
            scalar_params = self.scalar_params(data)

        out = [data, mean, stddev, normalized, scalar_mean, scalar_stddev,
                ext_mean, ext_stddev, ext_all, scalar_mean_ext, scalar_stddev_ext]
        if self.axes is None:
            out.append(scalar_params)
        return out

    def check_batch(self, data, mean, stddev, normalized, scalar_mean, scalar_stddev,
                ext_mean, ext_stddev, ext_all, scalar_mean_ext, scalar_stddev_ext, scalar_params = None):
        axes = self.axes
        if type(axes) is list:
            axes = tuple(axes)
        batch = self.batch
        mean_func = custom_mean(batch, axes)
        stddev_func = custom_stddev(batch, axes)

        ref = normalize_list(batch, data, axes)
        ref_scalar_mean = normalize_list(batch, data, axes, mean = 1)
        ref_scalar_stddev = normalize_list(batch, data, axes, stddev = 2)
        mean, = mean_func(data)
        stddev, = stddev_func(data)
        ref_ext_mean = normalize_list(batch, data, axes, mean = mean)
        ref_ext_stddev = normalize_list(batch, data, axes, stddev = stddev)
        ref_ext_all = normalize_list(batch, data, axes, mean = mean, stddev = stddev)
        ref_ext_all = normalize_list(batch, data, axes, mean = mean, stddev = stddev)
        ref_scalar_mean_ext = normalize_list(batch, data, axes, mean = 1, stddev = stddev)
        ref_scalar_stddev_ext = normalize_list(batch, data, axes, mean = mean, stddev = 2)
        check(scalar_stddev, ref_scalar_stddev)
        check(scalar_mean, ref_scalar_mean)
        check(ext_mean, ref_ext_mean)
        check(ext_stddev, ref_ext_stddev)
        check(ext_all, ref_ext_all)
        check(scalar_mean_ext, ref_scalar_mean_ext)
        check(scalar_stddev_ext, ref_scalar_stddev_ext)
        if scalar_params is not None:
            ref_scalar_params = normalize_list(batch, data, axes, mean = 1, stddev = 2)
            check(scalar_params, ref_scalar_params)

    def iter_setup(self):
        self.feed_input(self.input_data, generate_data(self.dims, self.batch_size, self.batch, self.axes))

def to_list(tensor_list):
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

def main():
    for whole_batch in [False, True]:
        kind = "batch" if whole_batch else "per-sample"
        for dim in range(1, 6):
            print("Testing from dimensionality = ", dim)
            for axes in all_axes(dim):
                print(kind, ", dim", dim, " axes: ", axes)
                pipe = NormalizePipeline("cpu", 10, dim, None, axes, True)
                pipe.build()
                out = pipe.run()
                pipe.check_batch(*[to_list(x) for x in out])

if __name__ == '__main__':
    main()


