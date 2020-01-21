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
    for x in batch:
        tmp = np.sum((x - mean)**2, axis = axes, keepdims = True)
        if stddev is None:
            stddev = tmp
        else:
            stddev += tmp
    return np.sqrt(stddev / batch_reduced_vol(batch, axes))

# normalize a batch as a whole
# non-reduced dims must have same extent in all batch items
def normalize_batch(in_batch, axes = None, mean = None, stddev = None):
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
    if batch_norm:
        for i in range(1, batch_size):
            for a in range(dims):
                if a not in axes:
                    shapes[i, a] = shapes[0, a]
    shapes = shapes.tolist()
    return [np.random.rand(*s) for s in shapes]

def custom_mean(axes, batch_norm):
    bias = 0.3  # make the result purposefully slightly off
    if type(axes) is list:
        axes = tuple(axes)
    if batch_norm:
        def whole_batch_mean(batch):
            out = batch_mean(batch, axes) + bias
            return [[out for _ in range(len(batch))]]
        return whole_batch_mean
    else:
        def per_sample_mean(batch):
            return [[x.mean(axis = axes, keepdims = True) + bias for x in batch]]
        return per_sample_mean

def custom_stddev(axes, batch_norm):
    bias = 1.3  # make the result purposefully slightly off
    mean_func = custom_mean(axes, batch_norm)
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
            for i in len(batch):
                stddev = bias * np.sqrt(((batch[i] - mean[i])**2).mean(axis = axes, keepdims = True))
                out.append(stddev)
            return [out]
        return per_sample_stddev

class NormalizePipeline(Pipeline):
    def __init__(self, device, batch_size, dims, axis_names, axes, batch = False,
                num_threads=3, device_id=0, num_gpus=1):
        super(NormalizePipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.device = device
        self.input = ops.ExternalSource()
        self.axes = axes
        self.batch = batch
        self.dims = dims
        self.mean = ops.PythonFunction(function = custom_mean(axes, batch), batch_processing=True)
        self.stddev = ops.PythonFunction(function = custom_stddev(axes, batch), batch_processing=True)
        self.normalize = ops.Normalize(axes = axes, axis_names = axis_names)
        self.no_norm = ops.Normalize(axes = axes, axis_names = axis_names, stddev = 1)
        self.no_center = ops.Normalize(axes = axes, axis_names = axis_names, mean = 0)

    def define_graph(self):
        data = self.input_data = self.input()
        mean = self.mean(data)
        stddev = self.stddev(data)
        normalized = self.normalize(data)
        no_norm = self.no_norm(data)
        no_center = self.no_center(data)
        #ext_mean = self.normalize(data, mean = mean)
        #ext_stddev = self.normalize(data, stddev = stddev)
        #ext_all = self.normalize(data, mean = mean, stddev = stddev)
        return [data]#, normalized, no_norm, no_center, ext_mean, ext_stddev, ext_all]

    def check_batch(batch):
        pass

    def iter_setup(self):
        self.feed_input(self.input_data, generate_data(self.dims, self.batch_size, self.batch, self.axes))

def main():
    axes = [1, 3]
    pipe = NormalizePipeline("cpu", 10, 4, None, axes, True)
    pipe.build()
    out = pipe.run()

if __name__ == '__main__':
    main()


