# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from nvidia.dali.edge import EdgeReference
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os
from functools import partial
from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import get_dali_extra_path
from test_utils import RandomDataIterator

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')
test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')

class SlicePipeline(Pipeline):
    def __init__(self, device, batch_size, pos_size_iter,
                 num_threads=1, device_id=0, is_fused_decoder=False,
                 dims=(1,0), dim_names="WH"):
        super(SlicePipeline, self).__init__(batch_size,
                                            num_threads,
                                            device_id,
                                            seed=1234)
        self.is_fused_decoder = is_fused_decoder
        self.pos_size_iter = pos_size_iter
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle=False)
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()

        if self.is_fused_decoder:
            self.decode = ops.ImageDecoderSlice(device = 'cpu',
                                                output_type = types.RGB)
        else:
            self.decode = ops.ImageDecoder(device = "cpu",
                                           output_type = types.RGB)
            self.slice = ops.Slice(device = device,
                                   image_type = types.RGB,
                                   dims=(1,0))

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()

        if self.is_fused_decoder:
            images = self.decode(inputs, self.crop_pos, self.crop_size)
        else:
            images = self.decode(inputs)
            if self.device == 'gpu':
                images = images.gpu()
            images = self.slice(images, self.crop_pos, self.crop_size)
        return images

    def iter_setup(self):
        (crop_pos, crop_size) = self.pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)

class SliceArgsIterator(object):
    def __init__(self,
                 batch_size,
                 num_dims=3,
                 image_shape=None,  # Needed if normalized_anchor and normalized_shape are False
                 image_layout=None, # Needed if dim_names is used to specify the slice
                 normalized_anchor=True,
                 normalized_shape=True,
                 dims=None,
                 dim_names=None,
                 min_norm_anchor=0.0,
                 max_norm_anchor=0.2,
                 min_norm_shape=0.4,
                 max_norm_shape=0.75,
                 seed=54643613):
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.image_shape = image_shape
        self.image_layout = image_layout
        self.normalized_anchor = normalized_anchor
        self.normalized_shape = normalized_shape
        self.dims = dims
        self.dim_names = dim_names
        self.min_norm_anchor=min_norm_anchor
        self.max_norm_anchor=max_norm_anchor
        self.min_norm_shape=min_norm_shape
        self.max_norm_shape=max_norm_shape
        self.seed=seed

        if not self.dim_names and not self.dims:
            self.dim_names = "WH"

        if self.dim_names:
            self.dims = []
            for dim_name in self.dim_names:
                assert dim_name in self.image_layout
                self.dims.append(self.image_layout.index(dim_name))
        assert(len(self.dims)>0)

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        pos = []
        size = []
        anchor_amplitude = self.max_norm_anchor - self.min_norm_anchor
        anchor_offset = self.min_norm_anchor
        shape_amplitude = self.max_norm_shape - self.min_norm_shape
        shape_offset = self.min_norm_shape
        np.random.seed(self.seed)
        for k in range(self.batch_size):
            norm_anchor = anchor_amplitude * np.random.rand(len(self.dims)) + anchor_offset
            norm_shape = shape_amplitude * np.random.rand(len(self.dims)) + shape_offset
            if self.normalized_anchor:
                anchor = norm_anchor
            else:
                anchor = norm_anchor * image_shape

            if self.normalized_shape:
                shape = norm_shape
            else:
                shape = norm_shape * image_shape
            pos.append(np.asarray(anchor, dtype=np.float32))
            size.append(np.asarray(shape, dtype=np.float32))
            self.i = (self.i + 1) % self.n
        return (pos, size)
    next = __next__

def check_slice_vs_fused_decoder(device, batch_size, normalized_anchor, normalized_shape, dims, dim_names):
    eii1 = SliceArgsIterator(batch_size)
    eii2 = SliceArgsIterator(batch_size)
    compare_pipelines(SlicePipeline(device, batch_size, iter(eii1), is_fused_decoder=True),
                      SlicePipeline(device, batch_size, iter(eii2), is_fused_decoder=False),
                      batch_size=batch_size, N_iterations=5)

def tes3t_slice_vs_fused_decoder():
    for device in ['cpu', 'gpu']:
        for batch_size in [1, 13, 64]:
            for normalized_anchor, normalized_shape, dims, dim_names in \
                [(True, True, (0,1), "WH"),
                 (True, True, (1,0), "HW"),
                 (True, True, (2), "C")]:
                yield check_slice_vs_fused_decoder, device, batch_size, normalized_anchor, \
                    normalized_shape, dims, dim_names

def check_slice_vs_cpu_vs_gpu(device, batch_size, normalized_anchor, normalized_shape, dims, dim_names):
    eii1 = SliceArgsIterator(batch_size)
    eii2 = SliceArgsIterator(batch_size)
    compare_pipelines(SlicePipeline(device, batch_size, iter(eii1), is_fused_decoder=True),
                      SlicePipeline(device, batch_size, iter(eii2), is_fused_decoder=False),
                      batch_size=batch_size, N_iterations=5)

def te3st_slice_cpu_vs_gpu():
    for device in ['cpu', 'gpu']:
        for batch_size in [1, 13, 64]:
            for normalized_anchor, normalized_shape, dims, dim_names in \
                [(True, True, (0,1), "WH"),
                 (True, True, (1,0), "HW")]:
                yield check_slice_vs_cpu_vs_gpu, device, batch_size, normalized_anchor, \
                    normalized_shape, dims, dim_names

class SliceArgsIteratorExtractFirstChannel(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        pos = []
        size = []
        for k in range(self.batch_size):
            pos.append(np.asarray([0.0, 0.0, 0.0], dtype=np.float32)) # yxc
            size.append(np.asarray([1.0, 1.0, 1./3.], dtype=np.float32)) # HWC
            self.i = (self.i + 1) % self.n
        return (pos, size)
    next = __next__

class PythonOperatorPipeline(Pipeline):
    def __init__(self, function, batch_size, num_threads=1, device_id=0):
        super(PythonOperatorPipeline, self).__init__(batch_size, num_threads, device_id,
                                                     exec_async=False,
                                                     exec_pipelined=False,
                                                     seed=1234)
        self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle=False)
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)
        self.python_function = ops.PythonFunction(function=function)

    def define_graph(self):
        jpegs, _ = self.input()
        decoded = self.decode(jpegs)
        processed = self.python_function(decoded)
        assert isinstance(processed, EdgeReference)
        return processed

def extract_first_channel(image):
    return image[:,:,0].reshape(image.shape[0:2] + (1,))

def te3st_slice_extract_channel_cpu():
    for batch_size in {1, 32, 64}:
        eii = SliceArgsIteratorExtractFirstChannel(batch_size)
        compare_pipelines(SlicePipeline('cpu', batch_size, iter(eii)),
                          PythonOperatorPipeline(extract_first_channel, batch_size),
                          batch_size=batch_size, N_iterations=10)

def te3st_slice_extract_channel_gpu():
    for batch_size in {1, 32, 64}:
        eii = SliceArgsIteratorExtractFirstChannel(batch_size)
        compare_pipelines(SlicePipeline('gpu', batch_size, iter(eii)),
                          PythonOperatorPipeline(extract_first_channel, batch_size),
                          batch_size=batch_size, N_iterations=10)

def slice_func(image):
    start_y = int(np.float32(image.shape[0]) * np.float32(0.2))
    end_y = int(np.float32(image.shape[0]) * np.float32(0.2 + 0.5))
    start_x = int(np.float32(image.shape[1]) * np.float32(0.4))
    end_x = int(np.float32(image.shape[1]) * np.float32(0.4 + 0.3))
    return image[start_y:end_y, start_x:end_x, :]

def te3st_slice_vs_numpy_slice_gpu():
    for batch_size in {1, 32, 64}:
        eii = SliceArgsIteratorAllDims(batch_size)
        compare_pipelines(SlicePipeline('gpu', batch_size, iter(eii)),
                          PythonOperatorPipeline(slice_func, batch_size),
                          batch_size=batch_size, N_iterations=10)

def t3st_slice_vs_numpy_slice_cpu():
    for batch_size in {1, 32, 64}:
        eii = SliceArgsIteratorAllDims(batch_size)
        compare_pipelines(SlicePipeline('cpu', batch_size, iter(eii)),
                          PythonOperatorPipeline(slice_func, batch_size),
                          batch_size=batch_size, N_iterations=10)

class SliceSynthDataPipeline(Pipeline):
    def __init__(self, device, batch_size, layout, iterator, pos_size_iter,
                 num_threads=1, device_id=0, num_gpus=1,
                 dims=None, dim_names=None):
        super(SliceSynthDataPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.pos_size_iter = pos_size_iter
        self.inputs = ops.ExternalSource()
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()

        if not dims and not dim_names:
            self.slice = ops.Slice(device = self.device)
        elif dim_names:
            self.slice = ops.Slice(device = self.device,
                                   dim_names = dim_names)
        elif dims:
            self.slice = ops.Slice(device = self.device,
                                   dims = dims)

    def define_graph(self):
        self.data = self.inputs()
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()
        data = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.slice(data, self.crop_pos, self.crop_size)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

        (crop_pos, crop_size) = self.pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)

class PythonOperatorPipeline(Pipeline):
    def __init__(self, function, batch_size, num_threads=1, device_id=0):
        super(PythonOperatorPipeline, self).__init__(batch_size, num_threads, device_id,
                                                     exec_async=False,
                                                     exec_pipelined=False,
                                                     seed=1234)
        self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle=False)
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)
        self.python_function = ops.PythonFunction(function=function)

    def define_graph(self):
        jpegs, _ = self.input()
        decoded = self.decode(jpegs)
        processed = self.python_function(decoded)
        assert isinstance(processed, EdgeReference)
        return processed

def slice_func_helper(dims, dim_names, layout, image, slice_anchor, slice_shape):
    # TODO(janton): remove this
    if not dims and not dim_names:
        dim_names = "WH"

    if dim_names:
        dims = []
        for dim_name in dim_names:
            assert(dim_name in layout)
            dim_pos = layout.find(dim_name)
            dims.append(dim_pos)

    shape = image.shape
    full_slice_anchor = [0] * len(shape)
    full_slice_shape = list(shape)
    for dim in dims:
        idx = dims.index(dim)
        full_slice_anchor[dim] = slice_anchor[idx]
        full_slice_shape[dim] = slice_shape[idx]

    start = [int(np.float32(shape[i]) * np.float32(full_slice_anchor[i]))
             for i in range(len(shape))]
    end = [int(np.float32(shape[i]) * np.float32(full_slice_anchor[i] + full_slice_shape[i]))
           for i in range(len(shape))]

    if len(full_slice_anchor) == 3:
        return image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    else:
        assert(False)

class SliceSynthDataPipelinePythonOp(Pipeline):
    def __init__(self, batch_size, layout, iterator, pos_size_iter,
                 num_threads=1, device_id=0, num_gpus=1,
                 dims=None, dim_names=None):
        super(SliceSynthDataPipelinePythonOp, self).__init__(
            batch_size, num_threads, device_id,
            seed=12345, exec_async=False, exec_pipelined=False)
        self.device = "cpu"
        self.layout = layout
        self.iterator = iterator
        self.pos_size_iter = pos_size_iter
        self.inputs = ops.ExternalSource()
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()

        function = partial(
            slice_func_helper, dims, dim_names, self.layout)
        self.slice = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()
        out = self.slice(self.data, self.crop_pos, self.crop_size)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

        (crop_pos, crop_size) = self.pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)


def check_slice_synth_data_cpu_vs_gpu(device, batch_size, input_shape, layout, dims, dim_names):
    eiis = [RandomDataIterator(batch_size, shape=input_shape)
            for k in range(2)]
    eii_args = [SliceArgsIterator(batch_size, len(input_shape), image_shape=input_shape,
                image_layout=layout, dims=dims, dim_names=dim_names)
                for k in range(2)]

    compare_pipelines(
        SliceSynthDataPipeline(device, batch_size, layout, iter(eiis[0]), iter(eii_args[0]),
            dims=dims, dim_names=dim_names),
        SliceSynthDataPipelinePythonOp(batch_size, layout, iter(eiis[0]), iter(eii_args[1]),
            dims=dims, dim_names=dim_names),
        batch_size=batch_size, N_iterations=5)

def test_slice_synth_data_cpu_vs_gpu():
    for device in ["cpu", "gpu"]:
        for batch_size in {1, 8}:
            for input_shape, layout, dims, dim_names in \
                [((200,400,3), "HWC", None, "WH"),
                ((200,400,3), "HWC", None, "HW"),
                ((200,400,3), "HWC", (1,0), None),
                ((200,400,3), "HWC", (0,1), None),]:
                #((80, 30, 20, 3), "DHWC", (2,1,0), None),]:
                yield check_slice_synth_data_cpu_vs_gpu, device, batch_size, \
                    input_shape, layout, dims, dim_names
