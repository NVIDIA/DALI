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
from nvidia.dali.ops import _EdgeReference
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
from math import floor

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')
test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')

class SliceSynthDataPipeline(Pipeline):
    def __init__(self, device, batch_size, layout, iterator, pos_size_iter,
                 num_threads=1, device_id=0, num_gpus=1,
                 axes=None, axis_names=None, normalized_anchor=True, normalized_shape=True):
        super(SliceSynthDataPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=1234)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.pos_size_iter = pos_size_iter
        self.inputs = ops.ExternalSource()
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()

        if axis_names:
            self.slice = ops.Slice(device = self.device,
                                   normalized_anchor=normalized_anchor,
                                   normalized_shape=normalized_shape,
                                   axis_names = axis_names)
        elif axes:
            self.slice = ops.Slice(device = self.device,
                                   normalized_anchor=normalized_anchor,
                                   normalized_shape=normalized_shape,
                                   axes = axes)
        else:
            self.slice = ops.Slice(device = self.device,
                                   normalized_anchor=normalized_anchor,
                                   normalized_shape=normalized_shape,
)

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

class SlicePipeline(Pipeline):
    def __init__(self, device, batch_size, pos_size_iter,
                 num_threads=1, device_id=0, is_fused_decoder=False,
                 axes=None, axis_names=None, normalized_anchor=True, normalized_shape=True):
        super(SlicePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=1234)
        self.is_fused_decoder = is_fused_decoder
        self.pos_size_iter = pos_size_iter
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle=False)
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()

        if self.is_fused_decoder:
            if axis_names:
                self.decode = ops.ImageDecoderSlice(device = "cpu",
                                                    output_type = types.RGB,
                                                    normalized_anchor=normalized_anchor,
                                                    normalized_shape=normalized_shape,
                                                    axis_names = axis_names)
            elif axes:
                self.decode = ops.ImageDecoderSlice(device = "cpu",
                                                    output_type = types.RGB,
                                                    normalized_anchor=normalized_anchor,
                                                    normalized_shape=normalized_shape,
                                                    axes = axes)
            else:
                self.decode = ops.ImageDecoderSlice(device = "cpu",
                                                    output_type = types.RGB,
                                                    normalized_anchor=normalized_anchor,
                                                    normalized_shape=normalized_shape)
        else:
            self.decode = ops.ImageDecoder(device = "cpu",
                                           output_type = types.RGB)
            if axis_names:
                self.slice = ops.Slice(device = self.device,
                                       normalized_anchor=normalized_anchor,
                                       normalized_shape=normalized_shape,
                                       axis_names = axis_names)
            elif axes:
                self.slice = ops.Slice(device = self.device,
                                       normalized_anchor=normalized_anchor,
                                       normalized_shape=normalized_shape,
                                       axes = axes)
            else:
                self.slice = ops.Slice(device = self.device,
                                       normalized_anchor=normalized_anchor,
                                       normalized_shape=normalized_shape)

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
                 image_layout=None, # Needed if axis_names is used to specify the slice
                 normalized_anchor=True,
                 normalized_shape=True,
                 axes=None,
                 axis_names=None,
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
        self.axes = axes
        self.axis_names = axis_names
        self.min_norm_anchor=min_norm_anchor
        self.max_norm_anchor=max_norm_anchor
        self.min_norm_shape=min_norm_shape
        self.max_norm_shape=max_norm_shape
        self.seed=seed

        if not self.axis_names and not self.axes:
            self.axis_names = "WH"

        if self.axis_names:
            self.axes = []
            for axis_name in self.axis_names:
                assert axis_name in self.image_layout
                self.axes.append(self.image_layout.index(axis_name))
        assert(len(self.axes)>0)

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
            norm_anchor = anchor_amplitude * np.random.rand(len(self.axes)) + anchor_offset
            norm_shape = shape_amplitude * np.random.rand(len(self.axes)) + shape_offset

            if self.normalized_anchor:
                anchor = norm_anchor
            else:
                anchor = [floor(norm_anchor[i] * self.image_shape[self.axes[i]]) for i in range(len(self.axes))]

            if self.normalized_shape:
                shape = norm_shape
            else:
                shape = [floor(norm_shape[i] * self.image_shape[self.axes[i]]) for i in range(len(self.axes))]

            pos.append(np.asarray(anchor, dtype=np.float32))
            size.append(np.asarray(shape, dtype=np.float32))
            self.i = (self.i + 1) % self.n
        return (pos, size)
    next = __next__

def slice_func_helper(axes, axis_names, layout, normalized_anchor, normalized_shape, image, slice_anchor, slice_shape):
    # TODO(janton): remove this
    if not axes and not axis_names:
        axis_names = "WH"

    if axis_names:
        axes = []
        for axis_name in axis_names:
            assert(axis_name in layout)
            axis_pos = layout.find(axis_name)
            axes.append(axis_pos)

    shape = image.shape
    full_slice_anchor = [0] * len(shape)
    full_slice_shape = list(shape)
    for axis in axes:
        idx = axes.index(axis)
        full_slice_anchor[axis] = slice_anchor[idx]
        full_slice_shape[axis] = slice_shape[idx]

    if normalized_anchor and normalized_shape:
        start = [int(np.float32(shape[i]) * np.float32(full_slice_anchor[i]))
                 for i in range(len(shape))]
        end = [int(np.float32(shape[i]) * np.float32(full_slice_anchor[i]+full_slice_shape[i]))
               for i in range(len(shape))]
    else:
        if normalized_anchor:
            start = [int(np.float32(shape[i]) * np.float32(full_slice_anchor[i]))
                    for i in range(len(shape))]
        else:
            start = [int(np.float32(full_slice_anchor[i]))
                    for i in range(len(shape))]

        if normalized_shape:
            end = [start[i] + int(np.float32(shape[i]) * np.float32(full_slice_shape[i]))
                for i in range(len(shape))]
        else:
            end = [start[i] + int(np.float32(full_slice_shape[i]))
                for i in range(len(shape))]

    if len(full_slice_anchor) == 3:
        return image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    elif len(full_slice_anchor) == 4:
        return image[start[0]:end[0], start[1]:end[1], start[2]:end[2], start[3]:end[3]]
    else:
        assert(False)

class SliceSynthDataPipelinePythonOp(Pipeline):
    def __init__(self, batch_size, layout, iterator, pos_size_iter,
                 num_threads=1, device_id=0, num_gpus=1,
                 axes=None, axis_names=None,
                 normalized_anchor=True, normalized_shape=True):
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
            slice_func_helper, axes, axis_names, self.layout,
            normalized_anchor, normalized_shape)
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


class SlicePythonOp(Pipeline):
    def __init__(self, batch_size, pos_size_iter,
                 num_threads=1, device_id=0, num_gpus=1,
                 axes=None, axis_names=None,
                 normalized_anchor=True, normalized_shape=True):
        super(SlicePythonOp, self).__init__(
            batch_size, num_threads, device_id,
            seed=12345, exec_async=False, exec_pipelined=False)
        self.device = "cpu"
        self.layout = "HWC"
        self.pos_size_iter = pos_size_iter

        self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle=False)
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)

        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()

        function = partial(
            slice_func_helper, axes, axis_names, self.layout,
            normalized_anchor, normalized_shape)
        self.slice = ops.PythonFunction(function=function)

    def define_graph(self):
        imgs, _ = self.input()
        imgs = self.decode(imgs)
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()
        out = self.slice(imgs, self.crop_pos, self.crop_size)
        return out

    def iter_setup(self):
        (crop_pos, crop_size) = self.pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)


def check_slice_synth_data_vs_numpy(device, batch_size, input_shape, layout, axes, axis_names,
                                    normalized_anchor, normalized_shape):
    eiis = [RandomDataIterator(batch_size, shape=input_shape)
            for k in range(2)]
    eii_args = [SliceArgsIterator(batch_size, len(input_shape), image_shape=input_shape,
                image_layout=layout, axes=axes, axis_names=axis_names, normalized_anchor=normalized_anchor,
                normalized_shape=normalized_shape)
                for k in range(2)]

    compare_pipelines(
        SliceSynthDataPipeline(device, batch_size, layout, iter(eiis[0]), iter(eii_args[0]),
            axes=axes, axis_names=axis_names, normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape),
        SliceSynthDataPipelinePythonOp(batch_size, layout, iter(eiis[0]), iter(eii_args[1]),
            axes=axes, axis_names=axis_names, normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape),
        batch_size=batch_size, N_iterations=5)

def test_slice_synth_data_vs_numpy():
    for device in ["cpu", "gpu"]:
        for batch_size in {1, 8}:
            for input_shape, layout, axes, axis_names in \
                [((200,400,3), "HWC", None, "WH"),
                ((200,400,3), "HWC", None, "HW"),
                ((200,400,3), "HWC", None, "C"),
                ((200,400,3), "HWC", (1,0), None),
                ((200,400,3), "HWC", (0,1), None),
                ((200,400,3), "HWC", (2,), None),
                ((80, 30, 20, 3), "DHWC", (2,1,0), None),
                ((80, 30, 20, 3), "DHWC", (0,1,2), None),
                ((80, 30, 20, 3), "DHWC", (2,1), None),
                ((80, 30, 20, 3), "DHWC", None, "WHD"),
                ((80, 30, 20, 3), "DHWC", None, "DHW"),
                ((80, 30, 20, 3), "DHWC", None, "WH"),
                ((80, 30, 20, 3), "DHWC", None, "C")]:
                for normalized_anchor in [True, False]:
                    for normalized_shape in [True, False]:
                        yield check_slice_synth_data_vs_numpy, device, batch_size, \
                            input_shape, layout, axes, axis_names, normalized_anchor, normalized_shape

def check_slice_vs_fused_decoder(device, batch_size, axes, axis_names):
    eii_args = [SliceArgsIterator(batch_size, image_layout="HWC", axes=axes, axis_names=axis_names)
                for k in range(2)]
    compare_pipelines(
        SlicePipeline(device, batch_size, iter(eii_args[0]), axes=axes, axis_names=axis_names, is_fused_decoder=False),
        SlicePipeline(device, batch_size, iter(eii_args[1]), axes=axes, axis_names=axis_names, is_fused_decoder=True),
        batch_size=batch_size, N_iterations=5)

def test_slice_vs_fused_decoder():
    for device in ["cpu", "gpu"]:
        for batch_size in {1}:
            for axes, axis_names in \
                [(None, "WH"), (None, "HW"),
                ((1,0), None), ((0,1), None)]:
                yield check_slice_vs_fused_decoder, device, batch_size, axes, axis_names

def check_slice_vs_numpy(device, batch_size, axes, axis_names):
    eii_args = [SliceArgsIterator(batch_size, image_layout="HWC", axes=axes, axis_names=axis_names)
                for k in range(2)]
    compare_pipelines(
        SlicePipeline(device, batch_size, iter(eii_args[0]), axes=axes, axis_names=axis_names),
        SlicePythonOp(batch_size, iter(eii_args[1]), axes=axes, axis_names=axis_names),
        batch_size=batch_size, N_iterations=5)

def test_slice_vs_numpy():
    for device in ["cpu", "gpu"]:
        for batch_size in {1}:
            for axes, axis_names in \
                [(None, "WH"), (None, "HW"),
                ((1,0), None), ((0,1), None)]:
                yield check_slice_vs_numpy, device, batch_size, axes, axis_names
