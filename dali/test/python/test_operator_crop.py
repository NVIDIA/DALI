# Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os
from nose.tools import assert_raises
from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import get_dali_extra_path
from test_operator_slice import check_slice_output, abs_slice_start_and_end


test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')
test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')

class CropPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, crop_shape=(224, 224), crop_x = 0.3, crop_y = 0.2,
                 is_fused_decoder=False,):
        super(CropPipeline, self).__init__(batch_size, num_threads, device_id)
        self.is_fused_decoder = is_fused_decoder
        self.device = device
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)

        if self.is_fused_decoder:
            self.decode = ops.decoders.ImageCrop(device = "cpu",
                                                 crop = crop_shape,
                                                 crop_pos_x = crop_x,
                                                 crop_pos_y = crop_y,
                                                 output_type = types.RGB)
        else:
            self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
            self.crop = ops.Crop(device = self.device,
                                 crop = crop_shape,
                                 crop_pos_x = crop_x,
                                 crop_pos_y = crop_y)

    def define_graph(self):
        inputs, _ = self.input(name="Reader")

        if self.is_fused_decoder:
            images = self.decode(inputs)
            return images
        else:
            images = self.decode(inputs)
            if self.device == 'gpu':
                images = images.gpu()
            out = self.crop(images)
            return out

def check_crop_vs_fused_decoder(device, batch_size):
    compare_pipelines(CropPipeline(device, batch_size, is_fused_decoder=True),
                      CropPipeline(device, batch_size, is_fused_decoder=False),
                      batch_size=batch_size, N_iterations=3)

def test_crop_vs_fused_decoder():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 32}:
            yield check_crop_vs_fused_decoder, device, batch_size

def check_crop_cpu_vs_gpu(batch_size):
    compare_pipelines(CropPipeline('cpu', batch_size),
                      CropPipeline('gpu', batch_size),
                      batch_size=batch_size, N_iterations=3)

def test_crop_cpu_vs_gpu():
    for batch_size in {1, 32}:
        yield check_crop_cpu_vs_gpu, batch_size

class CropSequencePipeline(Pipeline):
    def __init__(self, device, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(CropSequencePipeline, self).__init__(batch_size,
                                                   num_threads,
                                                   device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.crop = ops.Crop(device = self.device,
                             crop = (224, 224),
                             crop_pos_x = 0.3,
                             crop_pos_y = 0.2)

    def define_graph(self):
        self.data = self.inputs()
        sequence = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.crop(sequence)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

class CropSequencePythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(CropSequencePythonOpPipeline, self).__init__(batch_size,
                                                           num_threads,
                                                           device_id,
                                                           exec_async=False,
                                                           exec_pipelined=False)
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.crop = ops.PythonFunction(function = function, output_layouts=layout)

    def define_graph(self):
        self.data = self.inputs()
        out = self.crop(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

def crop_func_help(image, layout, crop_y = 0.2, crop_x = 0.3, crop_h = 224, crop_w = 224):
    if layout == "FHWC":
        assert len(image.shape) == 4
        H = image.shape[1]
        W = image.shape[2]
    elif layout == "HWC":
        assert len(image.shape) == 3
        H = image.shape[0]
        W = image.shape[1]

    assert H >= crop_h
    assert W >= crop_w

    start_y = int(np.float32(crop_y) * np.float32(H - crop_h) + np.float32(0.5))
    end_y = start_y + crop_h
    start_x = int(np.float32(crop_x) * np.float32(W - crop_w) + np.float32(0.5))
    end_x = start_x + crop_w

    if layout == "FHWC":
        return image[:, start_y:end_y, start_x:end_x, :]
    elif layout == "HWC":
        return image[start_y:end_y, start_x:end_x, :]
    else:
        assert(False)  # should not happen

def crop_NFHWC_func(image):
    return crop_func_help(image, "FHWC")

def crop_NHWC_func(image):
    return crop_func_help(image, "HWC")

def check_crop_NFHWC_vs_python_op_crop(device, batch_size):
    eii1 = RandomDataIterator(batch_size, shape=(10, 300, 400, 3))
    eii2 = RandomDataIterator(batch_size, shape=(10, 300, 400, 3))
    compare_pipelines(CropSequencePipeline(device, batch_size, "FHWC", iter(eii1)),
                      CropSequencePythonOpPipeline(crop_NFHWC_func, batch_size, "FHWC", iter(eii2)),
                      batch_size=batch_size, N_iterations=3)

def test_crop_NFHWC_vs_python_op_crop():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 4}:
            yield check_crop_NFHWC_vs_python_op_crop, device, batch_size

def check_crop_NHWC_vs_python_op_crop(device, batch_size):
    eii1 = RandomDataIterator(batch_size, shape=(300, 400, 3))
    eii2 = RandomDataIterator(batch_size, shape=(300, 400, 3))
    compare_pipelines(CropSequencePipeline(device, batch_size, "HWC", iter(eii1)),
                      CropSequencePythonOpPipeline(crop_NHWC_func, batch_size, "HWC", iter(eii2)),
                      batch_size=batch_size, N_iterations=3)

def test_crop_NHWC_vs_python_op_crop():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 4}:
            yield check_crop_NHWC_vs_python_op_crop, device, batch_size

class CropCastPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, should_perform_cast=False):
        super(CropCastPipeline, self).__init__(batch_size,
                                               num_threads,
                                               device_id)
        self.should_perform_cast = should_perform_cast
        self.device = device
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu",
                                         output_type = types.RGB)

        if self.should_perform_cast:
            self.crop = ops.Crop(device = self.device,
                                crop = (224, 224),
                                crop_pos_x = 0.3,
                                crop_pos_y = 0.2,
                                dtype = types.FLOAT)
            self.crop2 = ops.Crop(device = self.device,
                                  crop = (224, 224),
                                  crop_pos_x = 0.0,
                                  crop_pos_y = 0.0,
                                  dtype = types.UINT8)
        else:
            self.crop = ops.Crop(device = self.device,
                    crop = (224, 224),
                    crop_pos_x = 0.3,
                    crop_pos_y = 0.2)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        if self.should_perform_cast:
            images_float = self.crop(images)
            images = self.crop2(images_float)
        else:
            images = self.crop(images)
        return images

def check_crop_no_cast_vs_cast_to_float_and_back(device, batch_size):
    compare_pipelines(CropCastPipeline(device, batch_size, should_perform_cast=False),
                      CropCastPipeline(device, batch_size, should_perform_cast=True),
                      batch_size=batch_size, N_iterations=3)

def test_crop_no_cast_vs_cast_to_float_and_back():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 4}:
            yield check_crop_no_cast_vs_cast_to_float_and_back, device, batch_size

class Crop3dPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, data_shape, data_layout, num_threads=1, device_id=0, crop_seq_as_depth=False):
        super(Crop3dPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_shape = data_shape
        self.data_layout = data_layout

        if self.data_layout == "DHWC":
            D, H, W, _ = self.data_shape
        elif self.data_layout == "CDHW":
            _, D, H, W = self.data_shape
        elif self.data_layout == "FHWC" and crop_seq_as_depth:
            D, H, W, _ = self.data_shape
        elif self.data_layout == "FCHW" and crop_seq_as_depth:
            D, _, H, W = self.data_shape
        else:
            assert(False)

        self.crop = ops.Crop(device = self.device,
                             crop_pos_z = 0.1,
                             crop_pos_y = 0.2,
                             crop_pos_x = 0.3,
                             crop_d = D * 0.91,
                             crop_h = H * 0.85,
                             crop_w = W * 0.75)

    def define_graph(self):
        self.data = self.inputs()
        sequence = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.crop(sequence)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)

class Crop3dPythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, iterator, data_shape, data_layout, num_threads=1, device_id=0):
        super(Crop3dPythonOpPipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     exec_async=False,
                                                     exec_pipelined=False)
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_shape = data_shape
        self.data_layout = data_layout

        def crop_func(image):
            return function(image, layout=self.data_layout, shape=self.data_shape)

        self.crop = ops.PythonFunction(function=crop_func, output_layouts=data_layout)

    def define_graph(self):
        self.data = self.inputs()
        out = self.crop(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)

def crop_3d_func(image, layout, shape, crop_anchor=(0.1, 0.2, 0.3), crop_shape=(0.91, 0.85, 0.75)):
    assert len(image.shape) == 4
    assert len(crop_anchor) == 3
    assert len(crop_shape) == 3

    if layout == "DHWC":
        D, H, W = image.shape[0], image.shape[1], image.shape[2]
    elif layout == "CDHW":
        D, H, W = image.shape[1], image.shape[2], image.shape[3]
    else:
        assert(False)

    crop_d, crop_h, crop_w = int(crop_shape[0]*D), int(crop_shape[1]*H), int(crop_shape[2]*W),
    assert D >= crop_d
    assert H >= crop_h
    assert W >= crop_w
    crop_z, crop_y, crop_x = crop_anchor[0], crop_anchor[1], crop_anchor[2]

    start_z = int(np.float32(0.5) + np.float32(crop_z) * np.float32(D - crop_d))
    end_z = start_z + crop_d
    start_y = int(np.float32(0.5) + np.float32(crop_y) * np.float32(H - crop_h))
    end_y = start_y + crop_h
    start_x = int(np.float32(0.5) + np.float32(crop_x) * np.float32(W - crop_w))
    end_x = start_x + crop_w

    if layout == "DHWC":
        return image[start_z:end_z, start_y:end_y, start_x:end_x, :]
    elif layout == "CDHW":
        return image[:, start_z:end_z, start_y:end_y, start_x:end_x]
    else:
        assert(False)

def check_crop_3d_vs_python_op_crop(device, batch_size, layout, shape):
    eii1 = RandomDataIterator(batch_size, shape=shape)
    eii2 = RandomDataIterator(batch_size, shape=shape)
    compare_pipelines(Crop3dPipeline(device, batch_size, iter(eii1), data_shape=shape, data_layout=layout),
                      Crop3dPythonOpPipeline(crop_3d_func, batch_size, iter(eii2), data_shape=shape, data_layout=layout),
                      batch_size=batch_size, N_iterations=3)

def test_crop_3d_vs_python_op_crop():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 4}:
            for layout, shape in {("DHWC", (300, 100, 10, 3)),
                                  ("DHWC", (100, 300, 10, 1)),
                                  ("DHWC", (10, 30, 300, 1)),
                                  ("DHWC", (20, 50, 60, 8)),
                                  ("CDHW", (3, 300, 100, 10)),
                                  ("CDHW", (3, 300, 10, 100)),
                                  ("CDHW", (8, 30, 10, 50))}:
               yield check_crop_3d_vs_python_op_crop, device, batch_size, layout, shape


def check_crop_sequence_length(device, batch_size, dtype, input_layout, input_shape):
    crop_z, crop_y, crop_x = (0.1, 0.2, 0.3)

    if input_layout == "FHWC":
        D, H, W, C = input_shape
    elif input_layout == "FCHW":
        D, C, H, W = input_shape
    else:
        assert(False)

    crop_d = int(D * 0.91)
    crop_h = int(H * 0.85)
    crop_w = int(W * 0.75)

    if input_layout == "FHWC":
        crop_shape = (crop_d, crop_h, crop_w, C)
    elif input_layout == "FCHW":
        crop_shape = (crop_d, C, crop_h, crop_w)
    else:
        assert(False)

    eii1 = RandomDataIterator(batch_size, shape=input_shape)

    pipe = Crop3dPipeline(device, batch_size, iter(eii1),
                          data_shape=input_shape, data_layout=input_layout, crop_seq_as_depth=True)
    pipe.build()
    out = pipe.run()
    out_data = out[0]

    for i in range(batch_size):
        assert(out_data.at(i).shape == crop_shape), \
            "Shape mismatch {} != {}".format(out_data.at(i).shape, crop_shape)

# Tests cropping along the sequence dimension as if it was depth
def test_cmn_crop_sequence_length():
    input_configs = {("FHWC", (10, 60, 80, 3)),
                     ("FCHW", (10, 3, 60, 80))}
    for device in ['cpu']:
        for batch_size in [8]:
            for dtype in [types.FLOAT]:
                for input_layout, input_shape in input_configs:
                    assert len(input_layout) == len(input_shape)
                    yield check_crop_sequence_length, device, batch_size, dtype, \
                        input_layout, input_shape

class CropSynthPipe(Pipeline):
    def __init__(self, device, batch_size, data_iterator, num_threads=1, device_id=0, num_gpus=1, crop_shape=(224, 224), crop_x=0.3, crop_y=0.2,
                 extra_outputs=False, out_of_bounds_policy=None, fill_values=None, layout="HWC"):
        super(CropSynthPipe, self).__init__(
            batch_size, num_threads, device_id)
        self.device = device
        self.extra_outputs = extra_outputs
        self.inputs = ops.ExternalSource()
        self.data_iterator = data_iterator
        self.layout = layout

        self.crop = ops.Crop(device = self.device,
                                crop = crop_shape,
                                crop_pos_x = crop_x,
                                crop_pos_y = crop_y,
                                out_of_bounds_policy = out_of_bounds_policy,
                                fill_values = fill_values)

    def define_graph(self):
        self.data = self.inputs()
        images = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.crop(images)
        if self.extra_outputs:
            return out, images
        else:
            return out

    def iter_setup(self):
        data = self.data_iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def check_crop_with_out_of_bounds_policy_support(device, batch_size, input_shape=(100, 200, 3),
                                                 out_of_bounds_policy=None, fill_values=(0x76, 0xb9, 0x00)):
    # This test case is written with HWC layout in mind and "HW" axes in slice arguments
    layout = "HWC"
    assert(len(input_shape) == 3)
    if fill_values is not None and len(fill_values) > 1:
        assert(input_shape[2] == len(fill_values))

    eii = RandomDataIterator(batch_size, shape=input_shape)
    crop_shape = tuple(extent * 2 for extent in input_shape[:2])
    crop_y = 0.5
    crop_x = 0.5
    pipe = CropSynthPipe(device, batch_size, iter(eii),
                         layout = layout,
                         crop_shape = crop_shape,
                         crop_x = crop_y, crop_y = crop_x,
                         out_of_bounds_policy=out_of_bounds_policy,
                         fill_values=fill_values,
                         extra_outputs=True)
    if fill_values is None:
        fill_values = 0
    pipe.build()
    for k in range(3):
        outs = pipe.run()
        out = outs[0]
        in_data = outs[1]
        if isinstance(out, dali.backend_impl.TensorListGPU):
            out = out.as_cpu()
        if isinstance(in_data, dali.backend_impl.TensorListGPU):
            in_data = in_data.as_cpu()

        assert(batch_size == len(out))
        for idx in range(batch_size):
            sample_in = in_data.at(idx)
            sample_out = out.at(idx)
            in_shape = list(sample_in.shape)
            out_shape = list(sample_out.shape)
            crop_anchor_norm = [crop_y, crop_x]
            crop_anchor_abs = [crop_anchor_norm[k] * (input_shape[k] - crop_shape[k]) for k in range(2)]
            abs_start, abs_end, abs_slice_shape = abs_slice_start_and_end(in_shape[:2], crop_anchor_abs, crop_shape, False, False)
            check_slice_output(sample_in, sample_out, crop_anchor_abs, abs_slice_shape, abs_start, abs_end, out_of_bounds_policy, fill_values)

def test_crop_with_out_of_bounds_policy_support():
    in_shape = (40, 80, 3)
    for out_of_bounds_policy in ['pad', 'trim_to_shape']:
        for device in ['gpu', 'cpu']:
            for batch_size in [1, 3]:
                for fill_values in [None, (0x76, 0xb0, 0x00)]:
                    yield check_crop_with_out_of_bounds_policy_support, \
                        device, batch_size, in_shape, out_of_bounds_policy, fill_values

def check_crop_with_out_of_bounds_error(device, batch_size, input_shape=(100, 200, 3)):
    # This test case is written with HWC layout in mind and "HW" axes in slice arguments
    layout = "HWC"
    assert(len(input_shape) == 3)

    eii = RandomDataIterator(batch_size, shape=input_shape)
    crop_shape = tuple(extent * 2 for extent in input_shape[:2])
    crop_y = 0.5
    crop_x = 0.5
    pipe = CropSynthPipe(device, batch_size, iter(eii),
                         layout = layout,
                         crop_shape = crop_shape,
                         crop_x = crop_x, crop_y = crop_y,
                         out_of_bounds_policy="error")

    pipe.build()
    with assert_raises(RuntimeError):
        outs = pipe.run()

def test_slice_with_out_of_bounds_error():
    in_shape = (40, 80, 3)
    for device in ['gpu', 'cpu']:
        for batch_size in [1, 3]:
            yield check_crop_with_out_of_bounds_error, \
                device, batch_size, in_shape
