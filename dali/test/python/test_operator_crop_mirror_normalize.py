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
import numpy as np
import os
from functools import partial
from nose_utils import raises
from test_utils import compare_pipelines, dali_type_to_np
from test_utils import RandomDataIterator
from test_utils import get_dali_extra_path
from test_operator_slice import check_slice_output, abs_slice_start_and_end


test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')

def next_power_of_two(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

class CropMirrorNormalizePipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1,
                 dtype = types.FLOAT, output_layout = "HWC",
                 mirror_probability = 0.0, mean=[0., 0., 0.], std=[1., 1., 1.], scale=None, shift=None, pad_output=False):
        super(CropMirrorNormalizePipeline, self).__init__(batch_size, num_threads, device_id, seed=7865)
        self.device = device
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        self.cmn = ops.CropMirrorNormalize(device = self.device,
                                           dtype = dtype,
                                           output_layout = output_layout,
                                           crop = (224, 224),
                                           crop_pos_x = 0.3,
                                           crop_pos_y = 0.2,
                                           mean = mean,
                                           std = std,
                                           scale = scale,
                                           shift = shift,
                                           pad_output = pad_output)
        self.coin = ops.random.CoinFlip(probability=mirror_probability, seed=7865)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        rng = self.coin()
        images = self.cmn(images, mirror=rng)
        return images

class NoCropPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, decoder_only=False):
        super(NoCropPipeline, self).__init__(batch_size, num_threads, device_id)
        self.decoder_only = decoder_only
        self.device = device
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        if not self.decoder_only:
            self.cast = ops.CropMirrorNormalize(device = self.device,
                                               dtype = types.FLOAT,
                                               output_layout = "HWC")
        else:
            self.cast = ops.Cast(device = self.device,
                                 dtype = types.FLOAT)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        images = self.cast(images)
        return images

def check_cmn_no_crop_args_vs_decoder_only(device, batch_size):
    compare_pipelines(NoCropPipeline(device, batch_size, decoder_only=True),
                      NoCropPipeline(device, batch_size, decoder_only=False),
                      batch_size=batch_size, N_iterations=3)

def test_cmn_no_crop_args_vs_decoder_only():
    for device in {'cpu'}:
        for batch_size in {1, 4}:
            yield check_cmn_no_crop_args_vs_decoder_only, device, batch_size

class PythonOpPipeline(Pipeline):
    def __init__(self, batch_size, function, output_layout, mirror_probability, num_threads=1, device_id=0, num_gpus=1):

        super(PythonOpPipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        self.cmn = ops.PythonFunction(function=function, output_layouts=output_layout)
        self.coin = ops.random.CoinFlip(probability=mirror_probability, seed=7865)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        images = self.cmn(images, self.coin())
        return images

def crop_mirror_normalize_func(crop_z, crop_y, crop_x,
                               crop_d, crop_h, crop_w,
                               should_pad, mean, std,
                               scale, shift,
                               input_layout, output_layout, dtype, image, should_flip):
    scale = scale or 1
    shift = shift or 0

    assert input_layout == "HWC" or input_layout == "FHWC" \
        or input_layout == "DHWC" or input_layout == "FDHWC"
    assert len(input_layout) == len(image.shape)

    assert input_layout.count('H') > 0
    dim_h = input_layout.find('H')
    H = image.shape[dim_h]

    assert input_layout.count('W') > 0
    dim_w = input_layout.find('W')
    W = image.shape[dim_w]

    assert input_layout.count('C') > 0
    dim_c = input_layout.find('C')
    C = image.shape[dim_c]

    D = 1
    if input_layout.count('D') > 0:
        dim_d = input_layout.find('D')
        D = image.shape[dim_d]
        assert D >= crop_d

    F = 1
    if input_layout.count('F') > 0:
        dim_f = input_layout.find('F')
        F = image.shape[dim_f]

    assert H >= crop_h and W >= crop_w

    start_y = int(np.float32(crop_y) * np.float32(H - crop_h) + np.float32(0.5))
    end_y = start_y + crop_h
    start_x = int(np.float32(crop_x) * np.float32(W - crop_w) + np.float32(0.5))
    end_x = start_x + crop_w
    if input_layout.count('D') > 0:
        assert D >= crop_d
        start_z = int(np.float32(crop_z) * np.float32(D - crop_d) + np.float32(0.5))
        end_z = start_z + crop_d

    # Crop
    if input_layout == "HWC":
        out = image[start_y:end_y, start_x:end_x, :]
        H, W = out.shape[0], out.shape[1]
    elif input_layout == "FHWC":
        out = image[:, start_y:end_y, start_x:end_x, :]
        H, W = out.shape[1], out.shape[2]
    elif input_layout == "DHWC":
        out = image[start_z:end_z, start_y:end_y, start_x:end_x, :]
        D, H, W = out.shape[0], out.shape[1], out.shape[2]
    elif input_layout == "FDHWC":
        out = image[:, start_z:end_z, start_y:end_y, start_x:end_x, :]
        D, H, W = out.shape[1], out.shape[2], out.shape[3]

    if not mean:
        mean = [0.0]
    if not std:
        std = [1.0]

    if len(mean) == 1:
        mean = C * mean
    if len(std) == 1:
        std = C * std

    assert len(mean) == C and len(std) == C
    inv_std = [np.float32(1.0) / np.float32(std[c]) for c in range(C)]
    mean = np.float32(mean)


    assert input_layout.count('W') > 0
    horizontal_dim = input_layout.find('W')
    out1 = np.flip(out, horizontal_dim) if should_flip else out

    # Pad, normalize, transpose

    out_C = next_power_of_two(C) if should_pad else C

    if input_layout == "HWC":
        out2 = np.zeros([H, W, out_C], dtype=np.float32)
        out2[:, :, 0:C] = (np.float32(out1) - mean) * inv_std * scale + shift
        ret = np.transpose(out2, (2, 0, 1)) if output_layout == "CHW" else out2
    elif input_layout == "FHWC":
        out2 = np.zeros([F, H, W, out_C], dtype=np.float32)
        out2[:, :, :, 0:C] = (np.float32(out1) - mean) * inv_std * scale + shift
        ret = np.transpose(out2, (0, 3, 1, 2)) if output_layout == "FCHW" else out2
    elif input_layout == "DHWC":
        out2 = np.zeros([D, H, W, out_C], dtype=np.float32)
        out2[:, :, :, 0:C] = (np.float32(out1) - mean) * inv_std * scale + shift
        ret = np.transpose(out2, (3, 0, 1, 2)) if output_layout == "CDHW" else out2
    elif input_layout == "FDHWC":
        out2 = np.zeros([F, D, H, W, out_C], dtype=np.float32)
        out2[:, :, :, :, 0:C] = (np.float32(out1) - mean) * inv_std * scale + shift
        ret = np.transpose(out2, (0, 4, 1, 2, 3)) if output_layout == "FCDHW" else out2
    else:
        raise RuntimeError("The test function received unsupported layout {}".format(input_layout))

    # clamp the result to output type's dynamic range
    if np.issubdtype(dtype, np.integer):
        lo = np.iinfo(dtype).min
        hi = np.iinfo(dtype).max
        ret = np.clip(ret, lo, hi)

    return ret

def check_cmn_vs_numpy(device, batch_size, dtype, output_layout,
                       mirror_probability, mean, std, scale, shift, should_pad):
    crop_z, crop_y, crop_x = (0.1, 0.2, 0.3)
    crop_d, crop_h, crop_w = (10, 224, 224)
    function = partial(crop_mirror_normalize_func,
                       crop_z, crop_y, crop_x,
                       crop_d, crop_h, crop_w,
                       should_pad,
                       mean, std, scale, shift,
                       "HWC", output_layout, dali_type_to_np(dtype))

    iterations = 8 if batch_size == 1 else 1
    eps, max_err = (1e-5, 1e-5) if dtype == types.FLOAT else (0.3, 0.6)
    compare_pipelines(CropMirrorNormalizePipeline(device, batch_size, dtype=dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, scale=scale, shift=shift, pad_output=should_pad),
                      PythonOpPipeline(batch_size, function, output_layout, mirror_probability),
                      batch_size=batch_size, N_iterations=iterations, eps=eps, max_allowed_error=max_err)


def test_cmn_vs_numpy():
    norm_data = [ ([0., 0., 0.], [1., 1., 1.]),
                  ([0.5 * 255], [0.225 * 255]),
                  ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]) ]

    type_scale_shift = [(types.FLOAT, None, None),
                        (types.FLOAT16, None, None),
                        (types.UINT8, 64, 128),
                        (types.INT8, 50, 5)]

    np.random.seed(12321)

    for device in ['cpu', 'gpu']:
        for batch_size in [1, 4]:
            for output_layout in ["HWC", "CHW"]:
                mirror_probs = [0.5] if batch_size > 1 else [0.0, 1.0]
                for mirror_probability in mirror_probs:
                    for should_pad in [False, True]:
                        mean, std = norm_data[np.random.randint(0, len(norm_data))]
                        dtype, default_scale, default_shift = type_scale_shift[np.random.randint(0, len(type_scale_shift))]
                        shift = default_shift if mean and mean[0] > 1 else None
                        scale = default_scale if std and std[0] > 1 else None
                        yield check_cmn_vs_numpy, device, batch_size, dtype, output_layout, mirror_probability, \
                            mean, std, scale, shift, should_pad

class CMNRandomDataPipeline(Pipeline):
    def __init__(self, device, batch_size, layout, iterator, num_threads=1, device_id=0, num_gpus=1,
                 dtype = types.FLOAT, output_layout = "FHWC",
                 mirror_probability = 0.0, mean=[0., 0., 0.], std=[1., 1., 1.], scale=None, shift=None,
                 pad_output=False, crop_seq_as_depth=False,
                 crop_d = 8, crop_h = 16, crop_w = 32, crop_pos_x = 0.3, crop_pos_y = 0.2, crop_pos_z = 0.1,
                 out_of_bounds_policy=None, fill_values=None, extra_outputs=False):
        super(CMNRandomDataPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.extra_outputs = extra_outputs

        if layout.count('D') <= 0 and not (crop_seq_as_depth and layout.count('F') > 0):
            crop_d = None
        self.cmn = ops.CropMirrorNormalize(device=self.device, dtype=dtype, output_layout=output_layout,
                                           crop_d=crop_d, crop_h=crop_h, crop_w=crop_w,
                                           crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y, crop_pos_z=crop_pos_z,
                                           mean=mean, std=std, pad_output=pad_output,
                                           scale=scale, shift=shift,
                                           out_of_bounds_policy=out_of_bounds_policy, fill_values=fill_values)
        self.coin = ops.random.CoinFlip(probability=mirror_probability, seed=7865)

    def define_graph(self):
        self.data = self.inputs()
        random_data = self.data.gpu() if self.device == 'gpu' else self.data
        rng = self.coin()
        out = self.cmn(random_data, mirror=rng)
        if self.extra_outputs:
            return out, random_data, rng
        else:
            return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

class CMNRandomDataPythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, layout, output_layout, mirror_probability, iterator, num_threads=1, device_id=0):
        super(CMNRandomDataPythonOpPipeline, self).__init__(batch_size,
                                                            num_threads,
                                                            device_id,
                                                            exec_async=False,
                                                            exec_pipelined=False)
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.cmn = ops.PythonFunction(function=function, output_layouts=output_layout)
        self.coin = ops.random.CoinFlip(probability=mirror_probability, seed=7865)

    def define_graph(self):
        self.data = self.inputs()
        out = self.cmn(self.data, self.coin())
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

def check_cmn_random_data_vs_numpy(device, batch_size, dtype, input_layout, input_shape,
                                   output_layout, mirror_probability, mean, std, scale, shift,
                                   should_pad):
    crop_z, crop_y, crop_x = (0.1, 0.2, 0.3)
    crop_d, crop_h, crop_w = (8, 16, 32)
    eii1 = RandomDataIterator(batch_size, shape=input_shape)
    eii2 = RandomDataIterator(batch_size, shape=input_shape)

    function = partial(crop_mirror_normalize_func,
                       crop_z, crop_y, crop_x,
                       crop_d, crop_h, crop_w,
                       should_pad,
                       mean, std, scale, shift, input_layout, output_layout,
                       dali_type_to_np(dtype))

    cmn_pipe = CMNRandomDataPipeline(device, batch_size, input_layout, iter(eii1),
                            dtype=dtype, output_layout=output_layout,
                            mirror_probability=mirror_probability,
                            mean=mean, std=std, scale=scale, shift=shift,
                            pad_output=should_pad)

    ref_pipe = CMNRandomDataPythonOpPipeline(function, batch_size, input_layout, output_layout,
                                             mirror_probability, iter(eii2))

    eps, max_err = (1e-5, 1e-5) if dtype == types.FLOAT else (0.3, 0.6)
    compare_pipelines(cmn_pipe, ref_pipe, batch_size, 2, eps=eps, max_allowed_error=max_err)


def test_cmn_random_data_vs_numpy():
    norm_data = [ ([0., 0., 0.], [1., 1., 1.]),
                  ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
                  ([0.485 * 255, 0.456 * 255, 0.406 * 255], None),
                  ([0.485 * 255, 0.456 * 255, 0.406 * 255], [255.0,]),
                  (None, [0.229 * 255, 0.224 * 255, 0.225 * 255]),
                  ([128,], [0.229 * 255, 0.224 * 255, 0.225 * 255]) ]
    output_layouts = {
        "HWC" : ["HWC", "CHW"],
        "FHWC" : ["FHWC", "FCHW"],
        "DHWC" : ["DHWC", "CDHW"],
        "FDHWC" :["FDHWC", "FCDHW"]
    }

    input_shapes = {
        "HWC" : [(60, 80, 3)],
        "FHWC" : [(3, 60, 80, 3)],
        "DHWC" : [(10, 60, 80, 3)],
        "FDHWC" : [(3, 10, 60, 80, 3)]
    }

    np.random.seed(12345)

    type_scale_shift = [(types.FLOAT, None, None),
                        (types.FLOAT16, None, None),
                        (types.UINT8, 64, 128),
                        (types.INT8, 50, 5)]


    for device in ['cpu', 'gpu']:
        for batch_size in [1, 4]:
            for input_layout in ["HWC", "FHWC", "DHWC", "FDHWC"]:
                for input_shape in input_shapes[input_layout]:
                    assert len(input_layout) == len(input_shape)
                    for output_layout in output_layouts[input_layout]:
                        mirror_probs = [0.5] if batch_size > 1 else [0.0, 1.0]
                        for mirror_probability in mirror_probs:
                            for should_pad in [False, True]:
                                mean, std = norm_data[np.random.randint(0, len(norm_data))]
                                dtype, default_scale, default_shift = type_scale_shift[np.random.randint(0, len(type_scale_shift))]
                                shift = default_shift if mean and mean[0] > 1 else None
                                scale = default_scale if std and std[0] > 1 else None
                                yield check_cmn_random_data_vs_numpy, device, batch_size, dtype, \
                                    input_layout, input_shape, output_layout, mirror_probability, \
                                    mean, std, scale, shift, should_pad


def check_cmn_crop_sequence_length(device, batch_size, dtype, input_layout, input_shape,
                                   output_layout, mirror_probability, mean, std, should_pad):
    crop_z, crop_y, crop_x = (0.1, 0.2, 0.3)
    crop_d, crop_h, crop_w = (8, 16, 32)
    eii1 = RandomDataIterator(batch_size, shape=input_shape)

    pipe = CMNRandomDataPipeline(device, batch_size, input_layout, iter(eii1),
                                 dtype = dtype, output_layout = output_layout,
                                 mirror_probability = mirror_probability, mean = mean, std= std,
                                 pad_output = should_pad, crop_seq_as_depth=True)
    pipe.build()
    out = pipe.run()
    out_data = out[0]

    expected_out_shape = (crop_d, 3, crop_h, crop_w) if output_layout == "FCHW" else (crop_d, crop_h, crop_w, 3)

    for i in range(batch_size):
        assert(out_data.at(i).shape == expected_out_shape), \
            "Shape mismatch {} != {}".format(out_data.at(i).shape, expected_out_shape)

# Tests cropping along the sequence dimension as if it was depth
def test_cmn_crop_sequence_length():
    input_layout = "FHWC"
    output_layouts = ["FHWC", "FCHW"]
    output_layouts = {
        "FHWC" : ["FHWC", "FCHW"],
    }

    input_shapes = {
        "FHWC" : [(10, 60, 80, 3)],
    }

    mean = [127, ]
    std = [127, ]
    should_pad = False
    mirror_probability = 0.5

    for device in ['cpu']:
        for batch_size in [8]:
            for dtype in [types.FLOAT]:
                for input_shape in input_shapes[input_layout]:
                    assert len(input_layout) == len(input_shape)
                    for output_layout in output_layouts[input_layout]:
                        yield check_cmn_crop_sequence_length, device, batch_size, dtype, \
                            input_layout, input_shape, output_layout, mirror_probability, \
                            mean, std, should_pad

def check_cmn_with_out_of_bounds_policy_support(device, batch_size, dtype, input_layout, input_shape, output_layout,
                                                mirror_probability, mean, std, should_pad,
                                                out_of_bounds_policy=None, fill_values=(0x76, 0xb9, 0x00)):
    # This test case is written with HWC layout in mind and "HW" axes in slice arguments
    assert(input_layout == "HWC")
    assert(len(input_shape) == 3)
    if fill_values is not None and len(fill_values) > 1:
        assert(input_shape[2] == len(fill_values))
    eii = RandomDataIterator(batch_size, shape=input_shape)
    crop_y, crop_x = 0.5, 0.5
    crop_h, crop_w = input_shape[0] * 2, input_shape[1] * 2
    pipe = CMNRandomDataPipeline(device, batch_size, input_layout, iter(eii),
                                 dtype = dtype, output_layout = output_layout,
                                 mirror_probability = mirror_probability, mean = mean, std = std, pad_output = should_pad,
                                 crop_w = crop_w, crop_h = crop_h,
                                 crop_pos_x = crop_x, crop_pos_y = crop_y,
                                 out_of_bounds_policy = out_of_bounds_policy, fill_values = fill_values, extra_outputs = True)
    permute = None
    if output_layout != input_layout:
        permute = []
        for d in range(len(input_layout)):
            perm_d = input_layout.find(output_layout[d])
            permute.append(perm_d)

    if fill_values is None:
        fill_values = 0
    pipe.build()
    for k in range(3):
        outs = pipe.run()
        out = outs[0]
        in_data = outs[1]
        mirror_data = outs[2]
        if isinstance(out, dali.backend_impl.TensorListGPU):
            out = out.as_cpu()
        if isinstance(in_data, dali.backend_impl.TensorListGPU):
            in_data = in_data.as_cpu()

        assert(batch_size == len(out))
        for idx in range(batch_size):
            sample_in = in_data.at(idx)
            sample_out = out.at(idx)
            mirror = mirror_data.at(idx)
            flip = [0, mirror]
            in_shape = list(sample_in.shape)
            out_shape = list(sample_out.shape)
            crop_anchor_norm = [crop_y, crop_x]
            crop_shape = [crop_h, crop_w]
            crop_anchor_abs = [crop_anchor_norm[k] * (input_shape[k] - crop_shape[k]) for k in range(2)]
            abs_start, abs_end, abs_slice_shape = abs_slice_start_and_end(in_shape[:2], crop_anchor_abs, crop_shape, False, False)
            check_slice_output(sample_in, sample_out, crop_anchor_abs, abs_slice_shape, abs_start, abs_end,
                               out_of_bounds_policy, fill_values, mean=mean, std=std, flip=flip, permute=permute)

def test_cmn_with_out_of_bounds_policy_support():
    in_shape = (40, 80, 3)
    in_layout = 'HWC'
    dtype = types.FLOAT
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    fill_values = (0x76, 0xb0, 0x00)
    for out_of_bounds_policy in ['pad', 'trim_to_shape']:
        for device in ['gpu', 'cpu']:
            for batch_size in [1, 3]:
                for out_layout in ['HWC', 'CHW']:
                    for mirror_probability in [0.5]:
                        for should_pad in [False, True]:
                            yield check_cmn_with_out_of_bounds_policy_support, \
                                device, batch_size, dtype, in_layout, in_shape, out_layout, mirror_probability, mean, std, should_pad, \
                                out_of_bounds_policy, fill_values

def check_cmn_with_out_of_bounds_error(device, batch_size, input_shape=(100, 200, 3)):
    # This test case is written with HWC layout in mind and "HW" axes in slice arguments
    layout = "HWC"
    assert(len(input_shape) == 3)
    eii = RandomDataIterator(batch_size, shape=input_shape)
    crop_y, crop_x = 0.5, 0.5
    crop_h, crop_w = input_shape[0] * 2, input_shape[1] * 2
    pipe = CMNRandomDataPipeline(device, batch_size, layout, iter(eii),
                                 dtype = types.FLOAT, output_layout = layout,
                                 mirror_probability = 0.5, mean = [127.], std = [127.], pad_output = True,
                                 crop_w = crop_w, crop_h = crop_h,
                                 crop_pos_x = crop_x, crop_pos_y = crop_y,
                                 out_of_bounds_policy="error")
    pipe.build()
    pipe.run()

def test_slice_with_out_of_bounds_error():
    in_shape = (40, 80, 3)
    for device in ['gpu', 'cpu']:
        for batch_size in [1, 3]:
            yield raises(RuntimeError, "Slice can't be placed out of bounds with current policy.")(
                check_cmn_with_out_of_bounds_error), device, batch_size, in_shape
