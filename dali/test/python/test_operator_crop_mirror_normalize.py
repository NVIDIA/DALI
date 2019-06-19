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
from test_utils import RandomDataIterator

test_data_root = os.environ['DALI_EXTRA_PATH']
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')

class CropMirrorNormalizePipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1,
                 is_new_cmn = False, output_dtype = types.FLOAT, output_layout = types.NHWC,
                 mirror_probability = 0.0, mean=[0., 0., 0.], std=[1., 1., 1.], pad_output=False):
        super(CropMirrorNormalizePipeline, self).__init__(batch_size, num_threads, device_id, seed=7865)
        self.device = device
        self.is_new_cmn = is_new_cmn
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        if self.is_new_cmn:
            self.cmn = ops.NewCropMirrorNormalize(device = self.device,
                                                  output_dtype = output_dtype,
                                                  output_layout = output_layout,
                                                  crop = (224, 224),
                                                  crop_pos_x = 0.3,
                                                  crop_pos_y = 0.2,
                                                  image_type = types.RGB,
                                                  mean = mean,
                                                  std = std,
                                                  pad_output = pad_output)
        else:
            self.cmn = ops.CropMirrorNormalize(device = self.device,
                                               output_dtype = output_dtype,
                                               output_layout = output_layout,
                                               crop = (224, 224),
                                               crop_pos_x = 0.3,
                                               crop_pos_y = 0.2,
                                               image_type = types.RGB,
                                               mean = mean,
                                               std = std,
                                               pad_output = pad_output)
        self.coin = ops.CoinFlip(probability=mirror_probability, seed=7865)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        rng = self.coin()
        images = self.cmn(images, mirror=rng)
        return images

def check_cmn_cpu_vs_gpu(batch_size, output_dtype, output_layout, mirror_probability, mean, std, pad_output, is_new_cmn):
    iterations = 8 if batch_size == 1 else 1
    eps = 1e-07 if output_dtype != types.INT32 else 0.01
    compare_pipelines(CropMirrorNormalizePipeline('cpu', batch_size, output_dtype=output_dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, pad_output=pad_output,
                                                  is_new_cmn=is_new_cmn),
                      CropMirrorNormalizePipeline('gpu', batch_size, output_dtype=output_dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, pad_output=pad_output,
                                                  is_new_cmn=is_new_cmn),
                      batch_size=batch_size, N_iterations=iterations, eps=eps)

def test_cmn_cpu_vs_gpu():
    for batch_size in [1, 8]:
        for output_dtype in [types.FLOAT, types.INT32]:
            for output_layout in [types.NHWC, types.NCHW]:
                for mirror_probability in [0.0, 0.5, 1.0]:
                    norm_data = [ ([0., 0., 0.], [1., 1., 1.]),
                                  ([0.5 * 255], [0.225 * 255]),
                                  ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]) ] \
                                if output_dtype != types.INT32 else \
                                [ ([0., 0., 0.], [1., 1., 1.]),
                                  ([0.5 * 255], [0.225]),
                                  ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229, 0.224, 0.225]) ]
                    for (mean, std) in norm_data:
                        for pad_output in [False, True]:
                            for is_new_cmn in [False, True]:
                                yield check_cmn_cpu_vs_gpu, batch_size, output_dtype, output_layout, mirror_probability, mean, std, pad_output, True

def check_cmn_cpu_old_vs_new(device_new, device_old, batch_size, output_dtype, output_layout, mirror_probability, mean, std, pad_output):
    iterations = 8 if batch_size == 1 else 1
    eps = 1e-07 if output_dtype != types.INT32 else 0.75
    compare_pipelines(CropMirrorNormalizePipeline(device_old, batch_size, output_dtype=output_dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, pad_output=pad_output,
                                                  is_new_cmn=False),
                      CropMirrorNormalizePipeline(device_new, batch_size, output_dtype=output_dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, pad_output=pad_output,
                                                  is_new_cmn=True),
                      batch_size=batch_size, N_iterations=iterations, eps=eps)

def test_cmn_cpu_old_vs_new():
    for device_new in ['cpu', 'gpu']:
        for device_old in ['cpu', 'gpu']:
            for batch_size in [1, 8]:
                for output_dtype in [types.FLOAT, types.INT32]:
                    for output_layout in [types.NHWC, types.NCHW]:
                        for mirror_probability in [0.0, 0.5, 1.0]:
                            norm_data = [ ([0., 0., 0.], [1., 1., 1.]),
                                          ([0.5 * 255], [0.225 * 255]),
                                          ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]) ] \
                                        if output_dtype != types.INT32 else \
                                        [ ([0., 0., 0.], [1., 1., 1.]),
                                          ([0.5 * 255], [0.225]),
                                          ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229, 0.224, 0.225]) ]
                            for (mean, std) in norm_data:
                                for pad_output in [False, True] if device_old != 'cpu' else [False]: # padding doesn't work in the old CPU version
                                    yield check_cmn_cpu_old_vs_new, device_new, device_old, batch_size, output_dtype, \
                                        output_layout, mirror_probability, mean, std, pad_output


class NoCropPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, decoder_only=False):
        super(NoCropPipeline, self).__init__(batch_size, num_threads, device_id)
        self.decoder_only = decoder_only
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        if not self.decoder_only:
            self.cmn = ops.NewCropMirrorNormalize(device = self.device,
                                                  image_type = types.RGB,
                                                  output_dtype = types.UINT8,
                                                  output_layout = types.NHWC)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if not self.decoder_only:
            images = self.decode(inputs)
            if self.device == 'gpu':
                images = images.gpu()
            images = self.cmn(images)
        return images

def check_cmn_no_crop_args_vs_decoder_only(device, batch_size):
    compare_pipelines(NoCropPipeline(device, batch_size, decoder_only=True),
                      NoCropPipeline(device, batch_size, decoder_only=False),
                      batch_size=batch_size, N_iterations=10)

def test_cmn_no_crop_args_vs_decoder_only():
    for device in {'cpu'}:
        for batch_size in {1, 4}:
            yield check_cmn_no_crop_args_vs_decoder_only, device, batch_size

class PythonOpPipeline(Pipeline):
    def __init__(self, batch_size, function, num_threads=1, device_id=0, num_gpus=1):

        super(PythonOpPipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        self.cmn = ops.PythonFunction(function=function)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        images = self.cmn(images)
        return images

# Those are hardcoded coin flip results when using `seed=7865`
cmn_coin_flip_samples = {
  0.0 : [False],
  1.0 : [True],
  0.5 : [False, False, True, True, False, True, True]
}
cmn_idx = 0

def crop_mirror_normalize_func(crop_y, crop_x, crop_h, crop_w, mirror_probability, should_pad, mean, std,
                               input_layout, output_layout, image):
    assert input_layout == types.NHWC or input_layout == types.NFHWC
    if input_layout == types.NHWC:
        assert output_layout == types.NHWC or output_layout == types.NCHW
        assert len(image.shape) == 3
        F, H, W, C = 1, image.shape[0], image.shape[1], image.shape[2]
    elif input_layout == types.NFHWC:
        assert output_layout == types.NFHWC or output_layout == types.NFCHW
        assert len(image.shape) == 4
        F, H, W, C = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
    assert H >= crop_h and W >= crop_w

    start_y = int(np.round(np.float32(crop_y) * np.float32(H - crop_h)))
    end_y = start_y + crop_h
    start_x = int(np.round(np.float32(crop_x) * np.float32(W - crop_w)))
    end_x = start_x + crop_w

    # Crop
    if input_layout == types.NHWC:
        out = image[start_y:end_y, start_x:end_x, :]
        H, W = out.shape[0], out.shape[1]
    elif input_layout == types.NFHWC:
        out = image[:, start_y:end_y, start_x:end_x, :]
        H, W = out.shape[1], out.shape[2]

    if len(mean) == 1:
        mean = C * mean
        std = C * std
    assert len(mean) == C and len(std) == C
    inv_std = [np.float32(1.0) / np.float32(std[c]) for c in range(C)]
    mean = np.float32(mean)

    # Flip
    global cmn_coin_flip_samples, cmn_idx
    should_flip = cmn_coin_flip_samples[mirror_probability][cmn_idx]
    cmn_idx = (cmn_idx + 1) % len(cmn_coin_flip_samples[mirror_probability])

    dim_h = 2 if input_layout == types.NFHWC else 1
    out1 = np.flip(out, dim_h) if should_flip else out

    # Pad, normalize, transpose
    out_C = C + 1 if should_pad else C
    if input_layout == types.NHWC:
        out2 = np.zeros([H, W, out_C], dtype=np.float32)
        out2[:, :, 0:C] = (np.float32(out1) - mean) * inv_std
        return np.transpose(out2, (2, 0, 1)) if output_layout == types.NCHW else out2
    elif input_layout == types.NFHWC:
        out2 = np.zeros([F, H, W, out_C], dtype=np.float32)
        out2[:, :, :, 0:C] = (np.float32(out1) - mean) * inv_std
        return np.transpose(out2, (0, 3, 1, 2)) if output_layout == types.NFCHW else out2
    else:
        assert False

def check_cmn_vs_numpy(device, batch_size, output_dtype, output_layout,
                       mirror_probability, mean, std, should_pad, is_new_cmn):
    assert mirror_probability in cmn_coin_flip_samples
    global cmn_idx
    cmn_idx = 0

    crop_y, crop_x, crop_h, crop_w = (0.2, 0.3, 224, 224)
    function = partial(crop_mirror_normalize_func,
                       crop_y, crop_x, crop_h, crop_w, mirror_probability, should_pad,
                       mean, std, types.NHWC, output_layout)

    iterations = 8 if batch_size == 1 else 1
    compare_pipelines(CropMirrorNormalizePipeline(device, batch_size, output_dtype=output_dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, pad_output=should_pad, is_new_cmn=True),
                      PythonOpPipeline(batch_size, function),
                      batch_size=batch_size, N_iterations=iterations)

def test_cmn_python_op():
    norm_data = [ ([0., 0., 0.], [1., 1., 1.]),
                  ([0.5 * 255], [0.225 * 255]),
                  ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]) ]
    for device in ['cpu', 'gpu']:
        for batch_size in [1, 8]:
            for output_dtype in [types.FLOAT]:
                for output_layout in [types.NHWC, types.NCHW]:
                    mirror_probs = [0.0, 0.5, 1.0] if batch_size > 1 else [0.0, 1.0]
                    for mirror_probability in mirror_probs:
                        for (mean, std) in norm_data:
                            for should_pad in [False, True]:
                                yield check_cmn_vs_numpy, device, batch_size, output_dtype, output_layout, mirror_probability, \
                                    mean, std, should_pad, True


class CMNRandomDataPipeline(Pipeline):
    def __init__(self, device, batch_size, layout, iterator, num_threads=1, device_id=0, num_gpus=1,
                 output_dtype = types.FLOAT, output_layout = types.NFHWC,
                 mirror_probability = 0.0, mean=[0., 0., 0.], std=[1., 1., 1.], pad_output=False):
        super(CMNRandomDataPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.cmn = ops.NewCropMirrorNormalize(device = self.device,
                                              output_dtype = output_dtype,
                                              output_layout = output_layout,
                                              crop = (224, 224),
                                              crop_pos_x = 0.3,
                                              crop_pos_y = 0.2,
                                              image_type = types.RGB,
                                              mean = mean,
                                              std = std,
                                              pad_output = pad_output)
        self.coin = ops.CoinFlip(probability=mirror_probability, seed=7865)

    def define_graph(self):
        self.data = self.inputs()
        random_data = self.data.gpu() if self.device == 'gpu' else self.data
        rng = self.coin()
        out = self.cmn(random_data, mirror=rng)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

class CMNRandomDataPythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(CMNRandomDataPythonOpPipeline, self).__init__(batch_size,
                                                          num_threads,
                                                          device_id,
                                                          exec_async=False,
                                                          exec_pipelined=False)
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.cmn = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.cmn(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

def check_cmn_random_data_vs_numpy(device, batch_size, output_dtype, input_layout, input_shape,
                                   output_layout, mirror_probability, mean, std, should_pad):
    crop_y, crop_x, crop_h, crop_w = (0.2, 0.3, 224, 224)
    eii1 = RandomDataIterator(batch_size, shape=input_shape)
    eii2 = RandomDataIterator(batch_size, shape=input_shape)

    assert mirror_probability in cmn_coin_flip_samples
    global cmn_idx
    cmn_idx = 0

    function = partial(crop_mirror_normalize_func,
                       crop_y, crop_x, crop_h, crop_w, mirror_probability, should_pad,
                       mean, std, input_layout, output_layout)

    compare_pipelines(CMNRandomDataPipeline(device, batch_size, input_layout, iter(eii1),
                                          output_dtype = output_dtype, output_layout = output_layout,
                                          mirror_probability = mirror_probability, mean = mean, std= std,
                                          pad_output = should_pad),
                      CMNRandomDataPythonOpPipeline(function, batch_size, input_layout, iter(eii2)),
                      batch_size=batch_size, N_iterations=1)

def test_cmn_random_data_vs_numpy():
    norm_data = [ ([0., 0., 0.], [1., 1., 1.]),
                  ([0.5 * 255], [0.225 * 255]),
                  ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]) ]
    output_layouts = {
        types.NHWC : [types.NHWC, types.NCHW],
        types.NFHWC : [types.NFHWC, types.NFCHW]
    }

    input_shapes = {
        types.NHWC : [(600, 800, 3)],
        types.NFHWC : [(5, 600, 800, 3)],
    }

    for device in ['cpu', 'gpu']:
        for batch_size in [1, 8]:
            for output_dtype in [types.FLOAT]:
                for input_layout in [types.NHWC, types.NFHWC]:
                    for input_shape in input_shapes[input_layout]:
                        if input_layout == types.NFHWC:
                            assert len(input_shape) == 4
                        elif input_layout == types.NHWC:
                            assert len(input_shape) == 3
                        for output_layout in output_layouts[input_layout]:
                            mirror_probs = [0.0, 0.5, 1.0] if batch_size > 1 else [0.0, 1.0]
                            for mirror_probability in mirror_probs:
                                for (mean, std) in norm_data:
                                    for should_pad in [False, True]:
                                        yield check_cmn_random_data_vs_numpy, device, batch_size, output_dtype, input_layout, input_shape, \
                                            output_layout, mirror_probability, mean, std, should_pad
