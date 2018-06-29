# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
import nvidia.dali.tfrecord as tfrec
import numpy as np
from timeit import default_timer as timer
import numpy as np

caffe_db_folder = "/data/imagenet/train-lmdb-256x256"

def test_tensor_multiple_uses():
    batch_size = 128
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
            self.dump_cpu = ops.DumpImage(device = "cpu", suffix = "cpu")
            self.dump_gpu = ops.DumpImage(device = "gpu", suffix = "gpu")

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images_cpu = self.dump_cpu(images)
            images_gpu = self.dump_gpu(images.gpu())
            return (images, images_cpu, images_gpu)

        def iter_setup(self):
            pass

    pipe = HybridPipe(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
    pipe.build()
    out = pipe.run()
    assert(out[0].is_dense_tensor())
    assert(out[1].is_dense_tensor())
    assert(out[2].is_dense_tensor())
    assert(out[0].as_tensor().shape() == out[1].as_tensor().shape())
    assert(out[0].as_tensor().shape() == out[2].as_tensor().shape())
    a_raw = out[0]
    a_cpu = out[1]
    a_gpu = out[2].asCPU()
    for i in range(batch_size):
        t_raw = a_raw.at(i)
        t_cpu = a_cpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_raw)) == 0)
        t_cpu = a_cpu.at(i)
        t_gpu = a_gpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_gpu)) == 0)

def test_cropmirrornormalize_layout():
    batch_size = 128
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
            self.cmnp_nhwc = ops.CropMirrorNormalize(device = "gpu",
                                                     output_dtype = types.FLOAT,
                                                     output_layout = types.NHWC,
                                                     crop = (224, 224),
                                                     image_type = types.RGB,
                                                     mean = [128., 128., 128.],
                                                     std = [1., 1., 1.])
            self.cmnp_nchw = ops.CropMirrorNormalize(device = "gpu",
                                                     output_dtype = types.FLOAT,
                                                     output_layout = types.NCHW,
                                                     crop = (224, 224),
                                                     image_type = types.RGB,
                                                     mean = [128., 128., 128.],
                                                     std = [1., 1., 1.])

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            output_nhwc = self.cmnp_nhwc(images.gpu())
            output_nchw = self.cmnp_nchw(images.gpu())
            return (output_nchw, output_nhwc)

        def iter_setup(self):
            pass

    pipe = HybridPipe(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
    pipe.build()
    out = pipe.run()
    assert(out[0].is_dense_tensor())
    assert(out[1].is_dense_tensor())
    shape_nchw = out[0].as_tensor().shape()
    shape_nhwc = out[1].as_tensor().shape()
    assert(shape_nchw[0] == shape_nhwc[0])
    a_nchw = out[0].asCPU()
    a_nhwc = out[1].asCPU()
    for i in range(batch_size):
        t_nchw = a_nchw.at(i)
        t_nhwc = a_nhwc.at(i)
        assert(t_nchw.shape == (3,224,224))
        assert(t_nhwc.shape == (224,224,3))
        assert(np.sum(np.abs(np.transpose(t_nchw, (1,2,0)) - t_nhwc)) == 0)

def test_cropmirrornormalize_pad():
    batch_size = 128
    class HybridPipe(Pipeline):
        def __init__(self, layout, batch_size, num_threads, device_id, num_gpus):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
            self.cmnp_pad  = ops.CropMirrorNormalize(device = "gpu",
                                                     output_dtype = types.FLOAT,
                                                     output_layout = layout,
                                                     crop = (224, 224),
                                                     image_type = types.RGB,
                                                     mean = [128., 128., 128.],
                                                     std = [1., 1., 1.],
                                                     pad_output = True)
            self.cmnp      = ops.CropMirrorNormalize(device = "gpu",
                                                     output_dtype = types.FLOAT,
                                                     output_layout = layout,
                                                     crop = (224, 224),
                                                     image_type = types.RGB,
                                                     mean = [128., 128., 128.],
                                                     std = [1., 1., 1.],
                                                     pad_output = False)

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            output_pad = self.cmnp_pad(images.gpu())
            output = self.cmnp(images.gpu())
            return (output, output_pad)

        def iter_setup(self):
            pass

    for layout in [types.NCHW, types.NHWC]:
        pipe = HybridPipe(layout, batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
        pipe.build()
        out = pipe.run()
        assert(out[0].is_dense_tensor())
        assert(out[1].is_dense_tensor())
        shape     = out[0].as_tensor().shape()
        shape_pad = out[1].as_tensor().shape()
        assert(shape[0] == shape_pad[0])
        a = out[0].asCPU()
        a_pad = out[1].asCPU()
        for i in range(batch_size):
            t     = a.at(i)
            t_pad = a_pad.at(i)
            if (layout == types.NCHW):
                assert(t.shape == (3,224,224))
                assert(t_pad.shape == (4,224,224))
                assert(np.sum(np.abs(t - t_pad[:3,:,:])) == 0)
                assert(np.sum(np.abs(t_pad[3,:,:])) == 0)
            else:
                assert(t.shape == (224,224,3))
                assert(t_pad.shape == (224,224,4))
                assert(np.sum(np.abs(t - t_pad[:,:,:3])) == 0)
                assert(np.sum(np.abs(t_pad[:,:,3])) == 0)

def test_seed():
    batch_size = 64
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id,
                                             seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                                output_dtype = types.FLOAT,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [128., 128., 128.],
                                                std = [1., 1., 1.])
            self.coin = ops.CoinFlip()
            self.uniform = ops.Uniform(range = (0.0,1.0))
            self.iter = 0

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            mirror = self.coin()
            output = self.cmnp(images, mirror = mirror, crop_pos_x = self.uniform(), crop_pos_y = self.uniform())
            return (output, self.labels)

        def iter_setup(self):
            pass
    n = 30
    for i in range(50):
        pipe = HybridPipe(batch_size=batch_size,
                          num_threads=2,
                          device_id = 0)
        pipe.build()
        pipe_out = pipe.run()
        pipe_out_cpu = pipe_out[0].asCPU()
        img_chw_test = pipe_out_cpu.at(n)
        if i == 0:
            img_chw = img_chw_test
        assert(np.sum(np.abs(img_chw - img_chw_test)) == 0)

def test_seed_serialize():
    batch_size = 64
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id,
                                             seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                                output_dtype = types.FLOAT,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [128., 128., 128.],
                                                std = [1., 1., 1.])
            self.coin = ops.CoinFlip()
            self.uniform = ops.Uniform(range = (0.0,1.0))
            self.iter = 0

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            mirror = self.coin()
            output = self.cmnp(images, mirror = mirror, crop_pos_x = self.uniform(), crop_pos_y = self.uniform())
            return (output, self.labels)

        def iter_setup(self):
            pass
    n = 30
    orig_pipe = HybridPipe(batch_size=batch_size,
                           num_threads=2,
                           device_id = 0)
    s = orig_pipe.serialize()
    for i in range(50):
        pipe = Pipeline()
        pipe.deserialize_and_build(s)
        pipe_out = pipe.run()
        pipe_out_cpu = pipe_out[0].asCPU()
        img_chw_test = pipe_out_cpu.at(n)
        if i == 0:
            img_chw = img_chw_test
        assert(np.sum(np.abs(img_chw - img_chw_test)) == 0)

def test_rotate():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                                output_dtype = types.FLOAT,
                                                output_layout = types.NHWC,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [128., 128., 128.],
                                                std = [1., 1., 1.])
            self.rotate = ops.Rotate(device = "gpu", angle = 45.0,
                                     fill_value = 128,
                                     interp_type=types.INTERP_LINEAR)
            self.uniform = ops.Uniform(range = (0.0,1.0))
            self.iter = 0

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            outputs = self.cmnp([images, images],
                                crop_pos_x = self.uniform(),
                                crop_pos_y = self.uniform())
            outputs[1] = self.rotate(outputs[1])
            return [self.labels] + outputs

        def iter_setup(self):
            pass
    pipe = HybridPipe(batch_size=128, num_threads=2, device_id = 0)
    pipe.build()
    pipe_out = pipe.run()
    import cv2
    orig_cpu = pipe_out[1].asCPU()
    for i in range(128):
        orig = orig_cpu.at(i)
        M = cv2.getRotationMatrix2D((112,112),45, 1)
        out = cv2.warpAffine(orig, M, (224,224), borderMode=cv2.BORDER_REPLICATE, flags = (cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR))
        rotated_dali = pipe_out[2].asCPU().at(i)
        diff = out - rotated_dali
        diff[rotated_dali==[128.,128.,128.]] = 0
        assert(np.max(np.abs(diff)/255.0) < 0.025)

def test_warpaffine():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                                output_dtype = types.FLOAT,
                                                output_layout = types.NHWC,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [128., 128., 128.],
                                                std = [1., 1., 1.])
            self.affine = ops.WarpAffine(device = "gpu",
                                         matrix = [1.0, 0.8, 0.0, 0.0, 1.2, 0.0],
                                         fill_value = 128,
                                         interp_type = types.INTERP_LINEAR,
                                         use_image_center = True)
            self.uniform = ops.Uniform(range = (0.0,1.0))
            self.iter = 0

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            outputs = self.cmnp([images, images],
                                crop_pos_x = self.uniform(),
                                crop_pos_y = self.uniform())
            outputs[1] = self.affine(outputs[1])
            return [self.labels] + outputs

        def iter_setup(self):
            pass
    pipe = HybridPipe(batch_size=128, num_threads=2, device_id = 0)
    pipe.build()
    pipe_out = pipe.run()
    import cv2
    orig_cpu = pipe_out[1].asCPU()
    for i in range(128):
        orig = orig_cpu.at(i)
        M = np.array([1.0, 0.8, -0.8*112, 0.0, 1.2, -0.2*112]).reshape((2,3))
        out = cv2.warpAffine(orig, M, (224,224), borderMode=cv2.BORDER_REPLICATE, flags = (cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR))
        dali_output = pipe_out[2].asCPU().at(i)
        diff = out - dali_output
        diff[dali_output==[128.,128.,128.]] = 0
        assert(np.max(np.abs(diff)/255.0) < 0.025)
