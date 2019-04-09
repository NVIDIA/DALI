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
from timeit import default_timer as timer
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

caffe_db_folder = "/data/imagenet/train-lmdb-256x256"

def check_batch(batch1, batch2, batch_size, eps = 0.0000001):
    for i in range(batch_size):
        err = np.mean( np.abs(batch1.at(i) - batch2.at(i)) )
        assert(err < eps)

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
    a_gpu = out[2].as_cpu()
    for i in range(batch_size):
        t_raw = a_raw.at(i)
        t_cpu = a_cpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_raw)) == 0)
        t_cpu = a_cpu.at(i)
        t_gpu = a_gpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_gpu)) == 0)

def test_pipeline_separated_exec_setup():
    batch_size = 128
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, prefetch_queue_depth):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id, prefetch_queue_depth = prefetch_queue_depth)
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

    pipe = HybridPipe(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1,
                      prefetch_queue_depth = {"cpu_size": 5, "gpu_size": 3})
    pipe.build()
    out = pipe.run()
    assert(out[0].is_dense_tensor())
    assert(out[1].is_dense_tensor())
    assert(out[2].is_dense_tensor())
    assert(out[0].as_tensor().shape() == out[1].as_tensor().shape())
    assert(out[0].as_tensor().shape() == out[2].as_tensor().shape())
    a_raw = out[0]
    a_cpu = out[1]
    a_gpu = out[2].as_cpu()
    for i in range(batch_size):
        t_raw = a_raw.at(i)
        t_cpu = a_cpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_raw)) == 0)
        t_cpu = a_cpu.at(i)
        t_gpu = a_gpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_gpu)) == 0)

def test_pipeline_simple_sync_no_prefetch():
    batch_size = 16
    n_iters = 12

    class HybridPipe(Pipeline):
        def __init__(self, batch_size):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads=1,
                                             device_id=0, prefetch_queue_depth=1,
                                             exec_async=False, exec_pipelined=False)
            self.input = ops.CaffeReader(path = caffe_db_folder)
            self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
            self.dump_gpu = ops.DumpImage(device = "gpu", suffix = "gpu")

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images_gpu = self.dump_gpu(images.gpu())
            return (images, images_gpu)

        def iter_setup(self):
            pass

    pipe = HybridPipe(batch_size=batch_size)
    pipe.build()
    for _ in range(n_iters):
        out = pipe.run()

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
    a_nchw = out[0].as_cpu()
    a_nhwc = out[1].as_cpu()
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
        a = out[0].as_cpu()
        a_pad = out[1].as_cpu()
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
        pipe_out_cpu = pipe_out[0].as_cpu()
        img_chw_test = pipe_out_cpu.at(n)
        if i == 0:
            img_chw = img_chw_test
        assert(np.sum(np.abs(img_chw - img_chw_test)) == 0)

def test_as_array():
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
        pipe_out_cpu = pipe_out[0].as_cpu()
        img_chw_test = pipe_out_cpu.as_array()
        if i == 0:
            img_chw = img_chw_test
        assert(img_chw_test.shape == (batch_size,3,224,224))
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
        pipe_out_cpu = pipe_out[0].as_cpu()
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
    orig_cpu = pipe_out[1].as_cpu()
    for i in range(128):
        orig = orig_cpu.at(i)
        M = cv2.getRotationMatrix2D((112,112),45, 1)
        out = cv2.warpAffine(orig, M, (224,224), borderMode=cv2.BORDER_REPLICATE, flags = (cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR))
        rotated_dali = pipe_out[2].as_cpu().at(i)
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
    orig_cpu = pipe_out[1].as_cpu()
    for i in range(128):
        orig = orig_cpu.at(i)
        M = np.array([1.0, 0.8, -0.8*112, 0.0, 1.2, -0.2*112]).reshape((2,3))
        out = cv2.warpAffine(orig, M, (224,224), borderMode=cv2.BORDER_REPLICATE, flags = (cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR))
        dali_output = pipe_out[2].as_cpu().at(i)
        diff = out - dali_output
        diff[dali_output==[128.,128.,128.]] = 0
        assert(np.max(np.abs(diff)/255.0) < 0.025)

def test_type_conversion():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.cmnp_all = ops.CropMirrorNormalize(device = "gpu",
                                                    output_dtype = types.FLOAT,
                                                    output_layout = types.NHWC,
                                                    crop = (224, 224),
                                                    image_type = types.RGB,
                                                    mean = [128., 128., 128.],
                                                    std = [1., 1., 1.])
            self.cmnp_int = ops.CropMirrorNormalize(device = "gpu",
                                                    output_dtype = types.FLOAT,
                                                    output_layout = types.NHWC,
                                                    crop = (224, 224),
                                                    image_type = types.RGB,
                                                    mean = [128, 128, 128],
                                                    std = [1., 1, 1])  # Left 1 of the arguments as float to test whether mixing types works
            self.cmnp_1arg = ops.CropMirrorNormalize(device = "gpu",
                                                     output_dtype = types.FLOAT,
                                                     output_layout = types.NHWC,
                                                     crop = (224, 224),
                                                     image_type = types.RGB,
                                                     mean = 128,
                                                     std = 1)
            self.uniform = ops.Uniform(range = (0,1))

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            outputs = [ None for i in range(3)]
            crop_pos_x = self.uniform()
            crop_pos_y = self.uniform()
            outputs[0] = self.cmnp_all(images,
                                       crop_pos_x = crop_pos_x,
                                       crop_pos_y = crop_pos_y)
            outputs[1] = self.cmnp_int(images,
                                       crop_pos_x = crop_pos_x,
                                       crop_pos_y = crop_pos_y)
            outputs[2] = self.cmnp_1arg(images,
                                        crop_pos_x = crop_pos_x,
                                        crop_pos_y = crop_pos_y)
            return [self.labels] + outputs

    pipe = HybridPipe(batch_size=128, num_threads=2, device_id = 0)
    pipe.build()
    for i in range(10):
        pipe_out = pipe.run()
        orig_cpu = pipe_out[1].as_cpu().as_tensor()
        int_cpu  = pipe_out[2].as_cpu().as_tensor()
        arg1_cpu = pipe_out[3].as_cpu().as_tensor()
        assert_array_equal(orig_cpu, int_cpu)
        assert_array_equal(orig_cpu, arg1_cpu)

def test_crop():
    class CMNvsCropPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(CMNvsCropPipe, self).__init__(batch_size, num_threads, device_id, seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = 1)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                                output_layout = types.NHWC,
                                                output_dtype = types.FLOAT,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [0., 0., 0.],
                                                std = [1., 1., 1.])
            self.crop = ops.Crop(device = "gpu",
                                 crop = (224, 224),
                                 image_type = types.RGB)
            self.uniform = ops.Uniform(range = (0.0, 1.0))
            self.cast = ops.Cast(device = "gpu",
                                 dtype = types.INT32)

        def define_graph(self):
            inputs, labels = self.input()
            images = self.decode(inputs)
            crop_x = self.uniform()
            crop_y = self.uniform()
            output_cmn = self.cmn(images, crop_pos_x = crop_x, crop_pos_y = crop_y)
            output_crop = self.crop(images, crop_pos_x = crop_x, crop_pos_y = crop_y)
            output_cmn = self.cast(output_cmn)
            output_crop = self.cast(output_crop)
            return (output_cmn, output_crop, labels.gpu())

    batch_size = 8
    iterations = 8

    pipe = CMNvsCropPipe(batch_size=batch_size, num_threads=2, device_id = 0)
    pipe.build()

    for _ in range(iterations):
        pipe_out = pipe.run()
        cmn_img_batch_cpu = pipe_out[0].as_cpu()
        crop_img_batch_cpu = pipe_out[1].as_cpu()
        for b in range(batch_size):
            img_cmn = cmn_img_batch_cpu.at(b)
            img_crop = crop_img_batch_cpu.at(b)
            assert(np.array_equal(img_cmn, img_crop))

def test_transpose():
    class TransposePipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(TransposePipe, self).__init__(batch_size, num_threads, device_id, seed=12)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = 1)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.crop = ops.Crop(device = "gpu",
                                 crop = (224, 224),
                                 image_type = types.RGB)
            self.transpose = ops.Transpose(device="gpu", perm=[2, 0, 1])

        def define_graph(self):
            imgs, labels = self.input()
            output = self.decode(imgs)
            cropped = self.crop(output)
            transposed = self.transpose(cropped)
            return (cropped, transposed, labels.gpu())

    batch_size = 8
    iterations = 8

    pipe = TransposePipe(batch_size=batch_size, num_threads=2, device_id = 0)
    pipe.build()

    for _ in range(iterations):
        pipe_out = pipe.run()
        images = pipe_out[0].asCPU().as_array()
        images_transposed = pipe_out[1].asCPU().as_array()

        for b in range(batch_size):
            np_transposed = images[b].transpose((2, 0, 1))
            np_transposed = np.ascontiguousarray(np_transposed)
            assert(np.array_equal(np_transposed, images_transposed[b]))

def test_equal_nvJPEGDecoderCrop_nvJPEGDecoder():
    """
        Comparing results of pipeline: (nvJPEGDecoder -> Crop), with the same operation performed by fused operator
    """
    batch_size =128

    class NonFusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(NonFusedPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.pos_rng_x = ops.Uniform(range = (0.0, 1.0), seed=1234)
            self.pos_rng_y = ops.Uniform(range = (0.0, 1.0), seed=5678)
            self.crop = ops.Crop(device="gpu", crop =(224,224))

        def define_graph(self):
            self.jpegs, self.labels = self.input()

            pos_x = self.pos_rng_x()
            pos_y = self.pos_rng_y()
            images = self.decode(self.jpegs)
            crop = self.crop(images, crop_pos_x=pos_x, crop_pos_y=pos_y)
            return (crop, self.labels)


        def iter_setup(self):
            pass

    class FusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(FusedPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.pos_rng_x = ops.Uniform(range = (0.0, 1.0), seed=1234)
            self.pos_rng_y = ops.Uniform(range = (0.0, 1.0), seed=5678)
            self.decode = ops.nvJPEGDecoderCrop(device = 'mixed', output_type = types.RGB, crop = (224, 224))

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            pos_x = self.pos_rng_x()
            pos_y = self.pos_rng_y()
            images = self.decode(self.jpegs, crop_pos_x=pos_x, crop_pos_y=pos_y)
            return (images, self.labels)

        def iter_setup(self):
            pass

    nonfused_pipe = NonFusedPipeline(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
    nonfused_pipe.build()
    nonfused_pipe_out = nonfused_pipe.run()
    fused_pipe = FusedPipeline(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
    fused_pipe.build()
    fused_pipe_out = fused_pipe.run()
    for i in range(batch_size):
        nonfused_pipe_out_cpu = nonfused_pipe_out[0].as_cpu()
        fused_pipe_out_cpu = fused_pipe_out[0].as_cpu()
        assert(np.sum(np.abs(nonfused_pipe_out_cpu.at(i)-fused_pipe_out_cpu.at(i)))==0)

def test_equal_nvJPEGDecoderRandomCrop_nvJPEGDecoder():
    """
        Comparing results of pipeline: (nvJPEGDecoder -> RandomCrop), with the same operation performed by fused operator
    """
    batch_size =128

    class NonFusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, seed):
            super(NonFusedPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus, seed = seed)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.res = ops.RandomResizedCrop(device="gpu", size =(224,224), seed=seed)
            self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                                output_dtype = types.FLOAT,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [128., 128., 128.],
                                                std = [1., 1., 1.])
            self.coin = ops.CoinFlip(seed = seed)

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            resized_images = self.res(images)
            mirror = self.coin()
            output = self.cmnp(resized_images, mirror = mirror)
            return (output, resized_images, self.labels)

        def iter_setup(self):
            pass

    class FusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, seed):
            super(FusedPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus, seed = seed)
            self.decode = ops.nvJPEGDecoderRandomCrop(device = "mixed", output_type = types.RGB, seed=seed)
            self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224)
            self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                                output_dtype = types.FLOAT,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [128., 128., 128.],
                                                std = [1., 1., 1.])
            self.coin = ops.CoinFlip(seed = seed)

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            resized_images = self.res(images)
            mirror = self.coin()
            output = self.cmnp(resized_images, mirror = mirror)
            return (output, resized_images, self.labels)

        def iter_setup(self):
            pass

    random_seed = 123456
    nonfused_pipe = NonFusedPipeline(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1, seed = random_seed)
    nonfused_pipe.build()
    nonfused_pipe_out = nonfused_pipe.run()

    fused_pipe = FusedPipeline(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1, seed = random_seed)
    fused_pipe.build()
    fused_pipe_out = fused_pipe.run()

    nonfused_pipe_out_cpu = nonfused_pipe_out[0].as_cpu()
    fused_pipe_out_cpu = fused_pipe_out[0].as_cpu()

    for i in range(batch_size):
        assert(np.mean(np.abs(nonfused_pipe_out_cpu.at(i)-fused_pipe_out_cpu.at(i))) < 0.5)

class ExternalInputIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        pos = []
        size = []
        for _ in range(self.batch_size):
            pos.append(np.asarray([0.4, 0.2], dtype=np.float32))
            size.append(np.asarray([0.3, 0.5], dtype=np.float32))
            self.i = (self.i + 1) % self.n
        return (pos, size)
    next = __next__

def test_equal_nvJPEGDecoderSlice_nvJPEGDecoder():
    """
        Comparing results of pipeline: (nvJPEGDecoder -> Slice), with the same operation performed by fused operator
    """
    batch_size =128
    eii = ExternalInputIterator(128)
    pos_size_iter = iter(eii)

    class NonFusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(NonFusedPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.input_crop_pos = ops.ExternalSource()
            self.input_crop_size = ops.ExternalSource()
            self.input_crop = ops.ExternalSource()
            self.decode = ops.nvJPEGDecoder(device='mixed', output_type=types.RGB)
            self.slice = ops.Slice(device = 'gpu')

        def define_graph(self):
            jpegs, labels = self.input()
            self.crop_pos = self.input_crop_pos()
            self.crop_size = self.input_crop_size()
            images = self.decode(jpegs)
            slice = self.slice(images, self.crop_pos, self.crop_size)
            return (slice, labels)


        def iter_setup(self):
            (crop_pos, crop_size) = pos_size_iter.next()
            self.feed_input(self.crop_pos, crop_pos)
            self.feed_input(self.crop_size, crop_size)

    class FusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(FusedPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.input_crop_pos = ops.ExternalSource()
            self.input_crop_size = ops.ExternalSource()
            self.input_crop = ops.ExternalSource()
            self.decode = ops.nvJPEGDecoderSlice(device = 'mixed', output_type = types.RGB)

        def define_graph(self):
            jpegs, labels = self.input()
            self.crop_pos = self.input_crop_pos()
            self.crop_size = self.input_crop_size()
            images = self.decode(jpegs, self.crop_pos, self.crop_size)
            return (images, labels)

        def iter_setup(self):
            (crop_pos, crop_size) = pos_size_iter.next()
            self.feed_input(self.crop_pos, crop_pos)
            self.feed_input(self.crop_size, crop_size)

    nonfused_pipe = NonFusedPipeline(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
    nonfused_pipe.build()
    nonfused_pipe_out = nonfused_pipe.run()
    fused_pipe = FusedPipeline(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
    fused_pipe.build()
    fused_pipe_out = fused_pipe.run()
    for i in range(batch_size):
        nonfused_pipe_out_cpu = nonfused_pipe_out[0].as_cpu()
        fused_pipe_out_cpu = fused_pipe_out[0].as_cpu()
        assert(np.sum(np.abs(nonfused_pipe_out_cpu.at(i)-fused_pipe_out_cpu.at(i)))==0)

def test_iter_setup():
    class TestIterator():
        def __init__(self, n):
            self.n = n

        def __iter__(self):
            self.i = 0
            return self

        def __next__(self):
            batch = []
            if self.i < self.n:
                batch.append(np.arange(0, 1, dtype=np.float))
                self.i += 1
                return batch
            else:
                self.i = 0
                raise StopIteration
        next = __next__

    class IterSetupPipeline(Pipeline):
        def __init__(self, iterator, num_threads, device_id):
            super(IterSetupPipeline, self).__init__(1, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.iterator = iterator

        def define_graph(self):
            self.batch = self.input()
            return self.batch

        def iter_setup(self):
            batch = next(self.iterator)
            self.feed_input(self.batch, batch)

    iter_num = 5
    iterator = iter(TestIterator(iter_num))
    i = 0
    while True:
        try:
            batch = next(iterator)
            i += 1
        except StopIteration:
            break
    assert(iter_num == i)

    iterator = iter(TestIterator(iter_num))
    pipe = IterSetupPipeline(iterator, 3, 0)
    pipe.build()

    i = 0
    while True:
        try:
            pipe_out = pipe.run()
            i += 1
        except StopIteration:
            break
    assert(iter_num == i)

    pipe.reset()
    i = 0
    while True:
        try:
            pipe_out = pipe.run()
            i += 1
        except StopIteration:
            break
    assert(iter_num == i)

def test_external_source():
    class TestIterator():
        def __init__(self, n):
            self.n = n

        def __iter__(self):
            self.i = 0
            return self

        def __next__(self):
            batch_1 = []
            batch_2 = []
            if self.i < self.n:
                batch_1.append(np.arange(0, 1, dtype=np.float))
                batch_2.append(np.arange(0, 1, dtype=np.float))
                self.i += 1
                return batch_1, batch_2
            else:
                self.i = 0
                raise StopIteration
        next = __next__

    class IterSetupPipeline(Pipeline):
        def __init__(self, iterator, num_threads, device_id):
            super(IterSetupPipeline, self).__init__(1, num_threads, device_id)
            self.input_1 = ops.ExternalSource()
            self.input_2 = ops.ExternalSource()
            self.iterator = iterator

        def define_graph(self):
            self.batch_1 = self.input_1()
            self.batch_2 = self.input_2()
            return [self.batch_1 ]

        def iter_setup(self):
            batch_1, batch_2 = next(self.iterator)
            self.feed_input(self.batch_1, batch_1)
            self.feed_input(self.batch_2, batch_2)

    iter_num = 5
    iterator = iter(TestIterator(iter_num))
    pipe = IterSetupPipeline(iterator, 3, 0)
    pipe.build()

    i = 0
    while True:
        try:
            pipe_out = pipe.run()
            i += 1
        except StopIteration:
            break
    assert(iter_num == i)

def test_element_extract_operator():
    batch_size = 4
    F = 10
    W = 32
    H = 32
    C = 3

    test_data = []
    for _ in range(batch_size):
        test_data.append( np.array( np.random.rand(F, H, W, C) * 255, dtype = np.uint8 ) )

    class ExternalInputIterator(object):
        def __init__(self, batch_size):
            self.batch_size = batch_size

        def __iter__(self):
            self.i = 0
            self.n = self.batch_size
            return self

        def __next__(self):
            batch = test_data
            self.i = (self.i + 1) % self.n
            return (batch)

        next = __next__

    eii = ExternalInputIterator(batch_size)
    iterator = iter(eii)

    class ElementExtractPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(ElementExtractPipeline, self).__init__(batch_size, num_threads, device_id)
            self.inputs = ops.ExternalSource()
            # Extract first element in each sample
            self.element_extract_first = ops.ElementExtract(element_map=[0])
            # Extract last element in each sample
            self.element_extract_last = ops.ElementExtract(element_map=[F-1])
            # Extract both first and last element in each sample to two separate outputs
            self.element_extract_first_last = ops.ElementExtract(element_map=[0, F-1])

        def define_graph(self):
            self.sequences = self.inputs()
            first_element_1 = self.element_extract_first(self.sequences)
            last_element_1 = self.element_extract_last(self.sequences)
            first_element_2, last_element_2 = self.element_extract_first_last(self.sequences)
            return (first_element_1, last_element_1, first_element_2, last_element_2)

        def iter_setup(self):
            sequences = iterator.next()
            self.feed_input(self.sequences, sequences)


    pipe = ElementExtractPipeline(batch_size, 1, 0)
    pipe.build()
    pipe_out = pipe.run()
    output1, output2, output3, output4 = pipe_out

    assert len(output1) == batch_size
    assert len(output2) == batch_size
    assert len(output3) == batch_size
    assert len(output4) == batch_size

    for i in range(batch_size):
        out1 = output1.at(i)
        out2 = output2.at(i)
        out3 = output3.at(i)
        out4 = output4.at(i)

        expected_first = test_data[i][0]
        assert out1.shape == out3.shape
        np.testing.assert_array_equal( expected_first, out1 )
        np.testing.assert_array_equal( expected_first, out3 )

        expected_last = test_data[i][F-1]
        assert out2.shape == out4.shape
        np.testing.assert_array_equal( expected_last, out2 )
        np.testing.assert_array_equal( expected_last, out4 )

def test_pipeline_default_cuda_stream_priority():
    batch_size = 16
    n_iters = 12

    class HybridPipe(Pipeline):
        def __init__(self, batch_size, default_cuda_stream_priority = 0):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads=1,
                                             device_id=0, prefetch_queue_depth=1,
                                             exec_async=False, exec_pipelined=False,
                                             default_cuda_stream_priority=default_cuda_stream_priority)
            self.input = ops.CaffeReader(path = caffe_db_folder)
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            return images

        def iter_setup(self):
            pass

    HIGH_PRIORITY = -1
    LOW_PRIORITY = 0
    pipe1 = HybridPipe(batch_size=batch_size, default_cuda_stream_priority=HIGH_PRIORITY)
    pipe2 = HybridPipe(batch_size=batch_size, default_cuda_stream_priority=LOW_PRIORITY)
    pipe1.build()
    pipe2.build()
    for _ in range(n_iters):
        out1 = pipe1.run()
        out2 = pipe2.run()
        for i in range(batch_size):
            out1_data = out1[0].as_cpu()
            out2_data = out2[0].as_cpu()
            assert(np.sum(np.abs(out1_data.at(i)-out2_data.at(i)))==0)

def test_nvjpegdecoder_cached_vs_non_cached():
    """
        Checking that cached nvJPEGDecoder produces the same output as non cached version
    """
    batch_size = 26

    class ComparePipeline(Pipeline):
        def __init__(self, batch_size=batch_size, num_threads=1, device_id=0, num_gpus=10000):
            super(ComparePipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth = 1)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus, stick_to_shard = True)
            self.decode_non_cached = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
            self.decode_cached     = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                                       cache_size=8000,
                                                       cache_threshold=0,
                                                       cache_type='threshold',
                                                       cache_debug=False)

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images_non_cached = self.decode_non_cached(self.jpegs)
            images_cached = self.decode_cached(self.jpegs)
            return (images_non_cached, images_cached)

        def iter_setup(self):
            pass

    pipe = ComparePipeline()
    pipe.build()
    N_iterations = 100
    for k in range(N_iterations):
        pipe_out = pipe.run()
        non_cached_data = pipe_out[0].as_cpu()
        cached_data = pipe_out[1].as_cpu()
        check_batch(non_cached_data, cached_data, batch_size)
