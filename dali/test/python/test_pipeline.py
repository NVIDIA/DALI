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

import glob
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
from timeit import default_timer as timer
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os
import random
from PIL import Image
from math import floor

from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import get_dali_extra_path
from test_utils import RandomDataIterator
from nose.tools import assert_raises

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')
caffe_no_label_db_folder = os.path.join(test_data_root, 'db', 'lmdb')
c2lmdb_db_folder = os.path.join(test_data_root, 'db', 'c2lmdb')
c2lmdb_no_label_db_folder = os.path.join(test_data_root, 'db', 'c2lmdb_no_label')
recordio_db_folder = os.path.join(test_data_root, 'db', 'recordio')
tfrecord_db_folder = os.path.join(test_data_root, 'db', 'tfrecord')
jpeg_folder = os.path.join(test_data_root, 'db', 'single', 'jpeg')
coco_image_folder = os.path.join(test_data_root, 'db', 'coco', 'images')
coco_annotation_file = os.path.join(test_data_root, 'db', 'coco', 'instances.json')
test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')

def test_tensor_multiple_uses():
    batch_size = 128
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
            self.res = ops.Resize(device="cpu", resize_x=224, resize_y=224)
            self.dump_cpu = ops.DumpImage(device = "cpu", suffix = "cpu")
            self.dump_gpu = ops.DumpImage(device = "gpu", suffix = "gpu")

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images = self.res(images)
            images_cpu = self.dump_cpu(images)
            images_gpu = self.dump_gpu(images.gpu())
            return (images, images_cpu, images_gpu)

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

def test_multiple_input_sets():
    batch_size = 32
    file_root = os.path.join(test_data_root, 'db', 'coco', 'images')
    annotations_file = os.path.join(test_data_root, 'db', 'coco', 'instances.json')

    class MISPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(MISPipe, self).__init__(batch_size, num_threads, device_id, num_gpus)

            # Reading COCO dataset
            self.input = ops.COCOReader(
                file_root=file_root,
                annotations_file=annotations_file,
                shard_id=device_id,
                num_shards=num_gpus,
                ratio=True,
                ltrb=True,
                random_shuffle=False)

            self.decode_cpu = ops.ImageDecoder(device="cpu", output_type=types.RGB)
            self.decode_crop = ops.ImageDecoderSlice(device="cpu", output_type=types.RGB)

            self.ssd_crop = ops.SSDRandomCrop(device="cpu", num_attempts=1, seed=0)
            default_boxes = [0.0, 0.0, 1.0, 1.0]
            self.box_encoder_cpu = ops.BoxEncoder(device="cpu", criteria=0.5, anchors=default_boxes)

        def define_graph(self):
            # Do separate augmentations
            inputs0, boxes0, labels0 = self.input(name="Reader0")
            image0 = self.decode_cpu(inputs0)
            image_ssd0, boxes_ssd0, labels_ssd0 = self.ssd_crop(image0, boxes0, labels0)

            inputs1, boxes1, labels1 = self.input(name="Reader1")
            image1 = self.decode_cpu(inputs1)
            image_ssd1, boxes_ssd1, labels_ssd1 = self.ssd_crop(image1, boxes1, labels1)

            encoded_boxes0, encoded_labels0 = self.box_encoder_cpu(boxes_ssd0, labels_ssd0)
            encoded_boxes1, encoded_labels1 = self.box_encoder_cpu(boxes_ssd1, labels_ssd1)

            # Pack into Multiple Input Sets and gather multiple output lists
            boxes = [boxes_ssd0, boxes_ssd1]
            labels = [labels_ssd0, labels_ssd1]
            enc_boxes0, enc_labels0 = self.box_encoder_cpu(boxes, labels)
            # Test one list with one _EdgeReference
            enc_boxes1, enc_labels1 = self.box_encoder_cpu(boxes, labels_ssd0)

            # Return everything (only _EdgeReference allowed)
            return (encoded_boxes0, encoded_labels0, encoded_boxes1, encoded_labels1,
                    enc_boxes0[0], enc_labels0[0], enc_boxes0[1], enc_labels0[1],
                    enc_boxes1[0], enc_labels1[0], enc_boxes1[1], enc_labels1[1])

    pipe = MISPipe(batch_size = batch_size, num_threads = 1, device_id = 0, num_gpus = 1)
    pipe.build()
    out = pipe.run()
    for i in range(batch_size):
        for j in range(0, len(out) - 2, 2):
            # All boxes should be the same
            assert(np.array_equal(out[j].at(i), out[j + 2].at(i)))
            # All labels should be the same
            assert(np.array_equal(out[j + 1].at(i), out[j + 3].at(i)))


def test_pipeline_separated_exec_setup():
    batch_size = 128
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, prefetch_queue_depth):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id, prefetch_queue_depth = prefetch_queue_depth)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
            self.res = ops.Resize(device="cpu", resize_x=224, resize_y=224)
            self.dump_cpu = ops.DumpImage(device = "cpu", suffix = "cpu")
            self.dump_gpu = ops.DumpImage(device = "gpu", suffix = "gpu")

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images = self.res(images)
            images_cpu = self.dump_cpu(images)
            images_gpu = self.dump_gpu(images.gpu())
            return (images, images_cpu, images_gpu)

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
            self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
            self.dump_gpu = ops.DumpImage(device = "gpu", suffix = "gpu")

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images_gpu = self.dump_gpu(images.gpu())
            return (images, images_gpu)

    pipe = HybridPipe(batch_size=batch_size)
    pipe.build()
    for _ in range(n_iters):
        out = pipe.run()

def test_use_twice():
    batch_size = 128
    class Pipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(Pipe, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
            self.res = ops.Resize(device="cpu", resize_x=224, resize_y=224)

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images0 = self.res(images)
            images1 = self.res(images)
            return (images0, images1)

    pipe = Pipe(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
    pipe.build()
    out = pipe.run()
    assert(out[0].is_dense_tensor())
    assert(out[1].is_dense_tensor())
    assert(out[0].as_tensor().shape() == out[1].as_tensor().shape())
    for i in range(batch_size):
        assert(np.array_equal(out[0].at(i), out[0].at(i)))

def test_cropmirrornormalize_layout():
    batch_size = 128
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
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
            self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
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

def test_cropmirrornormalize_multiple_inputs():
    batch_size = 13
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads=1, device_id=0, num_gpus=1, device="cpu"):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.device = device
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
            self.decode2 = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
            self.cmnp = ops.CropMirrorNormalize(device = device,
                                                output_dtype = types.FLOAT,
                                                output_layout = types.NHWC,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [128., 128., 128.],
                                                std = [1., 1., 1.])

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images2 = self.decode2(inputs)

            images_device  = images if self.device == "cpu" else images.gpu()
            images2_device = images2 if self.device == "cpu" else images2.gpu()

            output1, output2 = self.cmnp([images_device, images2_device])
            output3 = self.cmnp([images_device])
            output4 = self.cmnp([images2_device])
            return (output1, output2, output3, output4)

    for device in ["cpu", "gpu"]:
        pipe = HybridPipe(batch_size=batch_size, device=device)
        pipe.build()
        for _ in range(5):
            out1, out2, out3, out4 = pipe.run()
            outs = [out.as_cpu() if device == 'gpu' else out for out in [out1, out2, out3, out4] ]
            check_batch(outs[0], outs[1], batch_size)
            check_batch(outs[0], outs[2], batch_size)
            check_batch(outs[1], outs[3], batch_size)

def test_seed():
    batch_size = 64
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size,
                                             num_threads,
                                             device_id,
                                             seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
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
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
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
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
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

def test_make_continuous_serialize():
    batch_size = 32
    class COCOPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(COCOPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.COCOReader(file_root=coco_image_folder, annotations_file=coco_annotation_file, ratio=True, ltrb=True)
            self.decode = ops.ImageDecoder(device="mixed")
            self.crop = ops.RandomBBoxCrop(device="cpu", seed = 12)
            self.slice = ops.Slice(device="gpu")

        def define_graph(self):
            inputs, bboxes, labels = self.input()
            images = self.decode(inputs)
            crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
            images = self.slice(images, crop_begin, crop_size)
            return images

    pipe = COCOPipeline(batch_size=batch_size, num_threads=2, device_id=0)
    serialized_pipeline = pipe.serialize()
    del(pipe)
    new_pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    new_pipe.deserialize_and_build(serialized_pipeline)

def test_make_continuous_serialize_and_use():
    batch_size = 2
    class COCOPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(COCOPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.COCOReader(file_root=coco_image_folder, annotations_file=coco_annotation_file, ratio=True, ltrb=True)
            self.decode = ops.ImageDecoder(device="mixed")
            self.crop = ops.RandomBBoxCrop(device="cpu", seed = 25)
            self.slice = ops.Slice(device="gpu")

        def define_graph(self):
            inputs, bboxes, labels = self.input()
            images = self.decode(inputs)
            crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
            images = self.slice(images, crop_begin, crop_size)
            return images

    pipe = COCOPipeline(batch_size=batch_size, num_threads=2, device_id=0)
    serialized_pipeline = pipe.serialize()
    new_pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    new_pipe.deserialize_and_build(serialized_pipeline)

    compare_pipelines(pipe, new_pipe, batch_size, 50)

def test_warpaffine():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
            self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                                output_dtype = types.FLOAT,
                                                output_layout = types.NHWC,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [128., 128., 128.],
                                                std = [1., 1., 1.])
            self.affine = ops.WarpAffine(device = "gpu",
                                         matrix = [1.0, 0.8, -0.8*112, 0.0, 1.2, -0.2*112],
                                         fill_value = 128,
                                         interp_type = types.INTERP_LINEAR)
            self.iter = 0

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            outputs = self.cmnp([images, images])
            outputs[1] = self.affine(outputs[1])
            return [self.labels] + outputs

    pipe = HybridPipe(batch_size=128, num_threads=2, device_id = 0)
    pipe.build()
    pipe_out = pipe.run()
    import cv2
    orig_cpu = pipe_out[1].as_cpu()
    for i in range(128):
        orig = orig_cpu.at(i)
        # apply 0.5 correction for opencv's not-so-good notion of pixel centers
        M = np.array([1.0, 0.8, -0.8*(112 - 0.5), 0.0, 1.2, -0.2*(112 - 0.5)]).reshape((2,3))
        out = cv2.warpAffine(orig, M, (224,224), borderMode=cv2.BORDER_CONSTANT, borderValue = (128, 128, 128),
                             flags = (cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR))
        dali_output = pipe_out[2].as_cpu().at(i)
        maxdif = np.max(cv2.absdiff(out, dali_output)/255.0)
        assert(maxdif < 0.025)

def test_type_conversion():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
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
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
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
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
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
        images = pipe_out[0].as_cpu().as_array()
        images_transposed = pipe_out[1].as_cpu().as_array()

        for b in range(batch_size):
            np_transposed = images[b].transpose((2, 0, 1))
            np_transposed = np.ascontiguousarray(np_transposed)
            assert(np.array_equal(np_transposed, images_transposed[b]))

def test_equal_ImageDecoderCrop_ImageDecoder():
    """
        Comparing results of pipeline: (ImageDecoder -> Crop), with the same operation performed by fused operator
    """
    batch_size =128

    class NonFusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(NonFusedPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
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

    class FusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(FusedPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
            self.pos_rng_x = ops.Uniform(range = (0.0, 1.0), seed=1234)
            self.pos_rng_y = ops.Uniform(range = (0.0, 1.0), seed=5678)
            self.decode = ops.ImageDecoderCrop(device = 'mixed', output_type = types.RGB, crop = (224, 224))

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            pos_x = self.pos_rng_x()
            pos_y = self.pos_rng_y()
            images = self.decode(self.jpegs, crop_pos_x=pos_x, crop_pos_y=pos_y)
            return (images, self.labels)

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

def test_equal_ImageDecoderRandomCrop_ImageDecoder():
    """
        Comparing results of pipeline: (ImageDecoder -> RandomCrop), with the same operation performed by fused operator
    """
    batch_size =128

    class NonFusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, seed):
            super(NonFusedPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus, seed = seed)
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
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

    class FusedPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, seed):
            super(FusedPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus, seed = seed)
            self.decode = ops.ImageDecoderRandomCrop(device = "mixed", output_type = types.RGB, seed=seed)
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

class LazyPipeline(Pipeline):
    def __init__(self, batch_size, db_folder, lazy_type, num_threads=1, device_id=0, num_gpus=1):
        super(LazyPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id)
        self.input = ops.CaffeReader(path = db_folder, shard_id = device_id, num_shards = num_gpus, lazy_init = lazy_type)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
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

def test_lazy_init_empty_data_path():
    empty_db_folder="/data/empty"
    batch_size = 128

    nonlazy_pipe = LazyPipeline(batch_size, empty_db_folder, lazy_type=False)
    try:
        nonlazy_pipe.build()
        assert(False)
    except RuntimeError:
        assert(True)

    lazy_pipe = LazyPipeline(batch_size, empty_db_folder, lazy_type=True)
    try:
        lazy_pipe.build()
        assert(True)
    except BaseException:
        assert(False)

def test_lazy_init():
    """
        Comparing results of pipeline: lazy_init false and lazy_init true with empty folder and real folder
    """
    batch_size =128
    compare_pipelines(LazyPipeline(batch_size, caffe_db_folder, lazy_type=False),
                      LazyPipeline(batch_size, caffe_db_folder, lazy_type=True),
                      batch_size=batch_size, N_iterations=20)

def test_equal_ImageDecoderSlice_ImageDecoder():
    """
        Comparing results of pipeline: (ImageDecoder -> Slice), with the same operation performed by fused operator
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
            self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
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
            self.decode = ops.ImageDecoderSlice(device = 'mixed', output_type = types.RGB)

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
            return [self.batch_1, self.batch_2]

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

def test_external_source_fail():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_s_size, num_threads, device_id):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_s_size_ = external_s_size

        def define_graph(self):
            self.batch = self.input()
            return [self.batch]

        def iter_setup(self):
            batch = np.zeros([self.external_s_size_,4,5])
            self.feed_input(self.batch, batch)

    batch_size = 3
    pipe = ExternalSourcePipeline(batch_size, batch_size - 1, 3, 0)
    pipe.build()
    assert_raises(RuntimeError, pipe.run)
  
def test_external_source_fail_missing_output():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_s_size, num_threads, device_id):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.input_2 = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_s_size_ = external_s_size

        def define_graph(self):
            self.batch = self.input()
            self.batch_2 = self.input_2()
            return [self.batch]

        def iter_setup(self):
            batch = np.zeros([self.external_s_size_,4,5])
            self.feed_input(self.batch, batch)
            self.feed_input(self.batch_2, batch)

    batch_size = 3
    pipe = ExternalSourcePipeline(batch_size, batch_size, 3, 0)
    pipe.build()
    assert_raises(RuntimeError, pipe.run)

def test_external_source_fail_list():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_s_size, num_threads, device_id):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_s_size_ = external_s_size

        def define_graph(self):
            self.batch = self.input()
            return [self.batch]

        def iter_setup(self):
            batch = []
            for _ in range(self.external_s_size_):
                batch.append(np.zeros([3,4,5]))
            self.feed_input(self.batch, batch)

    batch_size = 3
    pipe = ExternalSourcePipeline(batch_size, batch_size - 1, 3, 0)
    pipe.build()
    assert_raises(RuntimeError, pipe.run)

def test_external_source_scalar_list():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_data, num_threads, device_id, label_data):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_data = external_data
            self.label_data_ = label_data

        def define_graph(self):
            self.batch = self.input()
            return [self.batch]

        def iter_setup(self):
            batch = []
            for elm in self.external_data:
                batch.append(np.array(elm, dtype=np.uint8))
            self.feed_input(self.batch, batch)

    batch_size = 3
    label_data = 10
    lists = []
    scalars = []
    for i in range(batch_size):
        lists.append([label_data + i])
        scalars.append(label_data + i * 10)
    for external_data in [lists, scalars]:
        print(external_data)
        pipe = ExternalSourcePipeline(batch_size, external_data, 3, 0, label_data)
        pipe.build()
        for _ in range(10):
            out = pipe.run()
            for i in range(batch_size):
                assert out[0].as_array()[i] == external_data[i]
        yield external_data_veri, external_data

def test_external_source_gpu():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, use_list):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource(device="gpu")
            self.crop = ops.Crop(device="gpu", crop_h=32, crop_w=32, crop_pos_x=0.2, crop_pos_y=0.2)
            self.use_list = use_list

        def define_graph(self):
            self.batch = self.input()
            output = self.crop(self.batch)
            return output

        def iter_setup(self):
            if use_list:
                batch_data = [np.random.rand(100, 100, 3) for _ in range(self.batch_size)]
            else:
                batch_data = np.random.rand(self.batch_size, 100, 100, 3)
            self.feed_input(self.batch, batch_data)

    for batch_size in [1, 10]:
        for use_list in (True, False):
            pipe = ExternalSourcePipeline(batch_size, 3, 0, use_list)
            pipe.build()
            try: 
                pipe.run()
            except RuntimeError as e:
                if not use_list:
                    assert(1), "For tensor list GPU external source should fail"

def external_data_veri(external_data):
    pass

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
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            return images

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

class CachedPipeline(Pipeline):
    def __init__(self, reader_type, batch_size, is_cached=False, is_cached_batch_copy=True,  seed=123456, skip_cached_images=False, num_shards=100000):
        super(CachedPipeline, self).__init__(batch_size, num_threads=1, device_id=0, prefetch_queue_depth=1, seed=seed)
        self.reader_type = reader_type
        if reader_type == "MXNetReader":
            self.input = ops.MXNetReader(path = os.path.join(recordio_db_folder, "train.rec"),
                                         index_path = os.path.join(recordio_db_folder, "train.idx"),
                                         shard_id = 0,
                                         num_shards = num_shards,
                                         stick_to_shard = True,
                                         skip_cached_images = skip_cached_images,
                                         prefetch_queue_depth = 1)
        elif reader_type == "CaffeReader":
            self.input = ops.CaffeReader(path = caffe_db_folder,
                                         shard_id = 0,
                                         num_shards = num_shards,
                                         stick_to_shard = True,
                                         skip_cached_images = skip_cached_images,
                                         prefetch_queue_depth = 1)
        elif reader_type == "Caffe2Reader":
            self.input = ops.Caffe2Reader(path = c2lmdb_db_folder,
                                          shard_id = 0,
                                          num_shards = num_shards,
                                          stick_to_shard = True,
                                          skip_cached_images = skip_cached_images,
                                          prefetch_queue_depth = 1)
        elif reader_type == "FileReader":
            self.input = ops.FileReader(file_root = jpeg_folder,
                                        shard_id = 0,
                                        num_shards = num_shards,
                                        stick_to_shard = True,
                                        skip_cached_images = skip_cached_images,
                                        prefetch_queue_depth = 1)

        elif reader_type == "TFRecordReader":
            tfrecord = sorted(glob.glob(os.path.join(tfrecord_db_folder, '*[!i][!d][!x]')))
            tfrecord_idx = sorted(glob.glob(os.path.join(tfrecord_db_folder, '*idx')))
            self.input = ops.TFRecordReader(path = tfrecord,
                                            index_path = tfrecord_idx,
                                            shard_id = 0,
                                            num_shards = num_shards,
                                            stick_to_shard = True,
                                            skip_cached_images = skip_cached_images,
                                            features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                                        "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)})

        if is_cached:
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB,
                                            cache_size=2000,
                                            cache_threshold=0,
                                            cache_type='threshold',
                                            cache_debug=False,
                                            cache_batch_copy=is_cached_batch_copy)
        else:
           self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)

    def define_graph(self):
        if self.reader_type == "TFRecordReader":
            inputs = self.input()
            jpegs = inputs["image/encoded"]
            labels = inputs["image/class/label"]
        else:
            jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)


def test_nvjpeg_cached_batch_copy_pipelines():
    batch_size = 26
    for reader_type in {"MXNetReader", "CaffeReader", "Caffe2Reader", "FileReader", "TFRecordReader"}:
        compare_pipelines(CachedPipeline(reader_type, batch_size, is_cached=True, is_cached_batch_copy=True),
                          CachedPipeline(reader_type, batch_size, is_cached=True, is_cached_batch_copy=False),
                          batch_size=batch_size, N_iterations=20)

def test_nvjpeg_cached_pipelines():
    batch_size = 26
    for reader_type in {"MXNetReader", "CaffeReader", "Caffe2Reader", "FileReader", "TFRecordReader"}:
        compare_pipelines(CachedPipeline(reader_type, batch_size, is_cached=False),
                          CachedPipeline(reader_type, batch_size, is_cached=True),
                          batch_size=batch_size, N_iterations=20)

def test_skip_cached_images():
    batch_size = 1
    for reader_type in {"MXNetReader", "CaffeReader", "Caffe2Reader", "FileReader"}:
        compare_pipelines(CachedPipeline(reader_type, batch_size, is_cached=False),
                          CachedPipeline(reader_type, batch_size, is_cached=True, skip_cached_images=True),
                          batch_size=batch_size, N_iterations=100)

def test_caffe_no_label():
    class CaffePipeline(Pipeline):
        def __init__(self, batch_size, path_to_data, labels, seed=123456, skip_cached_images=False, num_shards=1):
            super(CaffePipeline, self).__init__(batch_size, num_threads=1, device_id=0, prefetch_queue_depth=1, seed=seed)
            self.input = ops.CaffeReader(path = path_to_data,
                                          shard_id = 0,
                                          num_shards = num_shards,
                                          stick_to_shard = True,
                                          prefetch_queue_depth = 1,
                                          label_available = labels)
            self.decode = ops.ImageDecoder(output_type = types.RGB)
            self.labels = labels

        def define_graph(self):
            if not self.labels:
                jpegs = self.input()
            else:
                jpegs,_ = self.input()
            images = self.decode(jpegs)
            return (images)

    pipe = CaffePipeline(2, caffe_db_folder, True)
    pipe.build()
    pipe.run()
    pipe = CaffePipeline(2, caffe_no_label_db_folder, False)
    pipe.build()
    pipe.run()

def test_caffe2_no_label():
    class Caffe2Pipeline(Pipeline):
        def __init__(self, batch_size, path_to_data, label_type, seed=123456, skip_cached_images=False, num_shards=1):
            super(Caffe2Pipeline, self).__init__(batch_size, num_threads=1, device_id=0, prefetch_queue_depth=1, seed=seed)
            self.input = ops.Caffe2Reader(path = path_to_data,
                                          shard_id = 0,
                                          num_shards = num_shards,
                                          stick_to_shard = True,
                                          prefetch_queue_depth = 1,
                                          label_type = label_type)
            self.decode = ops.ImageDecoder(output_type = types.RGB)
            self.label_type = label_type

        def define_graph(self):
            if self.label_type == 4:
                jpegs = self.input()
            else:
                jpegs,_ = self.input()
            images = self.decode(jpegs)
            return (images)

    pipe = Caffe2Pipeline(2, c2lmdb_db_folder, 0)
    pipe.build()
    pipe.run()
    pipe = Caffe2Pipeline(2, c2lmdb_no_label_db_folder, 4)
    pipe.build()
    pipe.run()

def test_as_tensor():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)

        def define_graph(self):
            _, self.labels = self.input()
            return self.labels
    batch_size = 8
    shape = [[2, 2, 2], [8, 1], [1, 8], [4, 2], [2, 4], [8], [1, 2, 1, 2, 1, 2], [1, 1, 1, 8]]
    pipe = HybridPipe(batch_size=batch_size, num_threads=2, device_id = 0)
    pipe.build()
    for sh in shape:
        pipe_out = pipe.run()[0]
        assert(pipe_out.as_tensor().shape() == [batch_size, 1])
        assert(pipe_out.as_reshaped_tensor(sh).shape() == sh)
        different_shape = random.choice(shape)
        assert(pipe_out.as_reshaped_tensor(different_shape).shape() == different_shape)

def test_as_tensor_fail():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed = 12)
            self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle = True)

        def define_graph(self):
            _, self.labels = self.input()
            return self.labels
    batch_size = 8
    shape = [[2, 2, 2, 3], [8, 1, 6], [1, 8, 4], [4, 2, 9], [2, 4, 0], [8, 2], [1, 2, 1, 2, 1, 2, 3], [7, 1, 1, 1, 8]]
    pipe = HybridPipe(batch_size=batch_size, num_threads=2, device_id = 0)
    pipe.build()
    for sh in shape:
        pipe_out = pipe.run()[0]
        assert(pipe_out.as_tensor().shape() == [batch_size, 1])
        try:
            assert(pipe_out.as_reshaped_tensor(sh).shape() == sh)
            assert(False)
        except RuntimeError:
            assert(True)

def test_python_formats():
    class TestPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, test_array):
            super(TestPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input_data = ops.ExternalSource()
            self.test_array = test_array

        def define_graph(self):
            self.data = self.input_data()
            return (self.data)


        def iter_setup(self):
            self.feed_input(self.data, self.test_array)

    for t in [np.bool_, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64, np.float_, np.float32, np.float16,
             np.short, np.long, np.longlong, np.ushort, np.ulonglong]:
        test_array = np.array([[1, 1], [1, 1]], dtype=t)
        pipe = TestPipeline(2, 1, 0, 1, test_array)
        pipe.build()
        out = pipe.run()[0]
        out_dtype = out.at(0).dtype
        assert(test_array.dtype.itemsize == out_dtype.itemsize)
        assert(test_array.dtype.str == out_dtype.str)

def test_api_check1():
    batch_size = 1
    class TestPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(TestPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            return (inputs)

    pipe = TestPipeline(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
    pipe.build()
    pipe.run()
    for method in [pipe.schedule_run, pipe.share_outputs, pipe.release_outputs, pipe.outputs]:
        try:
            method()
            assert(False)
        except RuntimeError:
            assert(True)
    # disable check
    pipe.enable_api_check(False)
    for method in [pipe.schedule_run, pipe.share_outputs, pipe.release_outputs, pipe.outputs]:
        try:
            method()
            assert(True)
        except RuntimeError:
            assert(False)

def test_api_check2():
    batch_size = 1
    class TestPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(TestPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
            self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            return (inputs)

    pipe = TestPipeline(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1)
    pipe.build()
    pipe.schedule_run()
    pipe.share_outputs()
    pipe.release_outputs()
    pipe.schedule_run()
    pipe.outputs()
    try:
        pipe.run()
        assert(False)
    except RuntimeError:
        assert(True)
    # disable check
    pipe.enable_api_check(False)
    try:
        pipe.run()
        assert(True)
    except RuntimeError:
        assert(False)

class DupPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, first_out_device = "cpu", second_out_device = "cpu"):
        super(DupPipeline, self).__init__(batch_size, num_threads, device_id)
        self.first_out_device = first_out_device
        self.second_out_device = second_out_device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = 1)
        self.decode = ops.ImageDecoder(device = "mixed" if first_out_device == "mixed" else "cpu", output_type = types.RGB)
        if self.second_out_device:
            self.cmnp = ops.CropMirrorNormalize(device = second_out_device,
                                                output_dtype = types.FLOAT,
                                                output_layout = types.NHWC,
                                                crop = (224, 224),
                                                image_type = types.RGB,
                                                mean = [128., 128., 128.],
                                                std = [1., 1., 1.])

    def define_graph(self):
        inputs, _ = self.input()
        decoded_images = self.decode(inputs)
        if self.second_out_device:
            if self.first_out_device != "mixed" and self.second_out_device == "gpu":
                images = self.cmnp(decoded_images.gpu())
            else:
                images = self.cmnp(decoded_images)
        else:
            images = decoded_images
        images_2 = images
        return images, images_2, images, decoded_images

def check_duplicated_outs_pipeline(first_device, second_device):
    batch_size = 5
    pipe = DupPipeline(batch_size=batch_size, num_threads=2, device_id=0,
                       first_out_device = first_device, second_out_device = second_device)
    pipe.build()
    out = pipe.run()
    assert len(out) == 4
    for i in range(batch_size):
        assert isinstance(out[3][0], dali.backend_impl.TensorGPU) or first_device == "cpu"
        out1 = out[0].as_cpu().at(i) if isinstance(out[0][0], dali.backend_impl.TensorGPU) else out[0].at(i)
        out2 = out[1].as_cpu().at(i) if isinstance(out[1][0], dali.backend_impl.TensorGPU) else out[1].at(i)
        out3 = out[2].as_cpu().at(i) if isinstance(out[2][0], dali.backend_impl.TensorGPU) else out[2].at(i)

        np.testing.assert_array_equal( out1, out2 )
        np.testing.assert_array_equal( out1, out3 )

def test_duplicated_outs_pipeline():
    for first_device, second_device in [("cpu", None),
                                        ("cpu", "cpu"),
                                        ("cpu", "gpu"),
                                        ("mixed", None),
                                        ("mixed", "gpu")]:
        yield check_duplicated_outs_pipeline, first_device, second_device

def check_serialized_outs_duplicated_pipeline(first_device, second_device):
    batch_size = 5
    pipe = DupPipeline(batch_size=batch_size, num_threads=2, device_id=0,
                       first_out_device = first_device, second_out_device = second_device)
    serialized_pipeline = pipe.serialize()
    del(pipe)
    new_pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    new_pipe.deserialize_and_build(serialized_pipeline)
    out = new_pipe.run()
    assert len(out) == 4
    for i in range(batch_size):
        assert isinstance(out[3][0], dali.backend_impl.TensorGPU) or first_device == "cpu"
        out1 = out[0].as_cpu().at(i) if isinstance(out[0][0], dali.backend_impl.TensorGPU) else out[0].at(i)
        out2 = out[1].as_cpu().at(i) if isinstance(out[1][0], dali.backend_impl.TensorGPU) else out[1].at(i)
        out3 = out[2].as_cpu().at(i) if isinstance(out[2][0], dali.backend_impl.TensorGPU) else out[2].at(i)

        np.testing.assert_array_equal( out1, out2 )
        np.testing.assert_array_equal( out1, out3 )

def test_serialized_outs_duplicated_pipeline():
    for first_device, second_device in [("cpu", None),
                                        ("cpu", "cpu"),
                                        ("cpu", "gpu"),
                                        ("mixed", None),
                                        ("mixed", "gpu")]:
        yield check_serialized_outs_duplicated_pipeline, first_device, second_device

def check_duplicated_outs_cpu_to_gpu(device):
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
          return out, self.crop_pos, self.crop_size

      def iter_setup(self):
          data = self.iterator.next()
          self.feed_input(self.data, data, layout=self.layout)

          (crop_pos, crop_size) = self.pos_size_iter.next()
          self.feed_input(self.crop_pos, crop_pos)
          self.feed_input(self.crop_size, crop_size)

    batch_size = 1
    input_shape = (200,400,3)
    layout = "HWC"
    axes = None
    axis_names = "WH"
    normalized_anchor = False
    normalized_shape = False
    eiis = [RandomDataIterator(batch_size, shape=input_shape)
            for k in range(2)]
    eii_args = [SliceArgsIterator(batch_size, len(input_shape), image_shape=input_shape,
                image_layout=layout, axes=axes, axis_names=axis_names, normalized_anchor=normalized_anchor,
                normalized_shape=normalized_shape)
                for k in range(2)]

    pipe = SliceSynthDataPipeline(device, batch_size, layout, iter(eiis[0]), iter(eii_args[0]),
            axes=axes, axis_names=axis_names, normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape)
    pipe.build()
    out = pipe.run()
    assert isinstance(out[0][0], dali.backend_impl.TensorGPU) or device == "cpu"
    assert not isinstance(out[1][0], dali.backend_impl.TensorGPU)
    assert not isinstance(out[2][0], dali.backend_impl.TensorGPU)

# check if it is possible to return outputs from CPU op that goes directly to the GPU op without
# MakeContiguous as a CPU output from the pipeline
def test_duplicated_outs_cpu_op_to_gpu():
    for device in ["cpu", "gpu"]:
        yield check_duplicated_outs_cpu_to_gpu, device
