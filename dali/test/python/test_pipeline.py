# Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali as dali
from nvidia.dali import pipeline_def
import numpy as np
from numpy.testing import assert_array_equal
import os
import random
from math import floor, ceil
import sys
import warnings
from webdataset_base import generate_temp_index_file as generate_temp_wds_index

from test_utils import (
    check_batch,
    as_array,
    compare_pipelines,
    get_dali_extra_path,
    RandomDataIterator,
)
from nose_utils import raises, assert_raises, assert_warns, SkipTest

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, "db", "lmdb")
caffe_no_label_db_folder = os.path.join(test_data_root, "db", "lmdb")
c2lmdb_db_folder = os.path.join(test_data_root, "db", "c2lmdb")
c2lmdb_no_label_db_folder = os.path.join(test_data_root, "db", "c2lmdb_no_label")
recordio_db_folder = os.path.join(test_data_root, "db", "recordio")
tfrecord_db_folder = os.path.join(test_data_root, "db", "tfrecord")
jpeg_folder = os.path.join(test_data_root, "db", "single", "jpeg")
coco_image_folder = os.path.join(test_data_root, "db", "coco", "images")
coco_annotation_file = os.path.join(test_data_root, "db", "coco", "instances.json")
test_data_video = os.path.join(test_data_root, "db", "optical_flow", "sintel_trailer")
webdataset_db_folder = os.path.join(test_data_root, "db", "webdataset", "MNIST")


def test_tensor_multiple_uses():
    batch_size = 128

    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
            )
            self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.res = ops.Resize(device="cpu", resize_x=224, resize_y=224)
            self.dump_cpu = ops.DumpImage(device="cpu", suffix="cpu")
            self.dump_gpu = ops.DumpImage(device="gpu", suffix="gpu")

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images = self.res(images)
            images_cpu = self.dump_cpu(images)
            images_gpu = self.dump_gpu(images.gpu())
            return (images, images_cpu, images_gpu)

    pipe = HybridPipe(batch_size=batch_size, num_threads=1, device_id=0, num_gpus=1)
    out = pipe.run()
    assert out[0].is_dense_tensor()
    assert out[1].is_dense_tensor()
    assert out[2].is_dense_tensor()
    assert out[0].as_tensor().shape() == out[1].as_tensor().shape()
    assert out[0].as_tensor().shape() == out[2].as_tensor().shape()
    a_raw = out[0]
    a_cpu = out[1]
    a_gpu = out[2].as_cpu()
    for i in range(batch_size):
        t_raw = a_raw.at(i)
        t_cpu = a_cpu.at(i)
        assert np.sum(np.abs(t_cpu - t_raw)) == 0
        t_cpu = a_cpu.at(i)
        t_gpu = a_gpu.at(i)
        assert np.sum(np.abs(t_cpu - t_gpu)) == 0


def test_multiple_input_sets():
    batch_size = 32
    file_root = os.path.join(test_data_root, "db", "coco", "images")
    annotations_file = os.path.join(test_data_root, "db", "coco", "instances.json")

    class MISPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(MISPipe, self).__init__(batch_size, num_threads, device_id, num_gpus)

            # Reading COCO dataset
            self.input = ops.readers.COCO(
                file_root=file_root,
                annotations_file=annotations_file,
                shard_id=device_id,
                num_shards=num_gpus,
                ratio=True,
                ltrb=True,
                random_shuffle=False,
            )

            self.decode_cpu = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.decode_crop = ops.decoders.ImageSlice(device="cpu", output_type=types.RGB)

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
            # Test one list with one DataNode
            enc_boxes1, enc_labels1 = self.box_encoder_cpu(boxes, labels_ssd0)

            # Return everything (only DataNode allowed)
            return (
                encoded_boxes0,
                encoded_labels0,
                encoded_boxes1,
                encoded_labels1,
                enc_boxes0[0],
                enc_labels0[0],
                enc_boxes0[1],
                enc_labels0[1],
                enc_boxes1[0],
                enc_labels1[0],
                enc_boxes1[1],
                enc_labels1[1],
            )

    pipe = MISPipe(batch_size=batch_size, num_threads=1, device_id=0, num_gpus=1)
    out = pipe.run()
    for i in range(batch_size):
        for j in range(0, len(out) - 2, 2):
            # All boxes should be the same
            assert np.array_equal(out[j].at(i), out[j + 2].at(i))
            # All labels should be the same
            assert np.array_equal(out[j + 1].at(i), out[j + 3].at(i))


def test_pipeline_separated_exec_setup():
    batch_size = 128

    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, prefetch_queue_depth):
            super(HybridPipe, self).__init__(
                batch_size, num_threads, device_id, prefetch_queue_depth=prefetch_queue_depth
            )
            self.input = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
            )
            self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.res = ops.Resize(device="cpu", resize_x=224, resize_y=224)
            self.dump_cpu = ops.DumpImage(device="cpu", suffix="cpu")
            self.dump_gpu = ops.DumpImage(device="gpu", suffix="gpu")

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images = self.res(images)
            images_cpu = self.dump_cpu(images)
            images_gpu = self.dump_gpu(images.gpu())
            return (images, images_cpu, images_gpu)

    pipe = HybridPipe(
        batch_size=batch_size,
        num_threads=1,
        device_id=0,
        num_gpus=1,
        prefetch_queue_depth={"cpu_size": 5, "gpu_size": 3},
    )
    out = pipe.run()
    assert out[0].is_dense_tensor()
    assert out[1].is_dense_tensor()
    assert out[2].is_dense_tensor()
    assert out[0].as_tensor().shape() == out[1].as_tensor().shape()
    assert out[0].as_tensor().shape() == out[2].as_tensor().shape()
    a_raw = out[0]
    a_cpu = out[1]
    a_gpu = out[2].as_cpu()
    for i in range(batch_size):
        t_raw = a_raw.at(i)
        t_cpu = a_cpu.at(i)
        assert np.sum(np.abs(t_cpu - t_raw)) == 0
        t_cpu = a_cpu.at(i)
        t_gpu = a_gpu.at(i)
        assert np.sum(np.abs(t_cpu - t_gpu)) == 0


def test_pipeline_simple_sync_no_prefetch():
    batch_size = 16
    n_iters = 12

    class HybridPipe(Pipeline):
        def __init__(self, batch_size):
            super(HybridPipe, self).__init__(
                batch_size,
                num_threads=1,
                device_id=0,
                prefetch_queue_depth=1,
                exec_async=False,
                exec_pipelined=False,
            )
            self.input = ops.readers.Caffe(path=caffe_db_folder)
            self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.dump_gpu = ops.DumpImage(device="gpu", suffix="gpu")

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images_gpu = self.dump_gpu(images.gpu())
            return (images, images_gpu)

    pipe = HybridPipe(batch_size=batch_size)
    for _ in range(n_iters):
        pipe.run()


def test_use_twice():
    batch_size = 128

    class Pipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(Pipe, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
            )
            self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.res = ops.Resize(device="cpu", resize_x=224, resize_y=224)

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images0 = self.res(images)
            images1 = self.res(images)
            return (images0, images1)

    pipe = Pipe(batch_size=batch_size, num_threads=1, device_id=0, num_gpus=1)
    out = pipe.run()
    assert out[0].is_dense_tensor()
    assert out[1].is_dense_tensor()
    assert out[0].as_tensor().shape() == out[1].as_tensor().shape()
    for i in range(batch_size):
        assert np.array_equal(out[0].at(i), out[0].at(i))


def test_cropmirrornormalize_layout():
    batch_size = 128

    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
            )
            self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.cmnp_nhwc = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                output_layout=types.NHWC,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )
            self.cmnp_nchw = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            output_nhwc = self.cmnp_nhwc(images.gpu())
            output_nchw = self.cmnp_nchw(images.gpu())
            return (output_nchw, output_nhwc)

    pipe = HybridPipe(batch_size=batch_size, num_threads=1, device_id=0, num_gpus=1)
    out = pipe.run()
    assert out[0].is_dense_tensor()
    assert out[1].is_dense_tensor()
    shape_nchw = out[0].as_tensor().shape()
    shape_nhwc = out[1].as_tensor().shape()
    assert shape_nchw[0] == shape_nhwc[0]
    a_nchw = out[0].as_cpu()
    a_nhwc = out[1].as_cpu()
    for i in range(batch_size):
        t_nchw = a_nchw.at(i)
        t_nhwc = a_nhwc.at(i)
        assert t_nchw.shape == (3, 224, 224)
        assert t_nhwc.shape == (224, 224, 3)
        assert np.sum(np.abs(np.transpose(t_nchw, (1, 2, 0)) - t_nhwc)) == 0


def test_cropmirrornormalize_pad():
    batch_size = 128

    class HybridPipe(Pipeline):
        def __init__(self, layout, batch_size, num_threads, device_id, num_gpus):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
            )
            self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.cmnp_pad = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                output_layout=layout,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
                pad_output=True,
            )
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                output_layout=layout,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
                pad_output=False,
            )

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            output_pad = self.cmnp_pad(images.gpu())
            output = self.cmnp(images.gpu())
            return (output, output_pad)

    for layout in [types.NCHW, types.NHWC]:
        pipe = HybridPipe(layout, batch_size=batch_size, num_threads=1, device_id=0, num_gpus=1)
        out = pipe.run()
        assert out[0].is_dense_tensor()
        assert out[1].is_dense_tensor()
        shape = out[0].as_tensor().shape()
        shape_pad = out[1].as_tensor().shape()
        assert shape[0] == shape_pad[0]
        a = out[0].as_cpu()
        a_pad = out[1].as_cpu()
        for i in range(batch_size):
            t = a.at(i)
            t_pad = a_pad.at(i)
            if layout == types.NCHW:
                assert t.shape == (3, 224, 224)
                assert t_pad.shape == (4, 224, 224)
                assert np.sum(np.abs(t - t_pad[:3, :, :])) == 0
                assert np.sum(np.abs(t_pad[3, :, :])) == 0
            else:
                assert t.shape == (224, 224, 3)
                assert t_pad.shape == (224, 224, 4)
                assert np.sum(np.abs(t - t_pad[:, :, :3])) == 0
                assert np.sum(np.abs(t_pad[:, :, 3])) == 0


def test_cropmirrornormalize_multiple_inputs():
    batch_size = 13

    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads=1, device_id=0, num_gpus=1, device="cpu"):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id)
            self.device = device
            self.input = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
            )
            self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.decode2 = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.cmnp = ops.CropMirrorNormalize(
                device=device,
                dtype=types.FLOAT,
                output_layout=types.NHWC,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images2 = self.decode2(inputs)

            images_device = images if self.device == "cpu" else images.gpu()
            images2_device = images2 if self.device == "cpu" else images2.gpu()

            output1, output2 = self.cmnp([images_device, images2_device])
            output3 = self.cmnp([images_device])
            output4 = self.cmnp([images2_device])
            return (output1, output2, output3, output4)

    for device in ["cpu", "gpu"]:
        pipe = HybridPipe(batch_size=batch_size, device=device)
        for _ in range(5):
            out1, out2, out3, out4 = pipe.run()
            outs = [out.as_cpu() if device == "gpu" else out for out in [out1, out2, out3, out4]]
            check_batch(outs[0], outs[1], batch_size)
            check_batch(outs[0], outs[2], batch_size)
            check_batch(outs[1], outs[3], batch_size)


def test_seed():
    batch_size = 64

    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
            self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=True)
            self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )
            self.coin = ops.random.CoinFlip()
            self.uniform = ops.random.Uniform(range=(0.0, 1.0))
            self.iter = 0

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            mirror = self.coin()
            output = self.cmnp(
                images, mirror=mirror, crop_pos_x=self.uniform(), crop_pos_y=self.uniform()
            )
            return (output, self.labels)

    n = 30
    for i in range(50):
        pipe = HybridPipe(batch_size=batch_size, num_threads=2, device_id=0)
        pipe_out = pipe.run()
        pipe_out_cpu = pipe_out[0].as_cpu()
        img_chw_test = pipe_out_cpu.at(n)
        if i == 0:
            img_chw = img_chw_test
        assert np.sum(np.abs(img_chw - img_chw_test)) == 0


def test_none_seed():
    batch_size = 60

    for i in range(50):
        pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0, seed=None)
        with pipe:
            coin = fn.random.uniform(range=(0.0, 1.0))
        pipe.set_outputs(coin)
        pipe_out = pipe.run()[0]
        test_out = pipe_out.as_array()
        if i == 0:
            test_out_ref = test_out
        else:
            assert np.sum(np.abs(test_out_ref - test_out)) != 0


def test_seed_deprecated():
    @pipeline_def(batch_size=1, device_id=None, num_threads=1)
    def my_pipe():
        with assert_warns(
            DeprecationWarning,
            glob='The argument "seed" should not be used with operators '
            "that don't use random numbers.",
        ):
            return fn.reshape(np.float32([1, 2]), shape=[2], seed=123)

    my_pipe()


def test_as_array():
    batch_size = 64

    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
            self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=True)
            self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )
            self.coin = ops.random.CoinFlip()
            self.uniform = ops.random.Uniform(range=(0.0, 1.0))
            self.iter = 0

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            mirror = self.coin()
            output = self.cmnp(
                images, mirror=mirror, crop_pos_x=self.uniform(), crop_pos_y=self.uniform()
            )
            return (output, self.labels)

    for i in range(10):
        pipe = HybridPipe(batch_size=batch_size, num_threads=2, device_id=0)
        pipe_out = pipe.run()
        pipe_out_cpu = pipe_out[0].as_cpu()
        img_chw_test = pipe_out_cpu.as_array()
        if i == 0:
            img_chw = img_chw_test
        assert img_chw_test.shape == (batch_size, 3, 224, 224)
        assert np.sum(np.abs(img_chw - img_chw_test)) == 0


def test_seed_serialize():
    batch_size = 32

    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
            self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=True)
            self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )
            self.coin = ops.random.CoinFlip()
            self.uniform = ops.random.Uniform(range=(0.0, 1.0))
            self.iter = 0

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            mirror = self.coin()
            output = self.cmnp(
                images, mirror=mirror, crop_pos_x=self.uniform(), crop_pos_y=self.uniform()
            )
            return (output, self.labels)

    orig_pipe = HybridPipe(batch_size=batch_size, num_threads=2, device_id=0)
    s = orig_pipe.serialize()
    for i in range(10):
        pipe = Pipeline()
        pipe.deserialize_and_build(s)
        pipe_out = pipe.run()
        pipe_out_cpu = pipe_out[0].as_cpu()
        if i == 0:
            ref = pipe_out_cpu
        else:
            check_batch(pipe_out_cpu, ref)


def test_make_contiguous_serialize():
    batch_size = 32

    class COCOPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(COCOPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.COCO(
                file_root=coco_image_folder,
                annotations_file=coco_annotation_file,
                ratio=True,
                ltrb=True,
            )
            self.decode = ops.decoders.Image(device="mixed")
            self.crop = ops.RandomBBoxCrop(device="cpu", seed=12)
            self.slice = ops.Slice(device="gpu")

        def define_graph(self):
            inputs, bboxes, labels = self.input()
            images = self.decode(inputs)
            crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
            images = self.slice(images, crop_begin, crop_size)
            return images

    pipe = COCOPipeline(batch_size=batch_size, num_threads=2, device_id=0)
    serialized_pipeline = pipe.serialize()
    del pipe
    new_pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    new_pipe.deserialize_and_build(serialized_pipeline)


def test_make_contiguous_serialize_and_use():
    batch_size = 2

    class COCOPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(COCOPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.COCO(
                file_root=coco_image_folder,
                annotations_file=coco_annotation_file,
                ratio=True,
                ltrb=True,
            )
            self.decode = ops.decoders.Image(device="mixed")
            self.crop = ops.RandomBBoxCrop(device="cpu", seed=25)
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

    compare_pipelines(pipe, new_pipe, batch_size, 5)


def test_warpaffine():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
            self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=True)
            self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                output_layout=types.NHWC,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )
            self.affine = ops.WarpAffine(
                device="gpu",
                matrix=[1.0, 0.8, -0.8 * 112, 0.0, 1.2, -0.2 * 112],
                fill_value=128,
                interp_type=types.INTERP_LINEAR,
            )
            self.iter = 0

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            outputs = self.cmnp([images, images])
            outputs[1] = self.affine(outputs[1])
            return [self.labels] + outputs

    pipe = HybridPipe(batch_size=128, num_threads=2, device_id=0)
    _, orig_cpu, dali_output_batch = tuple(out.as_cpu() for out in pipe.run())
    import cv2

    for i in range(128):
        orig = orig_cpu.at(i)
        # apply 0.5 correction for opencv's not-so-good notion of pixel centers
        M = np.array([1.0, 0.8, -0.8 * (112 - 0.5), 0.0, 1.2, -0.2 * (112 - 0.5)]).reshape((2, 3))
        out = cv2.warpAffine(
            orig,
            M,
            (224, 224),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128),
            flags=(cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR),
        )
        dali_output = dali_output_batch.at(i)
        maxdif = np.max(cv2.absdiff(out, dali_output) / 255.0)
        assert maxdif < 0.025


def test_type_conversion():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
            self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=True)
            self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
            self.cmnp_all = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                output_layout=types.NHWC,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )
            self.cmnp_int = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                output_layout=types.NHWC,
                crop=(224, 224),
                mean=[128, 128, 128],
                std=[1.0, 1, 1],
            )  # Left 1 of the args as float to test whether mixing types works
            self.cmnp_1arg = ops.CropMirrorNormalize(
                device="gpu",
                dtype=types.FLOAT,
                output_layout=types.NHWC,
                crop=(224, 224),
                mean=128,
                std=1,
            )
            self.uniform = ops.random.Uniform(range=(0, 1))

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            outputs = [None for i in range(3)]
            crop_pos_x = self.uniform()
            crop_pos_y = self.uniform()
            outputs[0] = self.cmnp_all(images, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
            outputs[1] = self.cmnp_int(images, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
            outputs[2] = self.cmnp_1arg(images, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
            return [self.labels] + outputs

    pipe = HybridPipe(batch_size=128, num_threads=2, device_id=0)
    for i in range(10):
        pipe_out = pipe.run()
        orig_cpu = pipe_out[1].as_cpu().as_tensor()
        int_cpu = pipe_out[2].as_cpu().as_tensor()
        arg1_cpu = pipe_out[3].as_cpu().as_tensor()
        assert_array_equal(orig_cpu, int_cpu)
        assert_array_equal(orig_cpu, arg1_cpu)


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
        super(LazyPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.readers.Caffe(
            path=db_folder, shard_id=device_id, num_shards=num_gpus, lazy_init=lazy_type
        )
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.pos_rng_x = ops.random.Uniform(range=(0.0, 1.0), seed=1234)
        self.pos_rng_y = ops.random.Uniform(range=(0.0, 1.0), seed=5678)
        self.crop = ops.Crop(device="gpu", crop=(224, 224))

    def define_graph(self):
        self.jpegs, self.labels = self.input()

        pos_x = self.pos_rng_x()
        pos_y = self.pos_rng_y()
        images = self.decode(self.jpegs)
        crop = self.crop(images, crop_pos_x=pos_x, crop_pos_y=pos_y)
        return (crop, self.labels)


def test_lazy_init_empty_data_path():
    empty_db_folder = "/data/empty"
    batch_size = 128

    nonlazy_pipe = LazyPipeline(batch_size, empty_db_folder, lazy_type=False)
    with assert_raises(RuntimeError):
        nonlazy_pipe.build()
    lazy_pipe = LazyPipeline(batch_size, empty_db_folder, lazy_type=True)
    lazy_pipe.build()


def test_lazy_init():
    """
    Comparing results of pipeline: lazy_init false and lazy_init
    true with empty folder and real folder
    """
    batch_size = 128
    compare_pipelines(
        LazyPipeline(batch_size, caffe_db_folder, lazy_type=False),
        LazyPipeline(batch_size, caffe_db_folder, lazy_type=True),
        batch_size=batch_size,
        N_iterations=20,
    )


def test_iter_setup():
    class TestIterator:
        def __init__(self, n):
            self.n = n

        def __iter__(self):
            self.i = 0
            return self

        def __next__(self):
            batch = []
            if self.i < self.n:
                batch.append(np.arange(0, 1, dtype=float))
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
            next(iterator)
            i += 1
        except StopIteration:
            break
    assert iter_num == i

    iterator = iter(TestIterator(iter_num))
    pipe = IterSetupPipeline(iterator, 3, 0)

    i = 0
    while True:
        try:
            pipe.run()
            i += 1
        except StopIteration:
            break
    assert iter_num == i

    pipe.reset()
    i = 0
    while True:
        try:
            pipe.run()
            i += 1
        except StopIteration:
            break
    assert iter_num == i


class CachedPipeline(Pipeline):
    def __init__(
        self,
        reader_type,
        batch_size,
        is_cached=False,
        is_cached_batch_copy=True,
        seed=123456,
        skip_cached_images=False,
        num_shards=30,
    ):
        super(CachedPipeline, self).__init__(
            batch_size, num_threads=1, device_id=0, prefetch_queue_depth=1, seed=seed
        )
        self.reader_type = reader_type
        if reader_type == "readers.MXNet":
            self.input = ops.readers.MXNet(
                path=os.path.join(recordio_db_folder, "train.rec"),
                index_path=os.path.join(recordio_db_folder, "train.idx"),
                shard_id=0,
                num_shards=num_shards,
                stick_to_shard=True,
                skip_cached_images=skip_cached_images,
                prefetch_queue_depth=1,
            )
        elif reader_type == "readers.Caffe":
            self.input = ops.readers.Caffe(
                path=caffe_db_folder,
                shard_id=0,
                num_shards=num_shards,
                stick_to_shard=True,
                skip_cached_images=skip_cached_images,
                prefetch_queue_depth=1,
            )
        elif reader_type == "readers.Caffe2":
            self.input = ops.readers.Caffe2(
                path=c2lmdb_db_folder,
                shard_id=0,
                num_shards=num_shards,
                stick_to_shard=True,
                skip_cached_images=skip_cached_images,
                prefetch_queue_depth=1,
            )
        elif reader_type == "readers.File":
            self.input = ops.readers.File(
                file_root=jpeg_folder,
                shard_id=0,
                num_shards=num_shards,
                stick_to_shard=True,
                skip_cached_images=skip_cached_images,
                prefetch_queue_depth=1,
            )

        elif reader_type == "readers.TFRecord":
            tfrecord = sorted(glob.glob(os.path.join(tfrecord_db_folder, "*[!i][!d][!x]")))
            tfrecord_idx = sorted(glob.glob(os.path.join(tfrecord_db_folder, "*idx")))
            self.input = ops.readers.TFRecord(
                path=tfrecord,
                index_path=tfrecord_idx,
                shard_id=0,
                num_shards=num_shards,
                stick_to_shard=True,
                skip_cached_images=skip_cached_images,
                features={
                    "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                    "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
                },
            )
        elif reader_type == "readers.Webdataset":
            wds = [
                os.path.join(webdataset_db_folder, archive)
                for archive in ["devel-1.tar", "devel-2.tar", "devel-0.tar"]
            ]
            self.wds_index_files = [generate_temp_wds_index(archive) for archive in wds]
            self.input = ops.readers.Webdataset(
                paths=wds,
                index_paths=[idx.name for idx in self.wds_index_files],
                ext=["jpg", "cls"],
                shard_id=0,
                num_shards=num_shards,
                stick_to_shard=True,
                skip_cached_images=skip_cached_images,
            )

        if is_cached:
            self.decode = ops.decoders.Image(
                device="mixed",
                output_type=types.RGB,
                cache_size=2000,
                cache_threshold=0,
                cache_type="threshold",
                cache_debug=False,
                hw_decoder_load=0.0,  # 0.0 for deterministic results
                cache_batch_copy=is_cached_batch_copy,
            )
        else:
            # hw_decoder_load=0.0 for deterministic results
            self.decode = ops.decoders.Image(
                device="mixed", output_type=types.RGB, hw_decoder_load=0.0
            )

    def define_graph(self):
        if self.reader_type == "readers.TFRecord":
            inputs = self.input()
            jpegs = inputs["image/encoded"]
            labels = inputs["image/class/label"]
        else:
            jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)


def test_nvjpeg_cached_batch_copy_pipelines():
    batch_size = 26
    for reader_type in [
        "readers.MXNet",
        "readers.Caffe",
        "readers.Caffe2",
        "readers.File",
        "readers.TFRecord",
        "readers.Webdataset",
    ]:
        compare_pipelines(
            CachedPipeline(reader_type, batch_size, is_cached=True, is_cached_batch_copy=True),
            CachedPipeline(reader_type, batch_size, is_cached=True, is_cached_batch_copy=False),
            batch_size=batch_size,
            N_iterations=20,
        )


def test_nvjpeg_cached_pipelines():
    batch_size = 26
    for reader_type in [
        "readers.MXNet",
        "readers.Caffe",
        "readers.Caffe2",
        "readers.File",
        "readers.TFRecord",
        "readers.Webdataset",
    ]:
        compare_pipelines(
            CachedPipeline(reader_type, batch_size, is_cached=False),
            CachedPipeline(reader_type, batch_size, is_cached=True),
            batch_size=batch_size,
            N_iterations=20,
        )


def test_skip_cached_images():
    batch_size = 1
    for reader_type in [
        "readers.MXNet",
        "readers.Caffe",
        "readers.Caffe2",
        "readers.File",
        "readers.Webdataset",
    ]:
        compare_pipelines(
            CachedPipeline(reader_type, batch_size, is_cached=False),
            CachedPipeline(reader_type, batch_size, is_cached=True, skip_cached_images=True),
            batch_size=batch_size,
            N_iterations=100,
        )


def test_caffe_no_label():
    class CaffePipeline(Pipeline):
        def __init__(
            self,
            batch_size,
            path_to_data,
            labels,
            seed=123456,
            skip_cached_images=False,
            num_shards=1,
        ):
            super(CaffePipeline, self).__init__(
                batch_size, num_threads=1, device_id=0, prefetch_queue_depth=1, seed=seed
            )
            self.input = ops.readers.Caffe(
                path=path_to_data,
                shard_id=0,
                num_shards=num_shards,
                stick_to_shard=True,
                prefetch_queue_depth=1,
                label_available=labels,
            )
            self.decode = ops.decoders.Image(output_type=types.RGB)
            self.labels = labels

        def define_graph(self):
            if not self.labels:
                jpegs = self.input()
            else:
                jpegs, _ = self.input()
            images = self.decode(jpegs)
            return images

    pipe = CaffePipeline(2, caffe_db_folder, True)
    pipe.run()
    pipe = CaffePipeline(2, caffe_no_label_db_folder, False)
    pipe.run()


def test_caffe2_no_label():
    class Caffe2Pipeline(Pipeline):
        def __init__(
            self,
            batch_size,
            path_to_data,
            label_type,
            seed=123456,
            skip_cached_images=False,
            num_shards=1,
        ):
            super(Caffe2Pipeline, self).__init__(
                batch_size, num_threads=1, device_id=0, prefetch_queue_depth=1, seed=seed
            )
            self.input = ops.readers.Caffe2(
                path=path_to_data,
                shard_id=0,
                num_shards=num_shards,
                stick_to_shard=True,
                prefetch_queue_depth=1,
                label_type=label_type,
            )
            self.decode = ops.decoders.Image(output_type=types.RGB)
            self.label_type = label_type

        def define_graph(self):
            if self.label_type == 4:
                jpegs = self.input()
            else:
                jpegs, _ = self.input()
            images = self.decode(jpegs)
            return images

    pipe = Caffe2Pipeline(2, c2lmdb_db_folder, 0)
    pipe.run()
    pipe = Caffe2Pipeline(2, c2lmdb_no_label_db_folder, 4)
    pipe.run()


def test_as_tensor():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
            self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=True)

        def define_graph(self):
            _, self.labels = self.input()
            return self.labels

    batch_size = 8
    shape = [[2, 2, 2], [8, 1], [1, 8], [4, 2], [2, 4], [8], [1, 2, 1, 2, 1, 2], [1, 1, 1, 8]]
    pipe = HybridPipe(batch_size=batch_size, num_threads=2, device_id=0)
    for sh in shape:
        pipe_out = pipe.run()[0]
        assert pipe_out.as_tensor().shape() == [batch_size, 1]
        assert pipe_out.as_reshaped_tensor(sh).shape() == sh
        different_shape = random.choice(shape)
        assert pipe_out.as_reshaped_tensor(different_shape).shape() == different_shape


def test_as_tensor_fail():
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
            self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=True)

        def define_graph(self):
            _, self.labels = self.input()
            return self.labels

    batch_size = 8
    shape = [
        [2, 2, 2, 3],
        [8, 1, 6],
        [1, 8, 4],
        [4, 2, 9],
        [2, 4, 0],
        [8, 2],
        [1, 2, 1, 2, 1, 2, 3],
        [7, 1, 1, 1, 8],
    ]
    pipe = HybridPipe(batch_size=batch_size, num_threads=2, device_id=0)
    for sh in shape:
        pipe_out = pipe.run()[0]
        assert pipe_out.as_tensor().shape() == [batch_size, 1]
        with assert_raises(RuntimeError):
            pipe_out.as_reshaped_tensor(sh).shape()


def test_python_formats():
    class TestPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, test_array):
            super(TestPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input_data = ops.ExternalSource()
            self.test_array = test_array

        def define_graph(self):
            self.data = self.input_data()
            return self.data

        def iter_setup(self):
            self.feed_input(self.data, self.test_array)

    for t in [
        np.bool_,
        np.int_,
        np.intc,
        np.intp,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float16,
        np.short,
        int,
        np.longlong,
        np.ushort,
        np.ulonglong,
    ]:
        test_array = np.array([[1, 1], [1, 1]], dtype=t)
        pipe = TestPipeline(2, 1, 0, 1, test_array)
        out = pipe.run()[0]
        out_dtype = out.at(0).dtype
        assert test_array.dtype.itemsize == out_dtype.itemsize
        assert test_array.dtype.str == out_dtype.str


def test_api_check1():
    batch_size = 1

    class TestPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(TestPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
            )

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            return inputs

    pipe = TestPipeline(batch_size=batch_size, num_threads=1, device_id=0, num_gpus=1)
    pipe.run()
    for method in [pipe.schedule_run, pipe.share_outputs, pipe.release_outputs, pipe.outputs]:
        with assert_raises(
            RuntimeError,
            glob=(
                "Mixing pipeline API type. Currently used: PipelineAPIType.BASIC,"
                " but trying to use PipelineAPIType.SCHEDULED"
            ),
        ):
            method()
    # disable check
    pipe.enable_api_check(False)
    for method in [pipe.schedule_run, pipe.share_outputs, pipe.release_outputs, pipe.outputs]:
        method()


def test_api_check2():
    batch_size = 1

    class TestPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus):
            super(TestPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
            )

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            return inputs

    pipe = TestPipeline(batch_size=batch_size, num_threads=1, device_id=0, num_gpus=1)
    pipe.schedule_run()
    pipe.share_outputs()
    pipe.release_outputs()
    pipe.schedule_run()
    pipe.outputs()
    with assert_raises(
        RuntimeError,
        glob=(
            "Mixing pipeline API type. Currently used: PipelineAPIType.SCHEDULED,"
            " but trying to use PipelineAPIType.BASIC"
        ),
    ):
        pipe.run()
    # disable check
    pipe.enable_api_check(False)
    pipe.run()


class DupPipeline(Pipeline):
    def __init__(
        self, batch_size, num_threads, device_id, first_out_device="cpu", second_out_device="cpu"
    ):
        super(DupPipeline, self).__init__(batch_size, num_threads, device_id)
        self.first_out_device = first_out_device
        self.second_out_device = second_out_device
        self.input = ops.readers.Caffe(path=caffe_db_folder, shard_id=device_id, num_shards=1)
        self.decode = ops.decoders.Image(
            device="mixed" if first_out_device == "mixed" else "cpu", output_type=types.RGB
        )
        if self.second_out_device:
            self.cmnp = ops.CropMirrorNormalize(
                device=second_out_device,
                dtype=types.FLOAT,
                output_layout=types.NHWC,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )

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
    pipe = DupPipeline(
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
        first_out_device=first_device,
        second_out_device=second_device,
    )
    out = pipe.run()
    assert len(out) == 4
    for i in range(batch_size):
        assert isinstance(out[3][0], dali.backend_impl.TensorGPU) or first_device == "cpu"
        out1 = as_array(out[0][i])
        out2 = as_array(out[1][i])
        out3 = as_array(out[2][i])

        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_array_equal(out1, out3)


def test_duplicated_outs_pipeline():
    for first_device, second_device in [
        ("cpu", None),
        ("cpu", "cpu"),
        ("cpu", "gpu"),
        ("mixed", None),
        ("mixed", "gpu"),
    ]:
        yield check_duplicated_outs_pipeline, first_device, second_device


def check_serialized_outs_duplicated_pipeline(first_device, second_device):
    batch_size = 5
    pipe = DupPipeline(
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
        first_out_device=first_device,
        second_out_device=second_device,
    )
    serialized_pipeline = pipe.serialize()
    del pipe
    new_pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    new_pipe.deserialize_and_build(serialized_pipeline)
    out = new_pipe.run()
    assert len(out) == 4
    for i in range(batch_size):
        assert isinstance(out[3][0], dali.backend_impl.TensorGPU) or first_device == "cpu"
        out1 = as_array(out[0][i])
        out2 = as_array(out[1][i])
        out3 = as_array(out[2][i])

        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_array_equal(out1, out3)


def test_serialized_outs_duplicated_pipeline():
    for first_device, second_device in [
        ("cpu", None),
        ("cpu", "cpu"),
        ("cpu", "gpu"),
        ("mixed", None),
        ("mixed", "gpu"),
    ]:
        yield check_serialized_outs_duplicated_pipeline, first_device, second_device


def check_duplicated_outs_cpu_to_gpu(device):
    class SliceArgsIterator(object):
        def __init__(
            self,
            batch_size,
            num_dims=3,
            image_shape=None,  # Needed if normalized_anchor and normalized_shape are False
            image_layout=None,  # Needed if axis_names is used to specify the slice
            normalized_anchor=True,
            normalized_shape=True,
            axes=None,
            axis_names=None,
            min_norm_anchor=0.0,
            max_norm_anchor=0.2,
            min_norm_shape=0.4,
            max_norm_shape=0.75,
            seed=54643613,
        ):
            self.batch_size = batch_size
            self.num_dims = num_dims
            self.image_shape = image_shape
            self.image_layout = image_layout
            self.normalized_anchor = normalized_anchor
            self.normalized_shape = normalized_shape
            self.axes = axes
            self.axis_names = axis_names
            self.min_norm_anchor = min_norm_anchor
            self.max_norm_anchor = max_norm_anchor
            self.min_norm_shape = min_norm_shape
            self.max_norm_shape = max_norm_shape
            self.seed = seed

            if not self.axis_names and not self.axes:
                self.axis_names = "WH"

            if self.axis_names:
                self.axes = []
                for axis_name in self.axis_names:
                    assert axis_name in self.image_layout
                    self.axes.append(self.image_layout.index(axis_name))
            assert len(self.axes) > 0

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
                    anchor = [
                        floor(norm_anchor[i] * self.image_shape[self.axes[i]])
                        for i in range(len(self.axes))
                    ]

                if self.normalized_shape:
                    shape = norm_shape
                else:
                    shape = [
                        floor(norm_shape[i] * self.image_shape[self.axes[i]])
                        for i in range(len(self.axes))
                    ]

                pos.append(np.asarray(anchor, dtype=np.float32))
                size.append(np.asarray(shape, dtype=np.float32))
                self.i = (self.i + 1) % self.n
            return (pos, size)

        next = __next__

    class SliceSynthDataPipeline(Pipeline):
        def __init__(
            self,
            device,
            batch_size,
            layout,
            iterator,
            pos_size_iter,
            num_threads=1,
            device_id=0,
            num_gpus=1,
            axes=None,
            axis_names=None,
            normalized_anchor=True,
            normalized_shape=True,
        ):
            super(SliceSynthDataPipeline, self).__init__(
                batch_size, num_threads, device_id, seed=1234
            )
            self.device = device
            self.layout = layout
            self.iterator = iterator
            self.pos_size_iter = pos_size_iter
            self.inputs = ops.ExternalSource()
            self.input_crop_pos = ops.ExternalSource()
            self.input_crop_size = ops.ExternalSource()

            if axis_names:
                self.slice = ops.Slice(
                    device=self.device,
                    normalized_anchor=normalized_anchor,
                    normalized_shape=normalized_shape,
                    axis_names=axis_names,
                )
            elif axes:
                self.slice = ops.Slice(
                    device=self.device,
                    normalized_anchor=normalized_anchor,
                    normalized_shape=normalized_shape,
                    axes=axes,
                )
            else:
                self.slice = ops.Slice(
                    device=self.device,
                    normalized_anchor=normalized_anchor,
                    normalized_shape=normalized_shape,
                )

        def define_graph(self):
            self.data = self.inputs()
            self.crop_pos = self.input_crop_pos()
            self.crop_size = self.input_crop_size()
            data = self.data.gpu() if self.device == "gpu" else self.data
            out = self.slice(data, self.crop_pos, self.crop_size)
            return out, self.crop_pos, self.crop_size

        def iter_setup(self):
            data = self.iterator.next()
            self.feed_input(self.data, data, layout=self.layout)

            (crop_pos, crop_size) = self.pos_size_iter.next()
            self.feed_input(self.crop_pos, crop_pos)
            self.feed_input(self.crop_size, crop_size)

    batch_size = 1
    input_shape = (200, 400, 3)
    layout = "HWC"
    axes = None
    axis_names = "WH"
    normalized_anchor = False
    normalized_shape = False
    eiis = [RandomDataIterator(batch_size, shape=input_shape) for k in range(2)]
    eii_args = [
        SliceArgsIterator(
            batch_size,
            len(input_shape),
            image_shape=input_shape,
            image_layout=layout,
            axes=axes,
            axis_names=axis_names,
            normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape,
        )
        for k in range(2)
    ]

    pipe = SliceSynthDataPipeline(
        device,
        batch_size,
        layout,
        iter(eiis[0]),
        iter(eii_args[0]),
        axes=axes,
        axis_names=axis_names,
        normalized_anchor=normalized_anchor,
        normalized_shape=normalized_shape,
    )
    out = pipe.run()
    assert isinstance(out[0][0], dali.backend_impl.TensorGPU) or device == "cpu"
    assert not isinstance(out[1][0], dali.backend_impl.TensorGPU)
    assert not isinstance(out[2][0], dali.backend_impl.TensorGPU)


def test_duplicated_outs_cpu_op_to_gpu():
    # check if it is possible to return outputs from CPU op that goes directly to the GPU op without
    # MakeContiguous as a CPU output from the pipeline
    for device in ["cpu", "gpu"]:
        yield check_duplicated_outs_cpu_to_gpu, device


def test_ref_count():
    class HybridPipe(Pipeline):
        def __init__(self):
            super(HybridPipe, self).__init__(1, 1, 0, seed=12)
            self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=True)

        def define_graph(self):
            _, self.labels = self.input()
            return self.labels

    pipe = HybridPipe()
    assert sys.getrefcount(pipe) == 2
    assert sys.getrefcount(pipe) == 2


def test_executor_meta():
    class TestPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, seed):
            super(TestPipeline, self).__init__(
                batch_size,
                num_threads,
                device_id,
                enable_memory_stats=True,
                exec_async=False,
                exec_pipelined=False,
            )
            self.input = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus, seed=seed
            )
            self.decode = ops.decoders.ImageRandomCrop(
                device="mixed", output_type=types.RGB, seed=seed
            )
            self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224)
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                crop=(224, 224),
                mean=[128.0, 128.0, 128.0],
                std=[1.0, 1.0, 1.0],
            )
            self.coin = ops.random.CoinFlip(seed=seed)

        def define_graph(self):
            self.jpegs, self.labels = self.input()
            images = self.decode(self.jpegs)
            resized_images = self.res(images)
            mirror = self.coin()
            output = self.cmnp(resized_images, mirror=mirror)
            return (output, resized_images, self.labels)

    random_seed = 123456
    batch_size = 10
    test_pipe = TestPipeline(
        batch_size=batch_size, num_threads=1, device_id=0, num_gpus=1, seed=random_seed
    )
    test_pipe.run()
    meta = test_pipe.executor_statistics()
    # all operators (readers.Caffe, decoders.ImageRandomCrop, Resize, CropMirrorNormalize,
    # CoinFlip) + make_contiguous * 3 (all outputs)
    assert len(meta) == 8
    for k in meta.keys():
        if "CropMirrorNormalize" in k:
            crop_meta = meta[k]
    assert crop_meta["real_memory_size"] == crop_meta["reserved_memory_size"]
    # size of crop * num_of_channels * batch_size * data_size
    assert crop_meta["real_memory_size"][0] == 224 * 224 * 3 * batch_size * 4
    for k in meta.keys():
        if "CoinFlip" in k:
            coin_meta = meta[k]
    assert coin_meta["real_memory_size"] == coin_meta["reserved_memory_size"]
    # batch_size * data_size
    assert coin_meta["real_memory_size"][0] == batch_size * 4
    for k, v in meta.items():
        assert v["real_memory_size"] <= v["reserved_memory_size"]

        def calc_avg_max(val):
            return [int(ceil(v / batch_size)) for v in val]

        # for CPU the biggest tensor is usually bigger than the average,
        # for the GPU max is the average
        if "CPU" in k or "MIXED" in k:
            assert calc_avg_max(v["real_memory_size"]) <= v["max_real_memory_size"]
            assert calc_avg_max(v["reserved_memory_size"]) <= v["max_reserved_memory_size"]
        else:
            assert calc_avg_max(v["real_memory_size"]) == v["max_real_memory_size"]
            assert calc_avg_max(v["reserved_memory_size"]) == v["max_reserved_memory_size"]


def test_bytes_per_sample_hint():
    import nvidia.dali.backend

    if nvidia.dali.backend.RestrictPinnedMemUsage():
        raise SkipTest
    nvidia.dali.backend.SetHostBufferShrinkThreshold(0)

    def obtain_reader_meta(iters=3, **kvargs):
        batch_size = 10
        pipe = Pipeline(
            batch_size, 1, 0, exec_async=False, exec_pipelined=False, enable_memory_stats=True
        )
        with pipe:
            out = fn.readers.caffe(path=caffe_db_folder, shard_id=0, num_shards=1, **kvargs)
            out = [o.gpu() for o in out]
            pipe.set_outputs(*out)
        for _ in range(iters):
            pipe.run()
        meta = pipe.executor_statistics()
        reader_meta = None
        for k in meta.keys():
            if "CPU___Caffe" in k:
                reader_meta = meta[k]
        return reader_meta

    reader_meta = obtain_reader_meta(iters=10)
    new_reader_meta = obtain_reader_meta(
        iters=1,
        bytes_per_sample_hint=[int(v * 1.1) for v in reader_meta["max_reserved_memory_size"]],
    )
    assert new_reader_meta["max_reserved_memory_size"] > reader_meta["max_reserved_memory_size"]


def trigger_output_dtype_deprecated_warning():
    batch_size = 10
    shape = (120, 60, 3)
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    data = RandomDataIterator(batch_size, shape=shape, dtype=np.uint8)
    with pipe:
        input = fn.external_source(data, layout="HWC")
        cmn = fn.crop_mirror_normalize(
            input,
            device="cpu",
            output_dtype=types.FLOAT,
            output_layout="HWC",
            crop=(32, 32),
            mean=[128.0, 128.0, 128.0],
            std=[1.0, 1.0, 1.0],
        )
        pipe.set_outputs(cmn)

    (result,) = pipe.run()
    assert result.as_array().dtype == np.float32


def trigger_image_type_deprecated_warning():
    batch_size = 10
    shape = (120, 60, 3)
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    data = RandomDataIterator(batch_size, shape=shape, dtype=np.uint8)
    with pipe:
        input = fn.external_source(data, layout="HWC")
        cmn = fn.crop_mirror_normalize(
            input,
            device="cpu",
            dtype=types.FLOAT,
            image_type=types.RGB,
            output_layout="HWC",
            crop=(32, 32),
            mean=[128.0, 128.0, 128.0],
            std=[1.0, 1.0, 1.0],
        )
        pipe.set_outputs(cmn)

    (result,) = pipe.run()
    assert result.as_array().dtype == np.float32


def test_output_dtype_deprecation():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        trigger_output_dtype_deprecated_warning()
        # Verify DeprecationWarning
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        expected_msg = (
            "The argument `output_dtype` is a deprecated alias for `dtype`. Use `dtype` instead."
        )
        assert expected_msg == str(w[-1].message)


def test_image_type_deprecation():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        trigger_image_type_deprecated_warning()
        # Verify DeprecationWarning
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        expected_msg = (
            "The argument `image_type` is no longer used and will be removed "
            "in a future release."
        )
        assert expected_msg == str(w[-1].message)


@raises(TypeError, glob="unexpected*output_dtype*dtype")
def test_output_dtype_both_error():
    batch_size = 10
    shape = (120, 60, 3)
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    data = RandomDataIterator(batch_size, shape=shape, dtype=np.uint8)
    with pipe:
        input = fn.external_source(data, layout="HWC")
        cmn = fn.crop_mirror_normalize(
            input,
            device="cpu",
            output_dtype=types.FLOAT,
            dtype=types.FLOAT,
            output_layout="HWC",
            crop=(32, 32),
            mean=[128.0, 128.0, 128.0],
            std=[1.0, 1.0, 1.0],
        )
        pipe.set_outputs(cmn)


def test_epoch_size():
    class ReaderPipeline(Pipeline):
        def __init__(self, batch_size):
            super(ReaderPipeline, self).__init__(
                batch_size, num_threads=1, device_id=0, prefetch_queue_depth=1
            )
            self.input_mxnet = ops.readers.MXNet(
                path=os.path.join(recordio_db_folder, "train.rec"),
                index_path=os.path.join(recordio_db_folder, "train.idx"),
                shard_id=0,
                num_shards=1,
                prefetch_queue_depth=1,
            )
            self.input_caffe = ops.readers.Caffe(
                path=caffe_db_folder, shard_id=0, num_shards=1, prefetch_queue_depth=1
            )
            self.input_caffe2 = ops.readers.Caffe2(
                path=c2lmdb_db_folder, shard_id=0, num_shards=1, prefetch_queue_depth=1
            )
            self.input_file = ops.readers.File(
                file_root=jpeg_folder, shard_id=0, num_shards=1, prefetch_queue_depth=1
            )

        def define_graph(self):
            jpegs_mxnet, _ = self.input_mxnet(name="readers.mxnet")
            jpegs_caffe, _ = self.input_caffe(name="readers.caffe")
            jpegs_caffe2, _ = self.input_caffe2(name="readers.caffe2")
            jpegs_file, _ = self.input_file(name="readers.file")
            return jpegs_mxnet, jpegs_caffe, jpegs_caffe2, jpegs_file

    pipe = ReaderPipeline(1)
    meta = pipe.reader_meta()
    assert len(meta) == 4
    assert pipe.epoch_size("readers.mxnet") != 0
    assert pipe.epoch_size("readers.caffe") != 0
    assert pipe.epoch_size("readers.caffe2") != 0
    assert pipe.epoch_size("readers.file") != 0
    assert len(pipe.epoch_size()) == 4


def test_pipeline_out_of_scope():
    def get_output():
        pipe = dali.Pipeline(1, 1, 0)
        with pipe:
            pipe.set_outputs(dali.fn.external_source(source=[[np.array([-0.5, 1.25])]]))
        return pipe.run()

    out = get_output()[0].at(0)
    assert out[0] == -0.5 and out[1] == 1.25


def test_return_constants():
    pipe = dali.Pipeline(1, 1, None)
    types = [bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float32]
    pipe.set_outputs(np.array([[1, 2], [3, 4]]), 10, *[t(42) for t in types])
    a, b, *other = pipe.run()
    assert np.array_equal(a.at(0), np.array([[1, 2], [3, 4]]))
    assert b.at(0) == 10
    for i, o in enumerate(other):
        assert o.at(0) == types[i](42)
        assert o.at(0).dtype == types[i]


def test_preserve_arg():
    pipe = dali.Pipeline(1, 1, 0)
    with pipe:
        out = dali.fn.external_source(source=[[np.array([-0.5, 1.25])]], preserve=True)
        res = dali.fn.resize(out, preserve=True)  # noqa: F841
        pipe.set_outputs(out)


def test_pipeline_wrong_device_id():
    pipe = dali.Pipeline(batch_size=1, num_threads=1, device_id=-123)
    with pipe:
        pipe.set_outputs(np.int32([1, 2, 3]))
    with assert_raises(Exception, regex="(wrong device_id)|(device_id.*is invalid)"):
        pipe.run()


def test_properties():
    @dali.pipeline_def(batch_size=11, prefetch_queue_depth={"cpu_size": 3, "gpu_size": 2})
    def my_pipe():
        pipe = Pipeline.current()
        assert pipe.max_batch_size == 11
        assert pipe.batch_size == 11
        assert pipe.num_threads == 3
        assert pipe.device_id == 0
        assert pipe.seed == 1234
        assert pipe.exec_pipelined is True
        assert pipe.exec_async is True
        assert pipe.set_affinity is True
        assert pipe.prefetch_queue_depth == {"cpu_size": 3, "gpu_size": 2}
        assert pipe.cpu_queue_size == 3
        assert pipe.gpu_queue_size == 2
        assert pipe.py_num_workers == 3
        assert pipe.py_start_method == "fork"
        assert pipe.enable_memory_stats is False
        return np.float32([1, 2, 3])

    my_pipe(device_id=0, seed=1234, num_threads=3, set_affinity=True, py_num_workers=3)


def test_not_iterable():
    import nvidia.dali._utils.hacks as hacks
    import collections.abc

    class X:
        def __iter__(self):
            pass

    class Y:
        def __iter__(self):
            pass

    assert isinstance(X(), collections.abc.Iterable)
    hacks.not_iterable(X)
    assert not isinstance(X(), collections.abc.Iterable)
    assert isinstance(Y(), collections.abc.Iterable)
    hacks.not_iterable(Y)
    assert not isinstance(Y(), collections.abc.Iterable)


@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def _identity_pipe():
    x = fn.external_source(device="gpu", name="identity_input")
    return x


@raises(TypeError, "*define_graph*callable*")
def test_invoke_serialize_error_handling_string():
    _identity_pipe().serialize("any_string")


@raises(TypeError, "*define_graph*callable*")
def test_invoke_serialize_error_handling_not_string():
    _identity_pipe().serialize(42)


def check_dtype_ndim(dali_pipeline, output_dtype, output_ndim, n_outputs):
    def ndim_dtype_matches(test_value, ref_value):
        ref_value = ref_value if isinstance(ref_value, (list, tuple)) else [ref_value] * n_outputs
        return ref_value == test_value

    import tempfile

    with tempfile.NamedTemporaryFile() as f:
        dali_pipeline.serialize(filename=f.name)
        deserialized_pipeline = Pipeline.deserialize(filename=f.name)
        assert ndim_dtype_matches(
            deserialized_pipeline.output_ndim(), output_ndim
        ), f"`output_ndim` is not serialized properly. {deserialized_pipeline.output_ndim()} vs {output_ndim}."  # noqa: E501
        assert ndim_dtype_matches(
            deserialized_pipeline.output_dtype(), output_dtype
        ), f"`output_dtype` is not serialized properly. {deserialized_pipeline.output_dtype()} vs {output_dtype}."  # noqa: E501
        deserialized_pipeline.run()
    dali_pipeline.run()


@raises(RuntimeError, glob="Data type * does not match*")
def check_dtype_with_raise(dali_pipeline, output_dtype, output_ndim, n_outputs):
    check_dtype_ndim(dali_pipeline, output_dtype, output_ndim, n_outputs)


@raises(RuntimeError, glob="Number of dimensions * does not match*")
def check_ndim_with_raise(dali_pipeline, output_dtype, output_ndim, n_outputs):
    check_dtype_ndim(dali_pipeline, output_dtype, output_ndim, n_outputs)


@raises(RuntimeError, glob="Lengths * do not match*")
def check_length_error(dali_pipeline, output_dtype, output_ndim, n_outputs):
    check_dtype_ndim(dali_pipeline, output_dtype, output_ndim, n_outputs)


def test_one_output_dtype_ndim():
    @pipeline_def
    def pipe():
        inputs, labels = fn.readers.file(
            file_root=os.path.join(get_dali_extra_path(), "db", "single", "jpeg"), name="Reader"
        )
        decoded = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
        return decoded

    def create_test_package(output_dtype=None, output_ndim=None):
        return (
            pipe(
                batch_size=1,
                num_threads=1,
                device_id=0,
                output_dtype=output_dtype,
                output_ndim=output_ndim,
            ),
            output_dtype,
            output_ndim,
        )

    both_correct = create_test_package(output_dtype=[types.UINT8], output_ndim=[3])
    ndim_correct_dtype_wildcard = create_test_package(output_dtype=[None], output_ndim=[3])
    dtype_correct_ndim_wildcard = create_test_package(output_dtype=types.UINT8)
    dtype_incorrect = create_test_package(output_dtype=[types.FLOAT], output_ndim=[3])
    ndim_incorrect = create_test_package(output_dtype=types.UINT8, output_ndim=0)
    both_correct_one_list = create_test_package(output_dtype=types.UINT8, output_ndim=[3])
    too_many_dtypes = create_test_package(output_dtype=[types.UINT8, types.FLOAT])
    correct_dtypes_but_too_many = create_test_package(output_dtype=[types.UINT8, types.UINT8])
    correct_ndims_but_too_many = create_test_package(output_ndim=[3, 3])
    all_wildcards = create_test_package()
    correct_test_packages = [
        both_correct,
        ndim_correct_dtype_wildcard,
        dtype_correct_ndim_wildcard,
        both_correct_one_list,
        all_wildcards,
    ]
    test_ndim_packages_with_raise = [ndim_incorrect]
    test_dtype_packages_with_raise = [dtype_incorrect]
    test_packages_length_mismatch = [
        too_many_dtypes,
        correct_dtypes_but_too_many,
        correct_ndims_but_too_many,
    ]

    for pipe_under_test, dtype, ndim in correct_test_packages:
        yield check_dtype_ndim, pipe_under_test, dtype, ndim, 1
    for pipe_under_test, dtype, ndim in test_ndim_packages_with_raise:
        yield check_ndim_with_raise, pipe_under_test, dtype, ndim, 1
    for pipe_under_test, dtype, ndim in test_dtype_packages_with_raise:
        yield check_dtype_with_raise, pipe_under_test, dtype, ndim, 1
    for pipe_under_test, dtype, ndim in test_packages_length_mismatch:
        yield check_length_error, pipe_under_test, dtype, ndim, 1


def test_double_output_dtype_ndim():
    @pipeline_def
    def pipe(cast_labels):
        inputs, labels = fn.readers.file(
            file_root=os.path.join(get_dali_extra_path(), "db", "single", "jpeg"), name="Reader"
        )
        decoded = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
        labels_casted = fn.cast(labels, dtype=types.UINT8)
        return decoded, labels_casted if cast_labels else labels

    def create_test_package(output_dtype=None, output_ndim=None, cast_labels=False):
        return (
            pipe(
                batch_size=1,
                num_threads=1,
                device_id=0,
                output_dtype=output_dtype,
                output_ndim=output_ndim,
                cast_labels=cast_labels,
            ),
            output_dtype,
            output_ndim,
        )

    both_correct = create_test_package(output_dtype=[types.UINT8, types.INT32], output_ndim=[3, 1])
    ndim_correct_dtype_wildcard = create_test_package(output_dtype=[None, None], output_ndim=[3, 1])
    dtype_correct_ndim_wildcard = create_test_package(
        output_dtype=[types.UINT8, types.UINT8], cast_labels=True
    )
    dtype_incorrect = create_test_package(output_dtype=[types.UINT8, types.FLOAT])
    ndim_incorrect = create_test_package(output_ndim=[3, 3])
    dtype_broadcast = create_test_package(output_dtype=types.UINT8, cast_labels=True)
    wildcard_in_dtype = create_test_package(output_dtype=[types.UINT8, None])
    wildcard_in_ndim = create_test_package(output_ndim=[3, None])
    not_enough_dtypes = create_test_package(output_dtype=[types.UINT8])
    not_enough_ndim = create_test_package(output_ndim=[1])
    all_wildcards = create_test_package()
    all_wildcards_but_shapes_dont_match = create_test_package(
        output_dtype=[None, None], output_ndim=[None]
    )
    correct_test_packages = [
        both_correct,
        ndim_correct_dtype_wildcard,
        dtype_correct_ndim_wildcard,
        dtype_broadcast,
        wildcard_in_dtype,
        wildcard_in_ndim,
        all_wildcards,
    ]
    test_ndim_packages_with_raise = [ndim_incorrect]
    test_dtype_packages_with_raise = [dtype_incorrect]
    test_packages_length_mismatch = [
        not_enough_ndim,
        not_enough_dtypes,
        all_wildcards_but_shapes_dont_match,
    ]

    for pipe_under_test, dtype, ndim in correct_test_packages:
        yield check_dtype_ndim, pipe_under_test, dtype, ndim, 2
    for pipe_under_test, dtype, ndim in test_ndim_packages_with_raise:
        yield check_ndim_with_raise, pipe_under_test, dtype, ndim, 2
    for pipe_under_test, dtype, ndim in test_dtype_packages_with_raise:
        yield check_dtype_with_raise, pipe_under_test, dtype, ndim, 2
    for pipe_under_test, dtype, ndim in test_packages_length_mismatch:
        yield check_length_error, pipe_under_test, dtype, ndim, 2
    with assert_raises(ValueError, glob="*must be non-negative*"):
        create_test_package(output_ndim=-1)
        create_test_package(output_ndim=-2137)
    with assert_raises(TypeError, glob="*must be either*"):
        create_test_package(output_dtype=int)
    with assert_raises(ValueError, glob="*types.NO_TYPE*"):
        create_test_package(output_dtype=types.NO_TYPE)


def test_dangling_subgraph():
    # This test ensures that operators defined outside of the pipeline are assigned
    # same ids when the pipeline is built.

    pipes = []
    op1 = fn.external_source(
        source=[np.int32([1, 2, 3]), np.int32([4, 5, 6])], cycle=True, batch=False
    )
    op2 = fn.external_source(
        source=[np.int32([6, 5, 4]), np.int32([3, 2, 1])], cycle=True, batch=False
    )
    for i in range(2):
        with Pipeline(batch_size=1, device_id=None, num_threads=1, seed=123) as p:
            ret1 = op1 + op2
            p.set_outputs(ret1)
        pipes.append(p)

    pipes[0].build()  # names and ids of op1 and op2 are adjusted here
    pipes[1].build()  # names and ids of op3 and op4 are adjusted here

    ser1 = pipes[0].serialize()
    ser2 = pipes[1].serialize()
    assert ser1 == ser2

    (o1,) = pipes[0].run()
    (o2,) = pipes[1].run()
    assert np.array_equal(o1[0], np.int32([7, 7, 7]))
    assert np.array_equal(o2[0], np.int32([7, 7, 7]))


def test_equal_serialized_pipelines_without_explicit_seed():
    # This test ensures that pipelines with the same operators but no explicit random seeds
    # are serialized to the same protobuf.

    pipes = []
    for i in range(2):
        with Pipeline(batch_size=1, device_id=None, num_threads=1) as p:
            p.set_outputs(fn.random.uniform() * fn.random.uniform() - 100)
        pipes.append(p)

    pipes[0].build()
    pipes[1].build()

    ser1 = pipes[0].serialize()
    ser2 = pipes[1].serialize()
    assert ser1 == ser2


def test_regression_without_current_pipeline1():
    def get_pipe(device):
        pipe = Pipeline(batch_size=1, num_threads=1, device_id=0)
        data = fn.external_source(source=[1, 2, 3], batch=False, cycle=True, device=device)
        dist = data + fn.random.normal()
        pipe.set_outputs(dist)
        return pipe

    p = get_pipe("gpu")
    p.build()


def test_regression_without_current_pipeline2():
    pipe = Pipeline(batch_size=4, num_threads=3, device_id=0)
    data = fn.external_source(source=[1, 2, 3], batch=False, cycle=True)
    pipe.set_outputs(data.gpu())


def test_subgraph_stealing():
    p1 = Pipeline(batch_size=1, device_id=None, num_threads=1)
    p2 = Pipeline(batch_size=1, device_id=None, num_threads=1)
    with p1:
        es1 = fn.external_source(source=[1, 2, 3], batch=False)
        x = es1 + 1
        p1.set_outputs(x)
    with p2:
        es2 = fn.external_source(source=[1, 2, 3], batch=False)
        p2.set_outputs(x + es2)
    with assert_raises(
        RuntimeError,
        glob="The pipeline is invalid because it contains operators with non-unique names",
    ):
        p2.build()


def test_gpu2cpu():
    bs = 8

    @pipeline_def(batch_size=bs, num_threads=4, device_id=0, exec_dynamic=True)
    def pdef():
        enc, _ = fn.readers.file(file_root=jpeg_folder)
        img = fn.decoders.image(enc, device="mixed")
        return img, img.cpu()

    pipe = pdef()
    for i in range(10):
        gpu, cpu = pipe.run()
        assert isinstance(gpu, dali.backend_impl.TensorListGPU)
        assert isinstance(cpu, dali.backend_impl.TensorListCPU)
        check_batch(cpu, gpu, bs, 0, 0, "HWC")


def test_gpu2cpu_arg_input():
    @pipeline_def(batch_size=1, num_threads=4, device_id=0, exec_dynamic=True)
    def pdef():
        data = dali.types.Constant([42], device="gpu")
        resized = fn.zeros(shape=data.cpu(), dtype=types.INT32)
        return resized

    pipe = pdef()
    (o,) = pipe.run()
    assert o[0].shape() == [42]


def test_gpu2cpu2mixed():
    bs = 8

    @pipeline_def(batch_size=bs, num_threads=4, device_id=0, exec_dynamic=True)
    def pdef():
        enc, _ = fn.readers.file(file_root=jpeg_folder)
        img = fn.decoders.image(enc, device="mixed", hw_decoder_load=0)
        enc2 = (enc.gpu() + np.uint8(0)).cpu()
        img2 = fn.decoders.image(enc2, device="mixed", hw_decoder_load=0)
        return img, img2

    pipe = pdef()
    for i in range(10):
        gpu, gpu2 = pipe.run()
        assert isinstance(gpu, dali.backend_impl.TensorListGPU)
        assert isinstance(gpu2, dali.backend_impl.TensorListGPU)
        check_batch(gpu, gpu2, bs, 0, 0, "HWC")


def test_shapes_gpu():
    bs = 8

    @pipeline_def(batch_size=bs, num_threads=4, device_id=0, exec_dynamic=True)
    def pdef():
        enc, _ = fn.readers.file(file_root=jpeg_folder)
        img = fn.decoders.image(enc, device="mixed")
        peek = fn.peek_image_shape(enc)
        shapes_of_gpu = fn._shape(img, device="cpu")
        shapes_of_cpu = fn._shape(img.cpu())
        return peek, shapes_of_gpu, shapes_of_cpu, img.shape(), img.cpu().shape()

    pipe = pdef()
    for i in range(10):
        peek, shape_of_gpu, shape_of_cpu, shape_func_gpu, shape_func_cpu = pipe.run()
        # all results must be CPU tensor lists
        assert isinstance(peek, dali.backend_impl.TensorListCPU)
        assert isinstance(shape_of_gpu, dali.backend_impl.TensorListCPU)
        assert isinstance(shape_of_cpu, dali.backend_impl.TensorListCPU)
        assert isinstance(shape_func_gpu, dali.backend_impl.TensorListCPU)
        assert isinstance(shape_func_cpu, dali.backend_impl.TensorListCPU)
        check_batch(shape_of_gpu, peek, bs, 0, 0)
        check_batch(shape_of_cpu, peek, bs, 0, 0)
        check_batch(shape_func_gpu, peek, bs, 0, 0)
        check_batch(shape_func_cpu, peek, bs, 0, 0)


def test_gpu2cpu_old_exec_error():
    bs = 8

    @pipeline_def(
        batch_size=bs,
        num_threads=4,
        device_id=0,
        exec_async=False,
        exec_pipelined=False,
        exec_dynamic=False,
    )
    def pdef(to_cpu):
        gpu = fn.external_source("input", device="gpu")
        return to_cpu(gpu)

    with assert_raises(RuntimeError, glob="doesn't support transition from GPU to CPU"):
        _ = pdef(lambda gpu: gpu.cpu())  # this will raise an error at construction time

    pipe = pdef(lambda gpu: gpu._to_backend("cpu"))  # this will not raise errors until build-time

    with assert_raises(RuntimeError, glob="doesn't support transition from GPU to CPU"):
        pipe.build()


def test_gpu2cpu_conditionals():
    bs = 4

    @pipeline_def(
        batch_size=bs,
        num_threads=4,
        device_id=0,
        exec_dynamic=True,  # use new executor
        enable_conditionals=True,
    )
    def def_test():
        enc, label = fn.readers.file(file_root=jpeg_folder)
        img = fn.decoders.image(enc, device="mixed")
        # return inverted image for even samples
        if (label[0] & 1) == 0:
            out = img ^ np.uint8(255)
            out_cpu = out.cpu()
        else:
            out = img
            out_cpu = out.cpu()
        return out, out_cpu

    @pipeline_def(
        batch_size=bs,
        num_threads=4,
        device_id=0,
        exec_async=False,  # use old executor, even in presence of DALI_USE_EXEC2
        exec_pipelined=False,
    )
    def def_ref():
        enc, label = fn.readers.file(file_root=jpeg_folder)
        img = fn.decoders.image(enc, device="mixed")
        # return inverted image for even samples
        even = (label[0] & 1) == 0
        mask = fn.cast(even * 255, dtype=types.UINT8)
        return img ^ mask

    test_pipe = def_test()
    ref_pipe = def_ref()
    for i in range(3):
        gpu, cpu = test_pipe.run()
        assert isinstance(gpu, dali.backend_impl.TensorListGPU)
        assert isinstance(cpu, dali.backend_impl.TensorListCPU)
        (ref,) = ref_pipe.run()
        check_batch(cpu, ref, bs, 0, 0, "HWC")
        check_batch(gpu, ref, bs, 0, 0, "HWC")


def test_cse():
    @pipeline_def(batch_size=8, num_threads=4, device_id=0)
    def my_pipe():
        a = fn.random.uniform(range=[0, 1], shape=(1,), seed=123)
        b = fn.random.uniform(range=[0, 1], shape=(1,), seed=123)
        c = fn.random.uniform(range=[0, 1], shape=(1,), seed=123)
        i = fn.random.uniform(range=[0, 1], shape=(1,), seed=1234)  # different seed - must not CSE
        j = fn.random.uniform(range=[0, 1], shape=(1,), seed=123, name="do_not_merge")

        d = a[0]
        e = a[0]  # repeated a[0] should be ignored
        f = c[0]  # c -> a, so it follows that c[0] -> a[0]

        g = a[0] + b[0] - c[0]  # a[0] + a[0] - a[0]
        h = c[0] + a[0] - b[0]  # likewise
        return a, b, c, d, e, f, g, h, i, j

    pipe = my_pipe()
    a, b, c, d, e, f, g, h, i, j = pipe.run()
    assert a.data_ptr() == b.data_ptr()
    assert a.data_ptr() == c.data_ptr()
    assert a.data_ptr() != i.data_ptr()
    assert j.data_ptr() != a.data_ptr()  # j has a manually specified name and should not be merged

    assert d.data_ptr() == e.data_ptr()
    assert d.data_ptr() == f.data_ptr()

    assert g.data_ptr() == h.data_ptr()


def test_cse_ext_src():
    @pipeline_def(batch_size=1, num_threads=4, device_id=0)
    def my_pipe():
        data1 = np.float32([1, 2, 3])
        d = [data1]
        es1 = fn.external_source(source=d, cycle=True, batch=False)
        es2 = fn.external_source(source=d, cycle=True, batch=False)
        return es1, es2

    pipe = my_pipe()
    pipe.build()
    a, b = pipe.run()

    # external source operators should not be merged, even if they're identical
    assert a.data_ptr() != b.data_ptr()
    assert np.array_equal(a[0], np.float32([1, 2, 3]))
    assert np.array_equal(b[0], np.float32([1, 2, 3]))


def test_cse_cond():
    @pipeline_def(
        batch_size=8,
        num_threads=4,
        device_id=0,
        enable_conditionals=True,
        exec_dynamic=True,  # required for opportunistic MakeContiguous
    )
    def my_pipe():
        a = fn.random.uniform(range=[0, 1], shape=(1,), seed=123)
        b = fn.random.uniform(range=[0, 1], shape=(1,), seed=123)

        if a[0] > 0:
            d = a
        else:
            d = b  # this is the same as `a`

        return a, b, d

    pipe = my_pipe()
    a, b, d = pipe.run()
    assert a.data_ptr() == b.data_ptr()
    # `d` is opportunistically reassembled and gets the same first sample pointer as `a`
    assert d.data_ptr() == a.data_ptr()


def test_optional_build():
    bs = 8

    @pipeline_def(batch_size=bs, num_threads=4, device_id=0)
    def pdef_regular():
        enc, _ = fn.readers.file(file_root=jpeg_folder, name="only_reader")
        img = fn.decoders.image(enc, device="mixed")
        return img

    @pipeline_def(batch_size=bs, num_threads=4, device_id=0)
    def pdef_source():
        source = fn.external_source(name="source")
        return source

    pipes = [pdef_regular() for _ in range(5)]
    pipes.append(pdef_source())

    for pipe in pipes:
        pipe.build()

    (res,) = pipes[0].run()
    assert len(res.shape()) == 8
    pipes[1].schedule_run()
    (res_2,) = pipes[1].outputs()
    assert len(res_2.shape()) == 8
    assert pipes[2].epoch_size("only_reader") != 0
    assert pipes[3].executor_statistics() == {}
    assert "shard_id" in pipes[4].reader_meta("only_reader")

    pipes[-1].feed_input("source", np.array([10, 10]))


def test_output_descs():
    @pipeline_def(batch_size=1, num_threads=1, device_id=None)
    def my_pipe():
        return fn.reshape(np.array([1, 2, 3, 4, 5, 6], dtype=np.int32), shape=[2, 3], layout="XY")

    pipe = my_pipe(output_ndim=2, output_dtype=types.INT32, output_layout="XY")
    (o,) = pipe.run()
    assert o[0].shape() == [2, 3]
    assert o[0].layout() == "XY"

    pipe = my_pipe(output_ndim=1)
    with assert_raises(
        RuntimeError,
        glob="Number of dimensions in the output_idx=0 does not match. Expected: 1. Received: 2.",
    ):
        pipe.run()

    pipe = my_pipe(output_dtype=types.FLOAT)
    with assert_raises(
        RuntimeError,
        glob="Data type in the output_idx=0 does not match. Expected: FLOAT. Received: INT32.",
    ):
        pipe.run()

    pipe = my_pipe(output_layout="AB")
    with assert_raises(
        RuntimeError, glob="Layout in the output_idx=0 does not match. Expected: AB. Received: XY."
    ):
        pipe.run()


def test_device_auto():
    @pipeline_def(batch_size=1, num_threads=1)
    def cpuonly():
        return 42

    p = cpuonly()
    (o,) = p.run()
    assert p.device_id is None
    assert np.array(o.as_cpu()[0]) == 42

    @pipeline_def(batch_size=1, num_threads=1)
    def gpu():
        return types.Constant(42, device="gpu")

    p = gpu()
    assert p.device_id is None
    (o,) = p.run()
    assert p.device_id == 0
    assert np.array(o.as_cpu()[0]) == 42


def test_executor_flags():
    @pipeline_def(batch_size=1, num_threads=1)
    def dummy():
        return types.Constant(42, device="cpu")

    for stream_policy, concurrency in [
        (dali.StreamPolicy.PER_BACKEND, dali.OperatorConcurrency.BACKEND),
        (dali.StreamPolicy.PER_OPERATOR, dali.OperatorConcurrency.NONE),
        (dali.StreamPolicy.SINGLE, dali.OperatorConcurrency.FULL),
    ]:
        p = dummy(stream_policy=stream_policy, concurrency=concurrency)
        # check that the properties are set correctly
        assert p.stream_policy == stream_policy
        assert p.concurrency == concurrency

        # check that the flags survive serialization
        s = p.serialize()
        p2 = Pipeline.deserialize(s)
        assert p2.stream_policy == stream_policy
        assert p2.concurrency == concurrency
