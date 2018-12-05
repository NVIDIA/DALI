# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import ctypes
import logging

import numpy as np

# DALI imports
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import time


class COCOPipeline(Pipeline):
    def __init__(
        self, 
        batch_size,  
        file_root, 
        annotations_file, 
        default_boxes,
        seed,
        device_id=0,
        num_threads=4):

        super(COCOPipeline, self).__init__(
            batch_size=batch_size, device_id=device_id, num_threads=num_threads, seed = seed)

        self.input = ops.COCOReader(
            file_root = file_root,
            annotations_file = annotations_file,
            ratio=True, ltrb=True,
            random_shuffle=True)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        # Augumentation techniques
        self.crop = ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.8, 1.0],
            ltrb=True)
        self.slice = ops.Slice(device="gpu")
        self.twist = ops.ColorTwist(device="gpu")
        self.resize = ops.Resize(device = "gpu", resize_x = 300, resize_y = 300)
        self.normalize = ops.CropMirrorNormalize(
            device="gpu", crop=(300, 300),
            mean=[0.485 * 255., 0.456 * 255., 0.406 * 255.],
            std=[0.229 * 255., 0.224 * 255., 0.225 * 255.])

        # Random variables
        self.rng1 = ops.Uniform(range=[0.5, 1.5])
        self.rng2 = ops.Uniform(range=[0.875, 1.125])
        self.rng3 = ops.Uniform(range=[-0.5, 0.5])

        self.flip = ops.Flip(device = "gpu")
        self.bbflip = ops.BbFlip(device = "cpu", ltrb=True)
        self.flip_coin = ops.CoinFlip(probability=0.5)

        self.box_encoder = ops.BoxEncoder(
            device="cpu", 
            criteria=0.5, 
            anchors=default_boxes.as_ltrb_list())

    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()

        coin_rnd = self.flip_coin()

        inputs, bboxes, labels = self.input(name="Reader")
        images = self.decode(inputs)

        images = images.gpu()

        crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
        images = self.slice(images, crop_begin, crop_size)

        images = self.flip(images, horizontal = coin_rnd)
        bboxes = self.bbflip(bboxes, horizontal = coin_rnd)

        images = self.resize(images)
        images = self.twist(images, saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)
        images = self.normalize(images)

        boxes, labels = self.box_encoder(bboxes, labels)

        return (images, boxes.gpu(), labels.gpu())