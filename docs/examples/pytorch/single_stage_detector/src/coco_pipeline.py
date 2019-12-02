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


import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class COCOPipeline(Pipeline):
    def __init__(self, default_boxes, args, seed):
        super(COCOPipeline, self).__init__(
            batch_size=args.batch_size,
            device_id=args.local_rank,
            num_threads=args.num_workers,
            seed=seed)

        try:
            shard_id = torch.distributed.get_rank()
            num_shards = torch.distributed.get_world_size()
        except RuntimeError:
            shard_id = 0
            num_shards = 1

        self.input = ops.COCOReader(
            file_root=args.train_coco_root,
            annotations_file=args.train_annotate,
            skip_empty=True,
            shard_id=shard_id,
            num_shards=num_shards,
            ratio=True,
            ltrb=True,
            random_shuffle=False,
            shuffle_after_epoch=True)

        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)

        # Augumentation techniques
        self.crop = ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            ltrb=True,
            allow_no_crop=True,
            num_attempts=1)
        self.slice = ops.Slice(device="cpu")
        self.bc = ops.BrightnessContrast(device="gpu")
        self.hsv = ops.Hsv(device="gpu")
        self.resize = ops.Resize(
            device="cpu",
            resize_x=300,
            resize_y=300,
            min_filter=types.DALIInterpType.INTERP_TRIANGULAR)

        output_dtype = types.FLOAT16 if args.fp16 else types.FLOAT

        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            crop=(300, 300),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=0,
            output_dtype=output_dtype,
            output_layout=types.NCHW,
            pad_output=False)

        # Random variables
        self.rng1 = ops.Uniform(range=[0.5, 1.5])
        self.rng2 = ops.Uniform(range=[0.875, 1.125])
        self.rng3 = ops.Uniform(range=[-0.5, 0.5])

        self.flip = ops.Flip(device="cpu")
        self.bbflip = ops.BbFlip(device="cpu", ltrb=True)
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

        crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
        images = self.slice(images, crop_begin, crop_size)

        images = self.flip(images, horizontal=coin_rnd)
        bboxes = self.bbflip(bboxes, horizontal=coin_rnd)
        images = self.resize(images)
        images = images.gpu()
        images = self.bc(images, brightness_delta=brightness, contrast_delta=contrast)
        images = self.hsv(images, hue=hue, saturation=saturation)
        images = self.normalize(images)
        bboxes, labels = self.box_encoder(bboxes, labels)

        return (images, bboxes.gpu(), labels.gpu())
