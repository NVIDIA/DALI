# Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn


@pipeline_def
def create_coco_pipeline(default_boxes, args):
    try:
        shard_id = torch.distributed.get_rank()
        num_shards = torch.distributed.get_world_size()
    except RuntimeError:
        shard_id = 0
        num_shards = 1

    images, bboxes, labels = fn.readers.coco(file_root=args.train_coco_root,
                                             annotations_file=args.train_annotate,
                                             skip_empty=True,
                                             shard_id=shard_id,
                                             num_shards=num_shards,
                                             ratio=True,
                                             ltrb=True,
                                             random_shuffle=False,
                                             shuffle_after_epoch=True,
                                             name="Reader")

    crop_begin, crop_size, bboxes, labels = fn.random_bbox_crop(bboxes, labels,
                                                                device="cpu",
                                                                aspect_ratio=[0.5, 2.0],
                                                                thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                                                scaling=[0.3, 1.0],
                                                                bbox_layout="xyXY",
                                                                allow_no_crop=True,
                                                                num_attempts=50)
    images = fn.decoders.image_slice(images, crop_begin, crop_size, device="mixed", output_type=types.RGB)
    flip_coin = fn.random.coin_flip(probability=0.5)
    images = fn.resize(images,
                       resize_x=300,
                       resize_y=300,
                       min_filter=types.DALIInterpType.INTERP_TRIANGULAR)

    saturation = fn.random.uniform(range=[0.5, 1.5])
    contrast = fn.random.uniform(range=[0.5, 1.5])
    brightness = fn.random.uniform(range=[0.875, 1.125])
    hue = fn.random.uniform(range=[-0.5, 0.5])

    images = fn.hsv(images, dtype=types.FLOAT, hue=hue, saturation=saturation)  # use float to avoid clipping and
                                                         # quantizing the intermediate result
    images = fn.brightness_contrast(images,
                                    contrast_center = 128,  # input is in float, but in 0..255 range
                                    dtype = types.UINT8,
                                    brightness = brightness,
                                    contrast = contrast)

    dtype = types.FLOAT16 if args.fp16_mode else types.FLOAT

    bboxes = fn.bb_flip(bboxes, ltrb=True, horizontal=flip_coin)
    images = fn.crop_mirror_normalize(images,
                                      crop=(300, 300),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                      mirror=flip_coin,
                                      dtype=dtype,
                                      output_layout="CHW",
                                      pad_output=False)

    bboxes, labels = fn.box_encoder(bboxes, labels,
                                    criteria=0.5,
                                    anchors=default_boxes.as_ltrb_list())

    labels=labels.gpu()
    bboxes=bboxes.gpu()

    return images, bboxes, labels
