# Copyright 2021 Kacper Kluk, Paweł Anikiel, Jagoda Kamińska. All Rights Reserved.
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
# ==============================================================================
import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
import math

from absl import logging
from glob import glob
from pipeline import anchors
from utils import InputType

from . import ops_util


class EfficientDetPipeline:
    def __init__(
        self,
        params,
        batch_size,
        args,
        is_training=True,
        num_shards=1,
        device_id=0,
        cpu_only=False,
    ):
        self._batch_size = batch_size
        self._image_size = params["image_size"]
        self._gridmask = params["grid_mask"]

        self._input_type = args.input_type
        if self._input_type == InputType.tfrecord:
            if is_training:
                file_pattern = args.train_file_pattern
            else:
                file_pattern = args.eval_file_pattern or args.train_file_pattern

            self._tfrecord_files = glob(file_pattern)
            self._tfrecord_idxs = [
                filename + "_idx" for filename in self._tfrecord_files
            ]
        else:
            self._images_path = args.images_path
            self._annotations_path = args.annotations_path

        self._is_training = is_training
        self._num_shards = num_shards
        self._shard_id = None if cpu_only else device_id
        self._device = "cpu" if cpu_only else "gpu"

        self._anchors = anchors.Anchors(
            3, 7, 3, [1.0, 2.0, 0.5], 4.0, params["image_size"]
        )
        self._boxes = self._get_boxes()
        self._max_instances_per_image = params["max_instances_per_image"] or 100
        seed = params["seed"] or -1

        self._pipe = self._define_pipeline(
            batch_size=self._batch_size,
            num_threads=self._num_shards,
            device_id=device_id,
            seed=seed,
        )

    def _get_boxes(self):
        boxes_t = self._anchors.boxes[:, 0] / self._image_size[0]
        boxes_l = self._anchors.boxes[:, 1] / self._image_size[1]
        boxes_b = self._anchors.boxes[:, 2] / self._image_size[0]
        boxes_r = self._anchors.boxes[:, 3] / self._image_size[1]
        boxes = tf.transpose(tf.stack([boxes_l, boxes_t, boxes_r, boxes_b]))
        return tf.reshape(boxes, boxes.shape[0] * 4).numpy().tolist()

    @pipeline_def
    def _define_pipeline(self):
        if self._input_type == InputType.tfrecord:
            images, bboxes, classes, widths, heights = ops_util.input_tfrecord(
                self._tfrecord_files,
                self._tfrecord_idxs,
                device=self._device,
                shard_id=self._shard_id,
                num_shards=self._num_shards,
                random_shuffle=self._is_training,
            )
        elif self._input_type == InputType.coco:
            images, bboxes, classes, widths, heights = ops_util.input_coco(
                self._images_path,
                self._annotations_path,
                device=self._device,
                shard_id=self._shard_id,
                num_shards=self._num_shards,
                random_shuffle=self._is_training,
            )

        if self._is_training and self._gridmask:
            images = ops_util.gridmask(images, widths, heights)

        images, bboxes = ops_util.normalize_flip(
            images, bboxes, 0.5 if self._is_training else 0.0
        )

        images, bboxes, classes = ops_util.random_crop_resize(
            images,
            bboxes,
            classes,
            widths,
            heights,
            self._image_size,
            [0.1, 2.0] if self._is_training else None,
        )

        if self._device == "gpu":
            bboxes = bboxes.gpu()
            classes = classes.gpu()

        enc_bboxes, enc_classes = dali.fn.box_encoder(
            bboxes, classes, anchors=self._boxes, offset=True
        )
        num_positives = dali.fn.reductions.sum(
            dali.fn.cast(enc_classes != 0, dtype=dali.types.FLOAT),
        )
        enc_classes -= 1

        # convert to tlbr
        enc_bboxes = dali.fn.coord_transform(
            enc_bboxes,
            M=[0, 1, 0, 0,
                1, 0, 0, 0,
                0, 0, 0, 1,
                0, 0, 1, 0]
        )

        # split into layers by size
        enc_bboxes_layers, enc_classes_layers = self._unpack_labels(
            enc_bboxes, enc_classes
        )

        # interleave enc_bboxes_layers and enc_classes_layers
        enc_layers = [
            item
            for pair in zip(enc_classes_layers, enc_bboxes_layers)
            for item in pair
        ]

        bboxes = ops_util.bbox_to_effdet_format(
            bboxes, self._image_size
        )
        bboxes = dali.fn.pad(
            bboxes,
            fill_value=-1,
            shape=(self._max_instances_per_image, 4),
        )
        classes = dali.fn.pad(
            classes,
            fill_value=-1,
            shape=(self._max_instances_per_image,),
        )

        return images, num_positives, bboxes, classes, *enc_layers

    def _unpack_labels(self, enc_bboxes, enc_classes):
        # from keras/anchors.py

        enc_bboxes_layers = []
        enc_classes_layers = []

        if self._device == "gpu":
            enc_bboxes = enc_bboxes.gpu()
            enc_classes = enc_classes.gpu()

        count = 0
        for level in range(self._anchors.min_level, self._anchors.max_level + 1):
            feat_size = self._anchors.feat_sizes[level]
            steps = (
                feat_size["height"]
                * feat_size["width"]
                * self._anchors.get_anchors_per_location()
            )

            enc_bboxes_layers.append(
                dali.fn.reshape(
                    enc_bboxes[count:count + steps, 0:4],
                    shape=[feat_size["height"], feat_size["width"], -1],
                )
            )
            enc_classes_layers.append(
                dali.fn.reshape(
                    enc_classes[count:count + steps],
                    shape=[feat_size["height"], feat_size["width"], -1],
                )
            )

            count += steps

        return enc_bboxes_layers, enc_classes_layers

    def get_dataset(self):
        output_shapes = [
            (self._batch_size, self._image_size[0], self._image_size[1], 3),
            (self._batch_size,),
            (self._batch_size, None, 4),
            (self._batch_size, None),
        ]
        output_dtypes = [tf.float32, tf.float32, tf.float32, tf.int32]

        for level in range(self._anchors.min_level, self._anchors.max_level + 1):
            feat_size = self._anchors.feat_sizes[level]
            output_shapes.append(
                (
                    self._batch_size,
                    feat_size["height"],
                    feat_size["width"],
                    self._anchors.get_anchors_per_location(),
                )
            )
            output_shapes.append(
                (
                    self._batch_size,
                    feat_size["height"],
                    feat_size["width"],
                    self._anchors.get_anchors_per_location() * 4,
                )
            )
            output_dtypes.append(tf.int32)
            output_dtypes.append(tf.float32)

        dataset = dali_tf.DALIDataset(
            pipeline=self._pipe,
            batch_size=self._batch_size,
            output_shapes=tuple(output_shapes),
            output_dtypes=tuple(output_dtypes),
        )
        return dataset

    def build(self):
        self._pipe.build()

    def run(self):
        return self._pipe.run()
