# Copyright 2021 Kacper Kluk. All Rights Reserved.
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

from .coco import COCO
import numpy as np
import tensorflow as tf

import os
import cv2
import random


class YOLOv4PipelineNumpy:
    def __init__(
        self, file_root, annotations_file,
        batch_size, image_size, num_threads, device_id, seed,
        **kwargs
    ):
        self._file_root = file_root
        self._annotations_file = annotations_file

        self._batch_size = batch_size
        self._image_size = image_size
        self._num_threads = num_threads
        self._device_id = device_id

        self._is_training = kwargs.get('is_training', False)
        self._use_mosaic = kwargs.get('use_mosaic', False)

        self._coco = COCO(annotations_file)

        self._batch_id = 0
        self._image_ids = self._coco.getImgIds()
        np.random.shuffle(self._image_ids)
        self._num_batches = len(self._image_ids) // (self._batch_size * self._num_threads)
        if self._num_batches == 0:
            raise Exception("Batch size exceeds training set size")


    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            num = 0
            if self._batch_id == self._num_batches:
                self._batch_id = 0
                np.random.shuffle(self._image_ids)

            start = self._batch_size * (self._batch_id * self._num_threads + self._device_id)
            image_ids = self._image_ids[start : start + self._batch_size]

            images, bboxes, classes = self._input(image_ids)
            if self._is_training:
                self._color_twist(images)
                self._flip(images, bboxes)
                if self._use_mosaic:
                    images, bboxes, classes = self._mosaic(images, bboxes, classes)

            self._batch_id += 1

            lengths = [len(b) for b in bboxes]
            bboxes = tf.RaggedTensor.from_row_lengths(tf.concat(bboxes, axis=0), lengths)
            bboxes = bboxes.to_tensor(-1)
            bboxes = tf.cast(bboxes, dtype=tf.float32)

            classes = tf.ragged.stack(classes)
            if self._batch_size > 1:
                classes = classes.to_tensor(-1)
            classes = tf.cast(tf.expand_dims(classes, axis=-1), dtype=tf.float32)

            return images, tf.concat([bboxes, classes], axis=-1)

    def __len__(self):
        return self.num_batches


    def _input(self, image_ids):
        image_data = self._coco.loadImgs(image_ids)
        images = np.zeros((self._batch_size, self._image_size[0], self._image_size[1], 3), dtype=np.float32)
        bboxes = []
        classes = []
        for i in range(self._batch_size):
            image_path = os.path.join(self._file_root, image_data[i]['file_name'])
            image_width = image_data[i]['width']
            image_height = image_data[i]['height']
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = tf.image.resize(image, self._image_size) / 255
            image = tf.reshape(image, (1, self._image_size[0], self._image_size[1], 3))
            images[i, ...] = image

            ann_ids = self._coco.getAnnIds(image_ids[i])
            anns = self._coco.loadAnns(ann_ids)
            sample_bboxes = np.array([ann['bbox'] for ann in anns])
            if len(sample_bboxes.shape) == 1:
                sample_bboxes = np.zeros((0, 4))
            sample_bboxes[ : , 0] /= image_width
            sample_bboxes[ : , 1] /= image_height
            sample_bboxes[ : , 2] /= image_width
            sample_bboxes[ : , 3] /= image_height
            sample_bboxes[ : , 0] += sample_bboxes[ : , 2] / 2
            sample_bboxes[ : , 1] += sample_bboxes[ : , 3] / 2

            bboxes.append(sample_bboxes)
            classes.append(np.array([ann['category_id'] for ann in anns], dtype=int))

        return images, bboxes, classes

    def _color_twist(self, images):
        def random_value():
            value = random.uniform(1.0, 1.5)
            coin = random.randrange(2)
            return coin * value + (1 - coin) * (1.0 / value)

        for i in range(self._batch_size):
            image = images[i, ...]

            hue = random.uniform(-18.0, 18.0)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image[..., 0] += hue
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

            brightness = random_value()
            contrast = random_value()
            image *= brightness * contrast
            image += (0.5 - 0.5 * contrast) * brightness

            images[i, ...] = image

    def _flip(self, images, bboxes):
        for i in range(self._batch_size):
            if random.randrange(2) == 0:
                images[i, ...] = images[i, : , ::-1 , : ]
                bboxes[i][: , 0] = 1.0 - bboxes[i][: , 0]

    def _mosaic(self, images, bboxes, classes):
        def trim_bboxes(bboxes, classes, x0, y0, x1, y1):
            bboxes_ltrb = np.copy(bboxes)
            bboxes_ltrb[ : , 0] -= bboxes[ : , 2] / 2
            bboxes_ltrb[ : , 1] -= bboxes[ : , 3] / 2
            bboxes_ltrb[ : , 2] += bboxes_ltrb[ : , 0]
            bboxes_ltrb[ : , 3] += bboxes_ltrb[ : , 1]

            bboxes_ltrb[ : , 0] = np.maximum(x0, bboxes_ltrb[ : , 0])
            bboxes_ltrb[ : , 1] = np.maximum(y0, bboxes_ltrb[ : , 1])
            bboxes_ltrb[ : , 2] = np.minimum(x1, bboxes_ltrb[ : , 2])
            bboxes_ltrb[ : , 3] = np.minimum(y1, bboxes_ltrb[ : , 3])

            bboxes_xywh = np.copy(bboxes_ltrb)
            bboxes_xywh[ : , 2] -= bboxes_xywh[ : , 0]
            bboxes_xywh[ : , 3] -= bboxes_xywh[ : , 1]
            bboxes_xywh[ : , 0] += bboxes_xywh[ : , 2] / 2
            bboxes_xywh[ : , 1] += bboxes_xywh[ : , 3] / 2

            not_null = np.logical_and(bboxes_xywh[ : , 2] > 0, bboxes_xywh[ : , 3] > 0)
            return bboxes_xywh[not_null, : ], classes[not_null]

        images_out = np.zeros(images.shape)
        bboxes_out = []
        classes_out = []
        for i in range(self._batch_size):
            if random.randrange(2) == 0:
                images_out[i, ...] = images[i, ...]
                bboxes_out.append(bboxes[i])
                classes_out.append(classes[i])
            else:
                ids = np.random.choice(self._batch_size, 4)
                prop_x = np.random.uniform(0.2, 0.8)
                prop_y = np.random.uniform(0.2, 0.8)
                size_x = int(prop_x * self._image_size[0])
                size_y = int(prop_y * self._image_size[1])
                images_out[i, : size_y, : size_x, : ] = images[ids[0], : size_y, : size_x, : ]
                images_out[i, : size_y, size_x :, : ] = images[ids[1], : size_y, size_x :, : ]
                images_out[i, size_y :, : size_x, : ] = images[ids[2], size_y :, : size_x, : ]
                images_out[i, size_y :, size_x :, : ] = images[ids[3], size_y :, size_x :, : ]
                bboxes00, classes00 = trim_bboxes(bboxes[ids[0]], classes[ids[0]], 0.0, 0.0, prop_x, prop_y)
                bboxes10, classes10 = trim_bboxes(bboxes[ids[1]], classes[ids[1]], prop_x, 0.0, 1.0, prop_y)
                bboxes01, classes01 = trim_bboxes(bboxes[ids[2]], classes[ids[2]], 0.0, prop_y, prop_x, 1.0)
                bboxes11, classes11 = trim_bboxes(bboxes[ids[3]], classes[ids[3]], prop_x, prop_y, 1.0, 1.0)
                bboxes_out.append(np.concatenate((bboxes00, bboxes10, bboxes01, bboxes11)))
                classes_out.append(np.concatenate((classes00, classes10, classes01, classes11)))

        return images_out, bboxes_out, classes_out


    def dataset(self):
        return tf.data.Dataset.from_generator(lambda: self,
            output_signature = (
                tf.TensorSpec(shape=(None, 608, 608, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32
            )
        )).prefetch(tf.data.AUTOTUNE)
