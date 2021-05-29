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

        self._use_gpu = kwargs.get('use_gpu', False)
        self._is_training = kwargs.get('is_training', False)
        self._use_mosaic = kwargs.get('use_mosaic', False)

        self._coco = COCO(annotations_file)

        self._batch_id = 0
        self._image_ids = self._coco.getImgIds()
        np.random.shuffle(self._image_ids)
        self._num_batches = len(self._image_ids) // self._batch_size


    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):

            num = 0
            if self._batch_id == self._num_batches:
                self._batch_id = 0
                np.random.shuffle(self._image_ids)

            start = self._batch_id * self._batch_size
            image_ids = self._image_ids[start : start + self._batch_size]

            images, bboxes, classes = self._input(image_ids)
            if self._is_training:
                self._color_twist(images)
                self._flip(images, bboxes)

            self._batch_id = self._batch_id + 1

            lengths = [len(b) for b in bboxes]
            bboxes = tf.RaggedTensor.from_row_lengths(tf.concat(bboxes, axis=0), lengths)
            bboxes = bboxes.to_tensor(-1)
            bboxes = tf.cast(bboxes, dtype=tf.float32)

            classes = tf.ragged.stack(classes)
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
