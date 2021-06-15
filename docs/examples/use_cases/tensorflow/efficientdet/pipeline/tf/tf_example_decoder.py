# Copyright 2020 Google Research. All Rights Reserved.
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
"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""

import tensorflow.compat.v1 as tf


class TfExampleDecoder(object):
    """Tensorflow Example proto decoder."""

    def __init__(self):
        self._keys_to_features = {
            "image/encoded": tf.FixedLenFeature((), tf.string),
            "image/height": tf.FixedLenFeature((), tf.int64, -1),
            "image/width": tf.FixedLenFeature((), tf.int64, -1),
            "image/object/bbox/xmin": tf.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.VarLenFeature(tf.float32),
            "image/object/class/label": tf.VarLenFeature(tf.int64),
            "image/object/area": tf.VarLenFeature(tf.float32),
        }

    def _decode_image(self, parsed_tensors):
        """Decodes the image and set its static shape."""
        image = tf.io.decode_image(parsed_tensors["image/encoded"], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors["image/object/bbox/xmin"]
        xmax = parsed_tensors["image/object/bbox/xmax"]
        ymin = parsed_tensors["image/object/bbox/ymin"]
        ymax = parsed_tensors["image/object/bbox/ymax"]
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def decode(self, serialized_example):
        """Decode the serialized example.

        Args:
          serialized_example: a single serialized tf.Example string.

        Returns:
          decoded_tensors: a dictionary of tensors with the following fields:
            - image: a uint8 tensor of shape [None, None, 3].
            - height: an integer scalar tensor.
            - width: an integer scalar tensor.
            - groundtruth_classes: a int64 tensor of shape [None].
            - groundtruth_boxes: a float32 tensor of shape [None, 4].
        """
        parsed_tensors = tf.io.parse_single_example(
            serialized_example, self._keys_to_features
        )
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse_tensor_to_dense(
                        parsed_tensors[k], default_value=""
                    )
                else:
                    parsed_tensors[k] = tf.sparse_tensor_to_dense(
                        parsed_tensors[k], default_value=0
                    )

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)

        decode_image_shape = tf.logical_or(
            tf.equal(parsed_tensors["image/height"], -1),
            tf.equal(parsed_tensors["image/width"], -1),
        )
        image_shape = tf.cast(tf.shape(image), dtype=tf.int64)

        parsed_tensors["image/height"] = tf.where(
            decode_image_shape, image_shape[0], parsed_tensors["image/height"]
        )
        parsed_tensors["image/width"] = tf.where(
            decode_image_shape, image_shape[1], parsed_tensors["image/width"]
        )

        decoded_tensors = {
            "image": image,
            "height": parsed_tensors["image/height"],
            "width": parsed_tensors["image/width"],
            "groundtruth_classes": parsed_tensors["image/object/class/label"],
            "groundtruth_boxes": boxes,
        }
        return decoded_tensors
