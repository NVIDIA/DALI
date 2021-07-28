# Copyright 2021 Kacper Kluk, Piotr Kowalewski. All Rights Reserved.
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
import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
from . import ops


class YOLOv4Pipeline:
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

        self._pipe = dali.pipeline.Pipeline(
            batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed
        )
        self._define_pipeline()

    def _define_pipeline(self):
        with self._pipe:
            images, bboxes, classes = ops.input(
                self._file_root,
                self._annotations_file,
                self._device_id,
                self._num_threads,
                "mixed" if self._use_gpu else "cpu"
            )
            images = dali.fn.resize(
                images,
                resize_x=self._image_size[0],
                resize_y=self._image_size[1],
                interp_type=dali.types.DALIInterpType.INTERP_LINEAR
            )

            if self._is_training:
                images = ops.color_twist(images)
                images, bboxes = ops.flip(images, bboxes)

                if self._use_mosaic:
                    do_mosaic = dali.fn.random.coin_flip()
                    images_m, bboxes_m, classes_m = ops.mosaic(images, bboxes, classes, self._image_size)

                    images = images * (1.0 - do_mosaic) + images_m * do_mosaic
                    bboxes = ops.select(do_mosaic, bboxes_m, bboxes)
                    classes = dali.fn.squeeze(ops.select(
                        do_mosaic,
                        dali.fn.expand_dims(classes_m, axes=1),
                        dali.fn.expand_dims(classes, axes=1)
                    ), axes=1)

            bboxes = ops.ltrb_to_xywh(bboxes)

            images = images * (1.0 / 255.0)
            # subtract one to be consistent with darknet's pretrained model weights
            classes = dali.fn.expand_dims(classes, axes=1) - 1.0

            labels = dali.fn.cat(bboxes, classes, axis=1)
            labels = dali.fn.pad(labels, fill_value=-1)
            labels = dali.fn.pad(labels, fill_value=-1, shape=(1, 5))

            self._pipe.set_outputs(images.gpu(), labels.gpu())

    def dataset(self):
        output_shapes = ((self._batch_size, self._image_size[0], self._image_size[0], 3), (self._batch_size, None, 5))
        output_dtypes = (tf.float32, tf.float32)
        return dali_tf.DALIDataset(
            pipeline=self._pipe,
            batch_size=self._batch_size,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            device_id=self._device_id
        )
