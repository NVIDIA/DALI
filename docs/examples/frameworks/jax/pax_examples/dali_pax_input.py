# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from praxis import base_input

from nvidia.dali.plugin import jax as dax
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


training_data_path = os.path.join(
    os.environ["DALI_EXTRA_PATH"], "db/MNIST/training/"
)
validation_data_path = os.path.join(
    os.environ["DALI_EXTRA_PATH"], "db/MNIST/testing/"
)


@pipeline_def(device_id=0, num_threads=4, seed=0)
def mnist_pipeline(data_path, random_shuffle):
    jpegs, labels = fn.readers.caffe2(
        path=data_path,
        random_shuffle=random_shuffle,
        name="mnist_caffe2_reader",
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.GRAY)
    images = fn.crop_mirror_normalize(
        images, dtype=types.FLOAT, std=[255.0], output_layout="HWC"
    )

    labels = labels.gpu()
    labels = fn.reshape(labels, shape=[])

    return images, labels


class MnistDaliInput(base_input.BaseInput):
    def __post_init__(self):
        super().__post_init__()

        data_path = (
            training_data_path if self.is_training else validation_data_path
        )

        training_pipeline = mnist_pipeline(
            data_path=data_path,
            random_shuffle=self.is_training,
            batch_size=self.batch_size,
        )
        self._iterator = dax.DALIGenericIterator(
            training_pipeline,
            output_map=["inputs", "labels"],
            reader_name="mnist_caffe2_reader",
            auto_reset=True,
        )

    def get_next(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator.reset()
            return next(self._iterator)

    def reset(self) -> None:
        super().reset()
        self._iterator = self._iterator.reset()
