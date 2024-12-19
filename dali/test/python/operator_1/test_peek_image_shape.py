# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import os
from test_utils import get_dali_extra_path, dali_type_to_np

test_data_root = get_dali_extra_path()
path = "db/single"
file_types = {"jpeg", "mixed", "png", "tiff", "pnm", "bmp", "jpeg2k"}


def run_decode(data_path, out_type):
    batch_size = 4
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    input, _ = fn.readers.file(file_root=data_path, shard_id=0, num_shards=1, name="reader")
    decoded = fn.decoders.image(input, output_type=types.RGB)
    decoded_shape = decoded.shape()
    raw_shape = fn.peek_image_shape(input, dtype=out_type)
    pipe.set_outputs(decoded, decoded_shape, raw_shape)
    samples = 0
    length = pipe.reader_meta(name="reader")["epoch_size"]
    while samples < length:
        samples += batch_size
        (images, decoded_shape, raw_shape) = pipe.run()
        for i in range(batch_size):
            # as we are asking for a particular color space it may
            # differ from the source image, so don't compare it
            image = images.at(i)
            shape_type = np.int64 if out_type is None else dali_type_to_np(out_type)
            for d in range(len(image.shape) - 1):
                assert image.shape[d] == decoded_shape.at(i)[d], "{} vs {}".format(
                    image.shape[d], decoded_shape.at(i)[d]
                )
                assert image.shape[d] == raw_shape.at(i)[d], "{} vs {}".format(
                    image.shape[d], raw_shape.at(i)[d]
                )
                assert raw_shape.at(i)[d].dtype == shape_type, "{} vs {}".format(
                    raw_shape.at(i)[d].dtyp, shape_type
                )


test_types = [
    None,
    types.INT32,
    types.UINT32,
    types.INT64,
    types.UINT64,
    types.FLOAT,
    types.FLOAT64,
]


def test_operator_peek_image_shape():
    for img_type in file_types:
        for out_type in test_types:
            data_path = os.path.join(test_data_root, path, img_type)
            yield run_decode, data_path, out_type
