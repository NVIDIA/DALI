# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.pipeline import pipeline
from nose.tools import nottest
import nvidia.dali.fn as fn
from test_utils import get_dali_extra_path, compare_pipelines
import os

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')

batch_size = 16
num_threads = 4
device_id = 0


def reference_pipeline(flip_vertical, flip_horizontal):
    pipeline = Pipeline(batch_size, num_threads, device_id)
    with pipeline:
        data, _ = fn.file_reader(file_root=images_dir)
        img = fn.image_decoder(data, device="mixed")
        flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
        pipeline.set_outputs(flipped, img)
    return pipeline


@nottest
@pipeline_args(batch_size, num_threads, device_id)
def pipeline_under_test1(flip_vertical, flip_horizontal):
    data, _ = fn.file_reader(file_root=images_dir)
    img = fn.image_decoder(data, device="mixed")
    flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
    return flipped, img


# @nottest
# @pipeline_class(batch_size, num_threads, device_id)
# def pipeline_under_test2(flip_vertical, flip_horizontal):
#     data, _ = fn.file_reader(file_root=images_dir)
#     img = fn.image_decoder(data, device="mixed")
#     flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
#     return flipped, img

@nottest
@pipeline_combined
def pipeline_under_test3(flip_vertical, flip_horizontal):
    data, _ = fn.file_reader(file_root=images_dir)
    img = fn.image_decoder(data, device="mixed")
    flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
    return flipped, img



@nottest
def test_pipeline_decorator_helper(flip_vertical, flip_horizontal):
    put_args = pipeline_under_test1(flip_vertical, flip_horizontal)

    # pipeline_under_test2.set_params(batch_size=batch_size, max_streams=-1)
    # pipeline_under_test2.batch_size=batch_size
    # put_class=pipeline_under_test2(flip_vertical, flip_horizontal)

    put_combined=pipeline_under_test3(flip_vertical, flip_horizontal, batch_size=batch_size)

    ref = reference_pipeline(flip_vertical, flip_horizontal)
    compare_pipelines(put_args, ref, batch_size=batch_size, N_iterations=7)
    # compare_pipelines(put_class, ref, batch_size=batch_size, N_iterations=7)
    compare_pipelines(put_combined, ref, batch_size=batch_size, N_iterations=7)


def test_pipeline_decorator():
    for vert in [0, 1]:
        for hori in [0, 1]:
            yield test_pipeline_decorator_helper, vert, hori
