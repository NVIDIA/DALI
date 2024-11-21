# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import Pipeline, pipeline_def
from nvidia.dali.pipeline import do_not_convert
from nose_utils import raises, assert_raises, nottest
import nvidia.dali.fn as fn
from test_utils import get_dali_extra_path, compare_pipelines
import os

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")

N_ITER = 2

max_batch_size = 4
num_threads = 4
device_id = 0


def reference_pipeline(flip_vertical, flip_horizontal, ref_batch_size=max_batch_size):
    pipeline = Pipeline(ref_batch_size, num_threads, device_id)
    with pipeline:
        data, _ = fn.readers.file(file_root=images_dir)
        img = fn.decoders.image(data)
        flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
        pipeline.set_outputs(flipped, img)
    return pipeline


@nottest  # pipeline_def works with other decorators too
@pipeline_def(batch_size=max_batch_size, num_threads=num_threads, device_id=device_id)
def pipeline_static(flip_vertical, flip_horizontal):
    data, _ = fn.readers.file(file_root=images_dir)
    img = fn.decoders.image(data)
    flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
    return flipped, img


@nottest
@pipeline_def
def pipeline_runtime(flip_vertical, flip_horizontal):
    data, _ = fn.readers.file(file_root=images_dir)
    img = fn.decoders.image(data)
    flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
    return flipped, img


@nottest
def test_pipeline_static(flip_vertical, flip_horizontal):
    put_args = pipeline_static(flip_vertical, flip_horizontal)
    ref = reference_pipeline(flip_vertical, flip_horizontal)
    compare_pipelines(put_args, ref, batch_size=max_batch_size, N_iterations=N_ITER)


@nottest
def test_pipeline_runtime(flip_vertical, flip_horizontal):
    put_combined = pipeline_runtime(
        flip_vertical,
        flip_horizontal,
        batch_size=max_batch_size,
        num_threads=num_threads,
        device_id=device_id,
    )
    ref = reference_pipeline(flip_vertical, flip_horizontal)
    compare_pipelines(put_combined, ref, batch_size=max_batch_size, N_iterations=N_ITER)


@nottest
def test_pipeline_override(flip_vertical, flip_horizontal, batch_size):
    put_combined = pipeline_static(
        flip_vertical,
        flip_horizontal,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
    )
    ref = reference_pipeline(flip_vertical, flip_horizontal, ref_batch_size=batch_size)
    compare_pipelines(put_combined, ref, batch_size=batch_size, N_iterations=N_ITER)


def test_pipeline_decorator():
    for vert in [0, 1]:
        for hori in [0, 1]:
            yield test_pipeline_static, vert, hori
            yield test_pipeline_runtime, vert, hori
            yield test_pipeline_override, vert, hori, 5
    yield test_pipeline_runtime, fn.random.coin_flip(seed=123), fn.random.coin_flip(seed=234)
    yield test_pipeline_static, fn.random.coin_flip(seed=123), fn.random.coin_flip(seed=234)


def test_duplicated_argument():
    @pipeline_def(batch_size=max_batch_size, num_threads=num_threads, device_id=device_id)
    def ref_pipeline(val):
        data, _ = fn.readers.file(file_root=images_dir)
        return data + val

    @pipeline_def(batch_size=max_batch_size, num_threads=num_threads, device_id=device_id)
    def pipeline_duplicated_arg(num_threads):
        data, _ = fn.readers.file(file_root=images_dir)
        return data + num_threads

    pipe = pipeline_duplicated_arg(num_threads=42)
    assert pipe._num_threads == num_threads
    ref = ref_pipeline(42)
    compare_pipelines(pipe, ref, batch_size=max_batch_size, N_iterations=N_ITER)


@pipeline_def
def pipeline_kwargs(arg1, arg2, *args, **kwargs):
    pass


@raises(TypeError, regex="\\*\\*kwargs.*not allowed")
def test_kwargs_exception():
    pipeline_kwargs(arg1=1, arg2=2, arg3=3)


def test_is_pipeline_def():
    @pipeline_def
    def pipe():
        return 42

    @pipeline_def()
    def pipe_unconf():
        return 42

    @pipeline_def(max_batch_size=1, num_threads=1, device_id=0)
    def pipe_conf():
        return 42

    assert getattr(pipe, "_is_pipeline_def", False)
    assert getattr(pipe_unconf, "_is_pipeline_def", False)
    assert getattr(pipe_conf, "_is_pipeline_def", False)


def test_pipeline_def_with_do_not_convert():
    with assert_raises(
        ValueError, glob="Pipeline definition cannot be marked with @do_not_convert."
    ):

        @pipeline_def
        @do_not_convert
        def dnc_then_def_pipe():
            return 42

    with assert_raises(
        ValueError, glob="Pipeline definition cannot be marked with @do_not_convert."
    ):

        @do_not_convert
        @pipeline_def
        def def_then_dnc_pipe():
            return 42
