# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali as dali
import nvidia.dali.fn as fn
import os
import numpy as np
from test_utils import get_dali_extra_path

file_root = os.path.join(get_dali_extra_path(), "db/single/jpeg")
img_list = os.path.join(file_root, "image_list.txt")
batch_size = 16
rng = ndd.random.RNG(seed=42)
rng_copy = rng.clone()


def ndd_rn50_pipeline(jpegs):
    batch_size = jpegs.batch_size
    images = ndd.decoders.image(jpegs, device="mixed")
    xy = ndd.random.uniform(batch_size=batch_size, range=[0, 1], shape=2, rng=rng_copy)
    do_mirror = ndd.random.coin_flip(batch_size=batch_size, probability=0.5, rng=rng_copy)
    size = ndd.random.uniform(batch_size=batch_size, range=[256, 480], rng=rng_copy)
    resized_images = ndd.resize_crop_mirror(
        images,
        crop=[224, 224],
        crop_pos_x=xy.slice[0],
        crop_pos_y=xy.slice[1],
        mirror=do_mirror,
        resize_shorter=size,
        interp_type=dali.types.INTERP_LANCZOS3,
        antialias=False,
    )
    output = ndd.crop_mirror_normalize(
        resized_images,
        device="gpu",
        dtype=ndd.float16,
        mean=[128, 128, 128],
        std=[1, 1, 1],
    )
    return output


def random_state_source_factory(rng, batch_size):
    def source_fun():
        states = [np.array([rng() for _ in range(7)], dtype=np.uint32) for _ in range(3)]
        out = tuple([state] * batch_size for state in states)
        return tuple(out)

    return source_fun


@dali.pipeline_def(num_threads=4, device_id=0, batch_size=batch_size)
def rn50_pipeline():
    jpegs, labels = fn.readers.file(
        name="reader", file_root=file_root, file_list=img_list, random_shuffle=False
    )
    state_1, state_2, state_3 = fn.external_source(
        source=random_state_source_factory(rng, batch_size), num_outputs=3
    )
    xy = fn.random.uniform(range=[0, 1], shape=2, _random_state=state_1)
    do_mirror = fn.random.coin_flip(probability=0.5, _random_state=state_2)
    size = fn.random.uniform(range=[256, 480], _random_state=state_3)
    images = fn.decoders.image(jpegs, device="mixed")
    resized_images = fn.resize_crop_mirror(
        images,
        crop=[224, 224],
        crop_pos_x=xy[0],
        crop_pos_y=xy[1],
        mirror=do_mirror,
        resize_shorter=size,
        interp_type=dali.types.DALIInterpType.INTERP_LANCZOS3,
        antialias=False,
    )
    output = fn.crop_mirror_normalize(
        resized_images,
        device="gpu",
        dtype=dali.types.DALIDataType.FLOAT16,
        mean=[128, 128, 128],
        std=[1, 1, 1],
    )
    return output, labels


def test_rn50_pipeline():
    r = ndd.readers.File(file_root=file_root, file_list=img_list, random_shuffle=False)
    iterations = 0
    p = rn50_pipeline()
    with ndd.EvalContext(num_threads=4, device_id=0):
        for epoch in range(10):
            for jpegs, lbl_dynamic in r.next_epoch(batch_size=batch_size):
                iterations += 1
                imgs = ndd_rn50_pipeline(jpegs)
                out_dynamic = imgs.cpu().evaluate()
                assert out_dynamic.batch_size == batch_size
                out_pipeline, lbl_pipeline = p.run()
                out_pipeline = out_pipeline.as_cpu()
                for i in range(batch_size):
                    assert np.array_equal(lbl_dynamic.tensors[i], np.array(lbl_pipeline[i]))
                    assert np.array_equal(out_dynamic.tensors[i], np.array(out_pipeline[i]))

    assert iterations >= 10, "Empty test - no iterations were run"
