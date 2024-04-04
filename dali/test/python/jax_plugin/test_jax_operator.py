# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

import jax
from nvidia.dali import pipeline_def, fn, types
import nvidia.dali.plugin.jax as dax

from nose2.tools import params
from test_utils import get_dali_extra_path, check_batch


test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, "db", "single", "jpeg")


@params(("cpu", True), ("cpu", False), ("gpu", True), ("gpu", False))
def test_identity(device, use_jit):
    """
    Test if sharing the inputs and outputs works correctly with identity operator
    """

    num_iters = 5
    batch_size = 8

    @dax.fn.jax_function
    @jax.jit
    def identity_jit(batch):
        return batch

    @dax.fn.jax_function
    def identity(batch):
        return batch

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def pipeline():
        idx = fn.external_source(
            lambda sample_info: np.array(
                [
                    sample_info.idx_in_batch,
                    sample_info.iteration,
                ],
                dtype=np.int32,
            ),
            batch=False,
            device=device,
        )
        return identity_jit(idx) if use_jit else identity(idx)

    p = pipeline()
    p.build()
    for i in range(num_iters):
        (batch,) = p.run()
        if device == "gpu":
            batch = batch.as_cpu()
        batch = [np.array(s) for s in batch]
        assert len(batch) == batch_size, f"{len(batch)}!= {batch_size}"
        for j, sample in enumerate(batch):
            np.testing.assert_array_equal(sample, np.array([j, i], dtype=np.int32))


@params("cpu", "gpu")
def test_conditionals(device):
    """
    Conditionals enforce non-contiguous inputs to the operator
    """

    num_iters = 5
    batch_size = 13

    @dax.fn.jax_function
    @jax.jit
    def plus_one(batch):
        return batch + 1

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, enable_conditionals=True)
    def pipeline():
        idx = fn.external_source(
            lambda sample_info: np.array(
                [
                    sample_info.idx_in_batch,
                    sample_info.iteration,
                ],
                dtype=np.int32,
            ),
            batch=False,
            device=device,
        )
        cf = fn.random.coin_flip(seed=42)
        if cf:
            idx = plus_one(idx)
        return idx, cf

    p = pipeline()
    p.build()
    for i in range(num_iters):
        batch, cfs = p.run()
        if device == "gpu":
            batch = batch.as_cpu()
        batch = [np.array(s) for s in batch]
        cfs = [np.array(s) for s in cfs]
        assert len(batch) == batch_size, f"{len(batch)}!= {batch_size}"
        assert len(cfs) == batch_size, f"{len(cfs)}!= {batch_size}"
        for j, (sample, cf) in enumerate(zip(batch, cfs)):
            ref = np.array([j, i], dtype=np.int32) + cf
            np.testing.assert_array_equal(sample, ref)


@params("cpu", "gpu")
def test_pre_and_post_ops(device):
    """Check if the operator works correctly with other DALI operators"""
    num_iters = 3
    batch_size = 13

    def center_crop_base(image):
        dest_image_size = 244
        h, w, _ = image.shape
        h_start = (h - dest_image_size) // 2
        h_end = h_start + dest_image_size
        w_start = (w - dest_image_size) // 2
        w_end = w_start + dest_image_size
        return image[h_start:h_end, w_start:w_end, :]

    @dax.fn.jax_function
    @jax.vmap
    @jax.jit
    def center_crop(image):
        return center_crop_base(image)

    h, w = 702, 502
    base_sample = np.arange(h * w * 3, dtype=np.uint8).reshape((h, w, 3))

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, enable_conditionals=True)
    def pipeline():
        img = types.Constant(base_sample, device=device)
        img = fn.flip(img, horizontal=True)
        img = center_crop(img)
        img = fn.flip(img, horizontal=True)
        return img

    p = pipeline()
    p.build()
    ref = center_crop_base(base_sample)
    for _ in range(num_iters):
        (batch,) = p.run()
        if device == "gpu":
            batch = batch.as_cpu()
        batch = [np.array(s) for s in batch]
        assert len(batch) == batch_size, f"{len(batch)}!= {batch_size}"
        for sample in batch:
            np.testing.assert_array_equal(sample, ref)


@params("cpu", "gpu")
def test_multi_input_output(device):

    num_iters = 3
    batch_size = 5

    def cond_flip_base(image, flip_vert, flip_hor):
        flip_channels = flip_hor * flip_vert
        return (
            flip_channels,
            image[:: (-1) ** flip_hor, :: (-1) ** flip_vert, :: (-1) ** flip_channels],
        )

    @dax.fn.jax_function(num_outputs=2)
    @jax.jit
    @jax.vmap
    def cond_flip(image, flip_vert, flip_hor):
        flip_channels = flip_hor * flip_vert
        image = jax.lax.cond(flip_hor, lambda x: x[::-1, :, :], lambda x: x, image)
        image = jax.lax.cond(flip_vert, lambda x: x[:, ::-1, :], lambda x: x, image)
        image = jax.lax.cond(flip_channels, lambda x: x[:, :, ::-1], lambda x: x, image)
        return flip_channels, image

    h, w = 487, 499
    base_sample = np.arange(h * w * 3, dtype=np.float32).reshape((h, w, 3))

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, enable_conditionals=True)
    def pipeline():
        img = types.Constant(base_sample, device=device)
        flip_vert = fn.random.coin_flip(seed=42)
        flip_hor = fn.random.coin_flip(seed=43)
        jax_flip_channels, jax_img = cond_flip(
            img,
            flip_vert if device == "cpu" else flip_vert.gpu(),
            flip_hor if device == "cpu" else flip_hor.gpu(),
        )
        return jax_flip_channels, flip_vert, flip_hor, jax_img

    p = pipeline()
    p.build()
    for _ in range(num_iters):
        jax_flip_channels, flip_vert, flip_hor, batch = p.run()
        if device == "gpu":
            batch = batch.as_cpu()
            jax_flip_channels = jax_flip_channels.as_cpu()
        jax_flip_channels, flip_vert, flip_hor, batch = tuple(
            [np.array(s) for s in b] for b in (jax_flip_channels, flip_vert, flip_hor, batch)
        )
        for i, b in enumerate((jax_flip_channels, flip_vert, flip_hor, batch)):
            assert len(b) == batch_size, f"{i}, {len(b)}!= {batch_size}"
        for jax_flip_channel, flip_h, flip_v, sample in zip(
            jax_flip_channels, flip_vert, flip_hor, batch
        ):
            flip_channels_ref, ref = cond_flip_base(base_sample, flip_h, flip_v)
            np.testing.assert_array_equal(jax_flip_channel, flip_channels_ref)
            np.testing.assert_array_equal(sample, ref)
            assert jax_flip_channel == flip_h * flip_v


@params("cpu", "gpu")
def test_multi_input_different_contiguity(device):

    batch_size = 11
    num_iters = 4

    @dax.fn.jax_function
    @jax.jit
    @jax.vmap
    def flip_hor(image, should_flip):
        return jax.lax.cond(should_flip, lambda x: x[:, ::-1, :], lambda x: x, image)

    @pipeline_def(
        batch_size=batch_size, device_id=0, num_threads=4, seed=42, enable_conditionals=True
    )
    def pipeline():
        img, _ = fn.readers.file(name="Reader", file_root=images_dir, random_shuffle=True, seed=42)
        img = fn.decoders.image(img, device="cpu" if device == "cpu" else "mixed")
        img = fn.resize(img, size=(224, 224))
        sample_idx = fn.external_source(
            lambda sample_info: np.array(sample_info.idx_in_batch, dtype=np.int32), batch=False
        )
        if sample_idx & 1 == 0:
            mod = sample_idx & 1
            mod_dev = mod.gpu() if device == "gpu" else mod
        else:
            mod = sample_idx & 1
            mod_dev = mod.gpu() if device == "gpu" else mod
        return flip_hor(img, mod_dev), fn.flip(img, horizontal=mod)

    p = pipeline()
    p.build()
    for _ in range(num_iters):
        jax_flip, dali_flip = p.run()
        check_batch(jax_flip, dali_flip, compare_layouts=True, max_allowed_error=0)


@params(
    ("cpu", True, False),
    ("cpu", False, False),
    ("cpu", False, True),
    ("gpu", True, False),
    ("gpu", False, False),
    ("gpu", False, True),
)
def test_preserve(device, preserve, no_in_out):

    num_iters = 3
    batch_size = 5

    counter = 0

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, enable_conditionals=True)
    def pipeline():

        def incr(img):
            nonlocal counter
            counter += 1
            return img

        def incr_no_in_out():
            nonlocal counter
            counter += 1

        img = types.Constant(np.full((3, 3), 1), device=device)
        _ = dax.fn.jax_function(incr_no_in_out if no_in_out else incr, preserve=preserve)(img)
        return img

    p = pipeline()
    p.build()
    for _ in range(num_iters):
        p.run()

    if preserve:
        assert counter != 0
    else:
        assert counter == 0
