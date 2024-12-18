# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nose2.tools import params

from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.auto_aug.core import augmentation, select

from test_utils import get_dali_extra_path, check_batch

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")


def sample_info(cb):
    def idx_in_batch_cb(sample_info):
        return np.array(cb(sample_info), dtype=np.int32)

    return fn.external_source(idx_in_batch_cb, batch=False)


@augmentation(mag_range=((1.1, 3)))
def overexpose(image, multiplier):
    return fn.cast_like(image * multiplier, image)


@augmentation(mag_range=(0.1, 0.9))
def blend_edges(image, blend_factor):
    edges = fn.laplacian(image, window_size=5, dtype=types.UINT8)
    return fn.cast_like((1.0 - blend_factor) * image + blend_factor * edges, image)


def as_square_shape(edge_len):
    return [edge_len, edge_len]


@augmentation(mag_range=(0.1, 0.7), mag_to_param=as_square_shape)
def cutout(image, shape):
    return fn.erase(image, shape=shape, anchor=[0, 0], normalized=True, fill_value=120)


@params(
    ("cpu",),
    ("gpu",),
)
def test_select(dev):

    def _collect_batch(p):
        batches = p.run()
        if dev == "gpu":
            batches = (batch.as_cpu() for batch in batches)
        return tuple([np.array(sample) for sample in batch] for batch in batches)

    ops = [overexpose, blend_edges, cutout]
    num_magnitude_bins = 4
    batch_size = num_magnitude_bins * len(ops)

    @pipeline_def(enable_conditionals=True, batch_size=batch_size, num_threads=4, device_id=0)
    def pipeline_select():
        op_idx = sample_info(lambda info: info.idx_in_batch % len(ops))
        magnitude_bin = sample_info(lambda info: info.idx_in_batch % num_magnitude_bins)
        image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(image, device="cpu" if dev == "cpu" else "mixed")
        return select(ops, op_idx, image, magnitude_bin=magnitude_bin, num_magnitude_bins=4)

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
    def pipeline_refs():
        magnitude_bin = sample_info(lambda info: info.idx_in_batch % num_magnitude_bins)
        image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(image, device="cpu" if dev == "cpu" else "mixed")
        return tuple(op(image, magnitude_bin=magnitude_bin, num_magnitude_bins=4) for op in ops)

    (batch_select,) = _collect_batch(pipeline_select())
    ref_batches = _collect_batch(pipeline_refs())
    ref_batch = [ref_batches[idx % len(ops)][idx] for idx in range(batch_size)]
    check_batch(batch_select, ref_batch, max_allowed_error=1e-6)
