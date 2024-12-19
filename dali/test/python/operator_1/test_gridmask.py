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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import math
import os
import cv2
import itertools
from nose2.tools import params
from test_utils import get_dali_extra_path

data_root = get_dali_extra_path()
img_dir = os.path.join(data_root, "db", "single", "jpeg")


def get_pipeline(device, batch_size, tile, ratio, angle):
    pipe = Pipeline(batch_size, 4, 0)
    with pipe:
        input, _ = fn.readers.file(file_root=img_dir)
        decoded = fn.decoders.image(input, device="cpu", output_type=types.RGB)
        decoded = decoded.gpu() if device == "gpu" else decoded
        grided = fn.grid_mask(decoded, device=device, tile=tile, ratio=ratio, angle=angle)
        pipe.set_outputs(grided, decoded)
    return pipe


def get_random_pipeline(device, batch_size):
    pipe = Pipeline(batch_size, 4, 0)
    with pipe:
        input, _ = fn.readers.file(file_root=img_dir)
        decoded = fn.decoders.image(input, device="cpu", output_type=types.RGB)
        decoded = decoded.gpu() if device == "gpu" else decoded
        tile = fn.cast(fn.random.uniform(range=(50, 200)), dtype=types.INT32)
        ratio = fn.random.uniform(range=(0.3, 0.7))
        angle = fn.random.uniform(range=(-math.pi, math.pi))
        grided = fn.grid_mask(decoded, device=device, tile=tile, ratio=ratio, angle=angle)
        pipe.set_outputs(grided, decoded, tile, ratio, angle)
    return pipe


def get_mask(w, h, tile, ratio, angle, d):
    ca = math.cos(angle)
    sa = math.sin(angle)
    b = tile * ratio

    i = np.tile(np.arange(w), (h, 1))
    j = np.transpose(np.tile(np.arange(h), (w, 1)))
    x = i * ca - j * sa
    y = i * sa + j * ca
    m = np.logical_or(((x + d) % tile > b + 2 * d), ((y + d) % tile > b + 2 * d))
    return m


def check(result, input, tile, ratio, angle):
    result = np.uint8(result)
    input = np.uint8(input)
    w = result.shape[1]
    h = result.shape[0]
    eps = 0.1

    # inside of squares should be black
    mask = np.uint8(1 - get_mask(w, h, tile, ratio, angle, -eps))
    result2 = cv2.bitwise_and(result, result, mask=mask)
    assert not np.any(result2)

    # outside of squares should be same as input
    mask = np.uint8(get_mask(w, h, tile, ratio, angle, eps))
    result2 = cv2.bitwise_and(result, result, mask=mask)
    input2 = cv2.bitwise_and(input, input, mask=mask)
    assert np.all(result2 == input2)


def run_test(batch_size, device, tile, ratio, angle):
    pipe = get_pipeline(device, batch_size, tile, ratio, angle)
    results, inputs = pipe.run()
    if device == "gpu":
        results, inputs = results.as_cpu(), inputs.as_cpu()
    for i in range(batch_size):
        check(results[i], inputs[i], tile, ratio, angle)


devices = ["cpu", "gpu"]
args = [
    (40, 0.5, 0),
    (100, 0.1, math.pi / 2),
    (200, 0.7, math.pi / 3),
    (150, 1 / 3, math.pi / 4),
    (50, 0.532, 1),
    (51, 0.38158387, 2.6810782),
    (123, 0.456, 0.789),
]


@params(*itertools.product(devices, args))
def test_gridmask_vs_cv(device, args):
    batch_size = 4
    tile, ratio, angle = args
    run_test(batch_size, device, tile, ratio, angle)


def run_random_test(batch_size, device):
    pipe = get_random_pipeline(device, batch_size)
    for _ in range(16):
        results, inputs, tiles, ratios, angles = pipe.run()
        if device == "gpu":
            results, inputs = results.as_cpu(), inputs.as_cpu()
        for i in range(batch_size):
            tile = np.int32(tiles[i])
            ratio = np.float32(ratios[i])
            angle = np.float32(angles[i])
            check(results[i], inputs[i], tile, ratio, angle)


@params(*devices)
def test_gridmask_vs_cv_random(device):
    batch_size = 4
    run_random_test(batch_size, device)
