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
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import math
import os
import cv2
from test_utils import get_dali_extra_path

data_root = get_dali_extra_path()
img_dir = os.path.join(data_root, 'db', 'single', 'jpeg')

def get_pipeline(batch_size, tile, ratio, angle):
  pipe = Pipeline(batch_size, 4, None)
  with pipe:
    input, _ = fn.file_reader(file_root=img_dir)
    decoded = fn.image_decoder(input, device='cpu', output_type=types.RGB)
    grided = fn.grid_mask(decoded, device='cpu', tile=tile, ratio=ratio, angle=angle)
    pipe.set_outputs(grided, decoded)
  return pipe

def get_random_pipeline(batch_size):
  pipe = Pipeline(batch_size, 4, None)
  with pipe:
    input, _ = fn.file_reader(file_root=img_dir)
    decoded = fn.image_decoder(input, device='cpu', output_type=types.RGB)
    tile = fn.cast(fn.uniform(range=(50, 200), shape=[1]), dtype=types.INT32)
    ratio = fn.uniform(range=(0.3, 0.7), shape=[1])
    angle = fn.uniform(range=(-math.pi, math.pi), shape=[1])
    grided = fn.grid_mask(decoded, device='cpu', tile=tile, ratio=ratio, angle=angle)
    pipe.set_outputs(grided, decoded, tile, ratio, angle)
  return pipe

def get_mask(w, h, tile, ratio, angle, d):
  black = int(round(tile * ratio))
  diag = math.sqrt(w**2 + h**2) + 1
  nrep = int(math.ceil(diag / tile))

  mask = np.ones((tile, tile))
  mask[:black, :black] = 0
  mask = np.tile(mask, (2 * nrep, 2 * nrep))

  p = tile * nrep
  c = math.cos(angle)
  s = math.sin(angle)
  R = np.array([
    [ c, s, -p * c - p * s + p],
    [-s, c, -p * c + p * s + p]])
  mask = cv2.warpAffine(mask, R, (2 * p, 2 * p))

  # shrink or expand the rotated squares
  ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  if d == 1:
    mask = cv2.erode(mask, ker)
  elif d == -1:
    mask = cv2.dilate(mask, ker)

  return mask[p:p+h, p:p+w]


def check(result, input, tile, ratio, angle):
  result = np.uint8(result)
  input = np.uint8(input)
  w = result.shape[1]
  h = result.shape[0]

  # inside of squares should be black
  mask = np.uint8(1 - get_mask(w, h, tile, ratio, angle, -1))
  result2 = cv2.bitwise_and(result, result, mask=mask)
  assert not np.any(result2)

  # outside of squares should be same as input
  mask = np.uint8(get_mask(w, h, tile, ratio, angle, 1))
  result2 = cv2.bitwise_and(result, result, mask=mask)
  input2 = cv2.bitwise_and(input, input, mask=mask)
  assert np.all(result2 == input2)

def test_cpu_vs_cv():
  batch_size = 4
  for tile in [40, 100, 200]:
    for ratio in [0.2, 0.5, 0.8]:
      for angle in [0.0, 0.34, -0.62]:
        pipe = get_pipeline(batch_size, tile, ratio, angle)
        pipe.build()
        results, inputs = pipe.run()
        for i in range(batch_size):
          yield check, results[i], inputs[i], tile, ratio, angle

def test_cpu_vs_cv_random():
  batch_size = 4
  pipe = get_random_pipeline(batch_size)
  pipe.build()
  for _ in range(16):
    results, inputs, tiles, ratios, angles = pipe.run()
    for i in range(batch_size):
      tile = np.int32(tiles[i])[0]
      ratio = np.float32(ratios[i])[0]
      angle = np.float32(angles[i])[0]
      yield check, results[i], inputs[i], tile, ratio, angle
