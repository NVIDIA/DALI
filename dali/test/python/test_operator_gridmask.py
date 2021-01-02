# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

def get_mask(w, h, tile, ratio, angle):
  black = round(tile * ratio)
  diag = math.sqrt(w**2 + h**2)
  nrep = int(math.ceil(diag / tile))

  angle_deg = angle * 180 / math.pi
  R = cv2.getRotationMatrix2D((tile * nrep, tile * nrep), angle_deg, 1)
  R[0,2] -= tile * nrep
  R[1,2] -= tile * nrep

  mask = np.ones((tile, tile))
  mask[0:black,0:black] = 0
  mask = np.tile(mask, (2 * nrep, 2 * nrep))
  mask = cv2.warpAffine(mask, R, (w, h))

  return mask


def check(result, input, tile, ratio, angle):
  result = np.uint8(result)
  input = np.uint8(input)
  mask = get_mask(result.shape[1], result.shape[0], tile, ratio, angle)
  ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

  # inside of squares should be black
  mask2 = np.uint8(1 - cv2.dilate(mask, ker))
  result2 = cv2.bitwise_and(result, result, mask=mask2)
  assert not np.any(result2)

  # outside of squares should be same as input
  mask2 = np.uint8(cv2.erode(mask, ker))
  result2 = cv2.bitwise_and(result, result, mask=mask2)
  input2 = cv2.bitwise_and(input, input, mask=mask2)
  assert np.all(result2 == input2)

def check_grid_mask(batch_size, tile, ratio, angle):
  pipe = get_pipeline(batch_size, tile, ratio, angle)
  pipe.build()
  results, inputs = pipe.run()
  for i in range(batch_size):
    check(results[i], inputs[i], tile, ratio, angle)

def test_cpu_vs_cv():
  for tile in [40, 100, 200]:
    for ratio in [0.2, 0.5, 0.8]:
      for angle in [0.0, 0.34, -0.62]:
        yield check_grid_mask, 4, tile, ratio, angle
