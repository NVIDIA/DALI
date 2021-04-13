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

from nvidia.dali import pipeline, pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os
import cv2

test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'tiff')
dump_images = False

def _testimpl_jpeg_compression_distortion(batch_size, device, quality):
  @pipeline_def()
  def jpeg_distortion_pipe(device='cpu', quality=None):
    encoded, _ = fn.readers.file(file_root=images_dir)
    in_images = fn.decoders.image(encoded, device='cpu')
    if quality is None:
      quality = fn.random.uniform(range=[1, 99], dtype=types.INT32)
    images = in_images.gpu() if device == 'gpu' else in_images
    out_images = fn.jpeg_compression_distortion(images, quality=quality)
    return out_images, in_images, quality

  pipe = jpeg_distortion_pipe(device=device, quality=quality,
                              batch_size=batch_size, num_threads=2, device_id=0)
  pipe.build()
  for _ in range(3):
    out = pipe.run()
    out_images = out[0].as_cpu() if device == 'gpu' else out[0]
    in_images = out[1]
    quality = out[2]
    for i in range(batch_size):
      out_img = np.array(out_images[i])
      in_img = np.array(in_images[i])
      q = int(np.array(quality[i]))

      bgr = cv2.cvtColor(in_img, cv2.COLOR_RGB2BGR)
      encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
      _, encoded_img = cv2.imencode('.jpg', bgr, params=encode_params)

      decoded_img_bgr = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
      decoded_img = cv2.cvtColor(decoded_img_bgr, cv2.COLOR_BGR2RGB)

      if dump_images:
        cv2.imwrite(f"./reference_q{q}_sample{i}.bmp", cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"./output_q{q}_sample{i}.bmp", cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))

      diff = cv2.absdiff(out_img, decoded_img)
      assert np.average(diff) < 5, f"Absolute difference with the reference is too big: {np.average(diff)}"

def test_jpeg_compression_distortion():
  for batch_size in [1, 15]:
    for device in ['cpu', 'gpu']:
      for quality in [2, None, 50]:
        yield _testimpl_jpeg_compression_distortion, batch_size, device, quality
