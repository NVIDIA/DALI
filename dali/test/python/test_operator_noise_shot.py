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

from nvidia.dali import pipeline_def
from nvidia.dali.backend_impl import TensorListGPU
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import random
import numpy as np
import os
from test_utils import get_dali_extra_path, check_batch
from test_noise_utils import PSNR

test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, 'db', 'single', 'png')
dump_images = False

def shot_noise_ref(x, peak):
    x = np.array(x) / 255.
    return (np.clip(np.random.poisson(x * peak) / peak, 0, 1) * 255).astype(np.uint8)

@pipeline_def
def pipe_shot_noise(peak, device='cpu'):
    encoded, _ = fn.readers.file(file_root=images_dir)
    in_data = fn.decoders.image(encoded, device="cpu", output_type=types.RGB)
    if device == 'gpu':
        in_data = in_data.gpu()
    peak_arg = peak or fn.random.uniform(range=(3.0, 60.0))
    out_data = fn.noise.shot(in_data, peak=peak_arg)
    return in_data, out_data, peak_arg

def _testimpl_operator_noise_shot(device, peak, batch_size, niter):
    pipe = pipe_shot_noise(peak, device=device, batch_size=batch_size,
                           num_threads=3, device_id=0, seed=12345)
    pipe.build()
    for _ in range(niter):
        out_data, in_data, peak_arg = pipe.run()
        peak_arg = peak_arg.as_array()
        if device == 'gpu':
            out_data = out_data.as_cpu()
            in_data = in_data.as_cpu()
        for s in range(batch_size):
            sample_in = np.array(out_data[s])
            sample_out = np.array(in_data[s])
            peak = peak_arg[s]
            sample_ref = shot_noise_ref(sample_in, peak)
            psnr_out = PSNR(sample_out, sample_in)
            psnr_ref = PSNR(sample_ref, sample_in)
            np.testing.assert_allclose(psnr_out, psnr_ref, atol=1)
            if dump_images:
                import cv2
                cv2.imwrite(f"./shotnoise_ref_p{peak}_s{s}.png", cv2.cvtColor(sample_ref, cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"./shotnoise_out_p{peak}_s{s}.png", cv2.cvtColor(sample_out, cv2.COLOR_BGR2RGB))

def test_operator_noise_shot():
    niter = 3
    peaks = [None, 3.0, 10.0, 30.0, 60.0]
    for device in ("cpu", "gpu"):
        for batch_size in (1, 3):
            for peak in peaks:
                yield _testimpl_operator_noise_shot, device, peak, batch_size, niter
