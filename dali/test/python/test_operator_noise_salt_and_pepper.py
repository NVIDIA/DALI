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

def salt_and_pepper_noise_ref(x, prob, salt_to_pepper_prob):
    x = np.array(x, dtype=np.float32)
    salt_prob = prob * salt_to_pepper_prob
    pepper_prob = prob * (1.0 - salt_to_pepper_prob)
    mask = np.random.choice(
        [0., 1., np.nan], p=[pepper_prob, 1 - prob, salt_prob], size=x.shape
    )
    y = np.where(np.isnan(mask), 255., x * mask).astype(np.uint8)
    return y

@pipeline_def
def pipe_salt_and_pepper_noise(prob, salt_to_pepper_prob, device='cpu'):
    encoded, _ = fn.readers.file(file_root=images_dir)
    in_data = fn.decoders.image(encoded, device="cpu", output_type=types.RGB)
    if device == 'gpu':
        in_data = in_data.gpu()
    prob_arg = prob or fn.random.uniform(range=(0.05, 0.5))
    salt_to_pepper_prob_arg = salt_to_pepper_prob or fn.random.uniform(range=(0.25, 0.75))
    out_data = fn.noise.salt_and_pepper(
        in_data, prob=prob_arg, salt_to_pepper_prob=salt_to_pepper_prob_arg)
    return in_data, out_data, prob_arg, salt_to_pepper_prob_arg

def _testimpl_operator_noise_salt_and_pepper(device, prob, salt_to_pepper_prob, batch_size, niter):
    pipe = pipe_salt_and_pepper_noise(prob, salt_to_pepper_prob, device=device, batch_size=batch_size,
                                      num_threads=3, device_id=0, seed=12345)
    pipe.build()
    for _ in range(niter):
        out_data, in_data, prob_arg, salt_to_pepper_prob_arg = pipe.run()
        prob_arg = prob_arg.as_array()
        salt_to_pepper_prob_arg = salt_to_pepper_prob_arg.as_array()
        if device == 'gpu':
            out_data = out_data.as_cpu()
            in_data = in_data.as_cpu()
        for s in range(batch_size):
            sample_in = np.array(out_data[s])
            sample_out = np.array(in_data[s])
            prob = float(prob_arg[s])
            salt_to_pepper_prob = float(salt_to_pepper_prob_arg[s])
            sample_ref = salt_and_pepper_noise_ref(sample_in, prob, salt_to_pepper_prob)
            psnr_out = PSNR(sample_out, sample_in)
            psnr_ref = PSNR(sample_ref, sample_in)
            if dump_images:
                import cv2
                cv2.imwrite(f"./salt_and_peppernoise_ref_p{prob}_{salt_to_pepper_prob}_s{s}.png", cv2.cvtColor(sample_ref, cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"./salt_and_peppernoise_out_p{prob}_{salt_to_pepper_prob}_s{s}.png", cv2.cvtColor(sample_out, cv2.COLOR_BGR2RGB))
            np.testing.assert_allclose(psnr_out, psnr_ref, atol=1)

def test_operator_noise_salt_and_pepper():
    niter = 3
    probs = [None, 0.2, 0.5, 0.8]
    salt_and_pepper_probs = [None, 1.0, 0.5, 0.0]
    for device in ("cpu", "gpu"):
        for prob in probs:
            for salt_and_pepper_prob in salt_and_pepper_probs:
                batch_size = random.choice([1, 3])
                yield _testimpl_operator_noise_salt_and_pepper, device, prob, salt_and_pepper_prob, batch_size, niter
