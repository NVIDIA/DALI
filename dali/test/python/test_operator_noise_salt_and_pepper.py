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

def salt_and_pepper_noise_ref(x, prob, salt_to_pepper_prob, per_channel=False):
    x = np.array(x, dtype=np.float32)
    salt_prob = prob * salt_to_pepper_prob
    pepper_prob = prob * (1.0 - salt_to_pepper_prob)
    if not per_channel:
        ndim = len(x.shape)
        channel_dim = ndim - 1
        nchannels = x.shape[channel_dim]
        mask_sh = x.shape[:-1]
        mask = np.random.choice(
            [0., 1., np.nan], p=[pepper_prob, 1 - prob, salt_prob], size=mask_sh
        )
        mask = np.stack([mask] * nchannels, axis=channel_dim)
    else:
        mask = np.random.choice(
            [0., 1., np.nan], p=[pepper_prob, 1 - prob, salt_prob], size=x.shape
        )
    y = np.where(np.isnan(mask), 255., x * mask).astype(np.uint8)
    return y

@pipeline_def
def pipe_salt_and_pepper_noise(prob, salt_to_pepper_prob, channel_first, per_channel=False, device='cpu'):
    encoded, _ = fn.readers.file(file_root=images_dir)
    in_data = fn.decoders.image(encoded, output_type=types.RGB)
    if device == 'gpu':
        in_data = in_data.gpu()
    if channel_first:
        in_data = fn.transpose(in_data, perm=[2, 0, 1])
    prob_arg = prob or fn.random.uniform(range=(0.05, 0.5))
    salt_to_pepper_prob_arg = salt_to_pepper_prob or fn.random.uniform(range=(0.25, 0.75))
    out_data = fn.noise.salt_and_pepper(
        in_data, per_channel=per_channel, prob=prob_arg, salt_to_pepper_prob=salt_to_pepper_prob_arg)
    return in_data, out_data, prob_arg, salt_to_pepper_prob_arg

def verify_salt_and_pepper(output, input, prob, salt_to_pepper_prob, per_channel):
    assert output.shape == input.shape
    height, width, nchannels = output.shape
    npixels = height * width
    salt_count = 0
    pepper_count = 0
    pixel_count = 0
    pepper_value = 0
    salt_value = 255
    if per_channel:
        output = np.reshape(output, (npixels * nchannels, 1))
        input = np.reshape(input, (npixels * nchannels, 1))
    passthrough_mask = np.all(output == input, axis=-1)
    pepper_mask = np.all(output == pepper_value, axis=-1)
    salt_mask = np.all(output == salt_value, axis=-1)
    pixel_mask = np.logical_and(np.all(input != pepper_value, axis=-1),
                                np.all(input != salt_value, axis=-1))
    salt_count = np.count_nonzero(np.logical_and(salt_mask, pixel_mask))
    pepper_count = np.count_nonzero(np.logical_and(pepper_mask, pixel_mask))
    pixel_count = np.count_nonzero(pixel_mask)
    assert np.logical_or(passthrough_mask,
                         np.logical_or(salt_mask, pepper_mask)).all()
    actual_noise_prob = (pepper_count + salt_count) / pixel_count
    actual_salt_to_pepper_prob = salt_count / (salt_count + pepper_count)
    np.testing.assert_allclose(actual_noise_prob, prob, atol=1e-2)
    np.testing.assert_allclose(actual_salt_to_pepper_prob, salt_to_pepper_prob, atol=1e-2)

def _testimpl_operator_noise_salt_and_pepper(device, per_channel, prob, salt_to_pepper_prob, channel_first, batch_size, niter):
    pipe = pipe_salt_and_pepper_noise(prob, salt_to_pepper_prob, channel_first, per_channel=per_channel,
                                      device=device, batch_size=batch_size,
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
            if channel_first:  # Convert back to channel-last before verifying
                sample_out = np.transpose(sample_out, axes=(1, 2, 0))
                sample_in = np.transpose(sample_in, axes=(1, 2, 0))
            prob = float(prob_arg[s])
            salt_to_pepper_prob = float(salt_to_pepper_prob_arg[s])
            sample_ref = salt_and_pepper_noise_ref(
                sample_in, prob, salt_to_pepper_prob, per_channel=per_channel)
            psnr_out = PSNR(sample_out, sample_in)
            psnr_ref = PSNR(sample_ref, sample_in)
            if dump_images:
                import cv2
                suffix_str = f"{prob}_{salt_to_pepper_prob}_s{s}"
                if not per_channel:
                    suffix_str = suffix_str + "_monochrome"
                cv2.imwrite(f"./snp_noise_ref_p{suffix_str}.png", cv2.cvtColor(sample_ref, cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"./snp_noise_out_p{suffix_str}.png", cv2.cvtColor(sample_out, cv2.COLOR_BGR2RGB))
            np.testing.assert_allclose(psnr_out, psnr_ref, atol=1)
            verify_salt_and_pepper(sample_out, sample_in, prob, salt_to_pepper_prob, per_channel)

def test_operator_noise_salt_and_pepper():
    niter = 3
    probs = [None, 0.021, 0.5]
    salt_and_pepper_probs = [None, 1.0, 0.5, 0.0]
    for device in ["cpu",]:
        for per_channel in [False, True]:
            for channel_first in [False, True]:
                for prob in probs:
                    salt_and_pepper_prob = random.choice(salt_and_pepper_probs)
                    batch_size = random.choice([1, 3])
                    yield _testimpl_operator_noise_salt_and_pepper, \
                        device, per_channel, prob, salt_and_pepper_prob, channel_first, batch_size, niter
