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

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import random
import numpy as np
import os
from test_utils import get_dali_extra_path
from test_noise_utils import PSNR

test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, "db", "single", "png")
dump_images = False


def salt_and_pepper_noise_ref(x, prob, salt_vs_pepper, per_channel, salt_val, pepper_val):
    x = np.array(x, dtype=np.float32)
    salt_prob = prob * salt_vs_pepper
    pepper_prob = prob * (1.0 - salt_vs_pepper)
    nchannels = x.shape[-1]
    mask_sh = x.shape if per_channel else x.shape[:-1]
    mask = np.random.choice(
        [pepper_val, np.nan, salt_val], p=[pepper_prob, 1 - prob, salt_prob], size=mask_sh
    )
    if not per_channel:
        mask = np.stack([mask] * nchannels, axis=-1)
    y = np.where(np.isnan(mask), x, mask).astype(np.uint8)
    return y


@pipeline_def
def pipe_salt_and_pepper_noise(
    prob, salt_vs_pepper, channel_first, per_channel, salt_val, pepper_val, device="cpu"
):
    encoded, _ = fn.readers.file(file_root=images_dir)
    in_data = fn.decoders.image(encoded, output_type=types.RGB)
    if device == "gpu":
        in_data = in_data.gpu()
    if channel_first:
        in_data = fn.transpose(in_data, perm=[2, 0, 1])
    prob_arg = prob or fn.random.uniform(range=(0.05, 0.5))
    salt_vs_pepper_arg = salt_vs_pepper or fn.random.uniform(range=(0.25, 0.75))
    out_data = fn.noise.salt_and_pepper(
        in_data,
        per_channel=per_channel,
        prob=prob_arg,
        salt_vs_pepper=salt_vs_pepper_arg,
        salt_val=salt_val,
        pepper_val=pepper_val,
    )
    return in_data, out_data, prob_arg, salt_vs_pepper_arg


def verify_salt_and_pepper(output, input, prob, salt_vs_pepper, per_channel, salt_val, pepper_val):
    assert output.shape == input.shape
    height, width, nchannels = output.shape
    npixels = height * width
    salt_count = 0
    pepper_count = 0
    pixel_count = 0
    if per_channel:
        output = np.reshape(output, (npixels * nchannels, 1))
        input = np.reshape(input, (npixels * nchannels, 1))
    passthrough_mask = np.all(output == input, axis=-1)
    pepper_mask = np.all(output == pepper_val, axis=-1)
    salt_mask = np.all(output == salt_val, axis=-1)
    # This mask is meant to select only the pixels that didn't have a 'salt' or 'pepper'
    # value before the noise application. Otherwise, the measured noise/pepper percentages,
    # might differ a lot in images with a lot of black or white pixels.
    in_pixel_mask = np.logical_and(
        np.all(input != pepper_val, axis=-1), np.all(input != salt_val, axis=-1)
    )
    salt_count = np.count_nonzero(np.logical_and(salt_mask, in_pixel_mask))
    pepper_count = np.count_nonzero(np.logical_and(pepper_mask, in_pixel_mask))
    pixel_count = np.count_nonzero(in_pixel_mask)
    assert (np.logical_or(passthrough_mask, np.logical_or(salt_mask, pepper_mask))).all()
    actual_noise_prob = (pepper_count + salt_count) / pixel_count
    actual_salt_vs_pepper = salt_count / (salt_count + pepper_count)
    np.testing.assert_allclose(actual_noise_prob, prob, atol=1e-2)
    np.testing.assert_allclose(actual_salt_vs_pepper, salt_vs_pepper, atol=1e-1)


def _testimpl_operator_noise_salt_and_pepper(
    device,
    per_channel,
    prob,
    salt_vs_pepper,
    channel_first,
    salt_val,
    pepper_val,
    batch_size,
    niter,
):
    pipe = pipe_salt_and_pepper_noise(
        prob,
        salt_vs_pepper,
        channel_first,
        per_channel,
        salt_val,
        pepper_val,
        device=device,
        batch_size=batch_size,
        num_threads=3,
        device_id=0,
        seed=12345,
    )
    salt_val = 255 if salt_val is None else salt_val
    pepper_val = 0 if pepper_val is None else pepper_val
    for _ in range(niter):
        out_data, in_data, prob_arg, salt_vs_pepper_arg = pipe.run()
        prob_arg = prob_arg.as_array()
        salt_vs_pepper_arg = salt_vs_pepper_arg.as_array()
        if device == "gpu":
            out_data = out_data.as_cpu()
            in_data = in_data.as_cpu()
        for s in range(batch_size):
            sample_in = np.array(out_data[s])
            sample_out = np.array(in_data[s])
            if channel_first:  # Convert back to channel-last before verifying
                sample_out = np.transpose(sample_out, axes=(1, 2, 0))
                sample_in = np.transpose(sample_in, axes=(1, 2, 0))
            prob = float(prob_arg[s])
            salt_vs_pepper = float(salt_vs_pepper_arg[s])
            sample_ref = salt_and_pepper_noise_ref(
                sample_in, prob, salt_vs_pepper, per_channel, salt_val, pepper_val
            )
            psnr_out = PSNR(sample_out, sample_in)
            psnr_ref = PSNR(sample_ref, sample_in)
            if dump_images:
                import cv2

                suffix_str = f"{prob}_{salt_vs_pepper}_s{s}"
                if not per_channel:
                    suffix_str = suffix_str + "_monochrome"
                cv2.imwrite(
                    f"./snp_noise_ref_p{suffix_str}.png",
                    cv2.cvtColor(sample_ref, cv2.COLOR_BGR2RGB),
                )
                cv2.imwrite(
                    f"./snp_noise_out_p{suffix_str}.png",
                    cv2.cvtColor(sample_out, cv2.COLOR_BGR2RGB),
                )
            verify_salt_and_pepper(
                sample_out, sample_in, prob, salt_vs_pepper, per_channel, salt_val, pepper_val
            )
            np.testing.assert_allclose(psnr_out, psnr_ref, atol=1)


def test_operator_noise_salt_and_pepper():
    niter = 3
    probs = [None, 0.021, 0.5]
    salt_and_pepper_probs = [None, 1.0, 0.5, 0.0]
    for device in ["cpu", "gpu"]:
        for per_channel in [False, True]:
            for channel_first in [False, True]:
                for pepper_val, salt_val in [(None, None), (10, 50)]:
                    for prob in probs:
                        salt_and_pepper_prob = random.choice(salt_and_pepper_probs)
                        batch_size = random.choice([1, 3])
                        yield (
                            _testimpl_operator_noise_salt_and_pepper,
                            device,
                            per_channel,
                            prob,
                            salt_and_pepper_prob,
                            channel_first,
                            salt_val,
                            pepper_val,
                            batch_size,
                            niter,
                        )
