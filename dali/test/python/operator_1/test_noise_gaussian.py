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
import os
from test_utils import get_dali_extra_path, check_batch

test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, "db", "single", "jpeg")


@pipeline_def
def pipe_gaussian_noise(mean, stddev, variable_dist_params, device=None):
    encoded, _ = fn.readers.file(file_root=images_dir)
    in_data = fn.cast(
        fn.decoders.image(encoded, device="cpu", output_type=types.RGB), dtype=types.FLOAT
    )
    if device == "gpu":
        in_data = in_data.gpu()
    mean_arg = mean
    stddev_arg = stddev
    if variable_dist_params:
        mean_arg = fn.random.uniform(range=(-50.0, 50.0))
        stddev_arg = fn.random.uniform(range=(1.0, 10.0))
    seed = 12345
    out_data1 = fn.noise.gaussian(in_data, mean=mean_arg, stddev=stddev_arg, seed=seed)
    out_data2 = in_data + fn.random.normal(in_data, mean=mean_arg, stddev=stddev_arg, seed=seed)
    return out_data1, out_data2


def _testimpl_operator_noise_gaussian_vs_add_normal_dist(
    device, mean, stddev, variable_dist_params, batch_size, niter
):
    pipe = pipe_gaussian_noise(
        mean,
        stddev,
        variable_dist_params,
        device=device,
        batch_size=batch_size,
        num_threads=3,
        device_id=0,
    )
    for _ in range(niter):
        out0, out1 = pipe.run()
        check_batch(out0, out1, batch_size=batch_size, eps=0.1)


def test_operator_noise_gaussian_vs_add_normal_dist():
    niter = 3
    for device in ("cpu", "gpu"):
        for batch_size in (1, 3):
            for mean, stddev, variable_dist_params in [(10.0, 57.0, False), (0.0, 0.0, True)]:
                yield (
                    _testimpl_operator_noise_gaussian_vs_add_normal_dist,
                    device,
                    mean,
                    stddev,
                    variable_dist_params,
                    batch_size,
                    niter,
                )
