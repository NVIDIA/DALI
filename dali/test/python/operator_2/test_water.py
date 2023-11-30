# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nose_utils  # noqa:F401   - for Python 3.10
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali
import numpy as np
import os
import cv2
from test_utils import compare_pipelines
from test_utils import get_dali_extra_path

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, "db", "lmdb")


class WaterPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        phase_y,
        phase_x,
        freq_x,
        freq_y,
        ampl_x,
        ampl_y,
        num_threads=3,
        device_id=0,
        num_gpus=1,
        dtype=types.UINT8,
        prime_size=False,
        do_mask=False,
    ):
        super(WaterPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.dtype = dtype
        self.prime_size = prime_size
        self.do_mask = do_mask
        self.input = ops.readers.Caffe(
            path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
        )
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
        self.water = ops.Water(
            device=self.device,
            ampl_x=ampl_x,
            ampl_y=ampl_y,
            phase_x=phase_x,
            phase_y=phase_y,
            freq_x=freq_x,
            freq_y=freq_y,
            interp_type=dali.types.INTERP_LINEAR,
        )

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if self.device == "gpu":
            images = images.gpu()
        if self.prime_size:
            images = fn.resize(images, resize_x=101, resize_y=43)
        mask = fn.random.coin_flip(seed=42) if self.do_mask else None
        images = fn.cast(images, dtype=self.dtype)
        images = self.water(images, mask=mask)
        return images


def python_water(img, phase_y, phase_x, freq_x, freq_y, ampl_x, ampl_y):
    nh, nw = img.shape[:2]
    img_x = np.zeros((nh, nw), np.float32)
    img_y = np.zeros((nh, nw), np.float32)
    x_idx = np.arange(0, nw, 1, np.float32)
    y_idx = np.arange(0, nh, 1, np.float32)
    x_wave = ampl_y * np.cos(freq_y * x_idx + phase_y)
    y_wave = ampl_x * np.sin(freq_x * y_idx + phase_x)
    for x in range(nw):
        img_x[:, x] = y_wave + x - 0.5

    for y in range(nh):
        img_y[y, :] = x_wave + y - 0.5

    return cv2.remap(img, img_x, img_y, cv2.INTER_LINEAR)


class WaterPythonPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        function,
        num_threads=1,
        device_id=0,
        num_gpus=1,
        dtype=types.UINT8,
        prime_size=False,
    ):
        super().__init__(batch_size, num_threads, device_id, exec_async=False, exec_pipelined=False)
        self.dtype = dtype
        self.prime_size = prime_size
        self.input = ops.readers.Caffe(
            path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
        )
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)

        self.water = ops.PythonFunction(function=function, output_layouts="HWC")

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if self.prime_size:
            images = fn.resize(images, resize_x=101, resize_y=43)
        images = fn.cast(images, dtype=self.dtype)
        images = self.water(images)
        return images


def check_water_cpu_vs_gpu(batch_size, niter, dtype, do_mask):
    phase_y = 0.5
    phase_x = 0.2
    freq_x = 0.06
    freq_y = 0.08
    ampl_x = 2.0
    ampl_y = 3.0
    compare_pipelines(
        WaterPipeline(
            "cpu",
            batch_size,
            ampl_x=ampl_x,
            ampl_y=ampl_y,
            phase_x=phase_x,
            phase_y=phase_y,
            freq_x=freq_x,
            freq_y=freq_y,
            dtype=dtype,
            do_mask=do_mask,
        ),
        WaterPipeline(
            "gpu",
            batch_size,
            ampl_x=ampl_x,
            ampl_y=ampl_y,
            phase_x=phase_x,
            phase_y=phase_y,
            freq_x=freq_x,
            freq_y=freq_y,
            dtype=dtype,
            do_mask=do_mask,
        ),
        batch_size=batch_size,
        N_iterations=niter,
        eps=1,
    )


def test_water_cpu_vs_gpu():
    niter = 3
    for batch_size in [1, 3]:
        for do_mask in [False, True]:
            for dtype in [types.UINT8, types.FLOAT]:
                yield check_water_cpu_vs_gpu, batch_size, niter, dtype, do_mask


def check_water_vs_cv(device, batch_size, niter, dtype, prime_size):
    phase_y = 0.5
    phase_x = 0.2
    freq_x = 0.06
    freq_y = 0.08
    ampl_x = 2.0
    ampl_y = 3.0

    def python_func(img):
        return python_water(img, phase_y, phase_x, freq_x, freq_y, ampl_x, ampl_y)

    compare_pipelines(
        WaterPipeline(
            device,
            batch_size,
            ampl_x=ampl_x,
            ampl_y=ampl_y,
            phase_x=phase_x,
            phase_y=phase_y,
            freq_x=freq_x,
            freq_y=freq_y,
            dtype=dtype,
            prime_size=prime_size,
        ),
        WaterPythonPipeline(batch_size, python_func, dtype=dtype, prime_size=prime_size),
        batch_size=batch_size,
        N_iterations=niter,
        eps=8,
    )


def test_water_vs_cv():
    niter = 3
    for device in ["cpu", "gpu"]:
        for batch_size in [1, 3]:
            for dtype in [types.UINT8, types.FLOAT]:
                for prime_size in [False, True]:
                    yield check_water_vs_cv, device, batch_size, niter, dtype, prime_size
