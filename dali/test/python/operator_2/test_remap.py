# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cv2
import numpy as np
import nvidia.dali.fn as fn
import os.path
import unittest
import time
from nose2.tools import params
from nvidia.dali.pipeline.experimental import pipeline_def
from nvidia.dali.types import DALIInterpType

test_data_root = os.environ["DALI_EXTRA_PATH"]
data_dir = os.path.join(test_data_root, "db", "single", "jpeg")

rng = np.random.default_rng()


def update_map(mode, shape, nimages=1):
    """
    Code for map calculation.
    Based on https://github.com/opencv/opencv/blob/3.4/samples/python/tutorial_code/ImgTrans/remap/Remap_Demo.py  # noqa
    :param mode: One of: 'identity', 'xflip', 'yflip', 'xyflip', 'random'
    :param shape: HWC shape of a sample.
    :param nimages: Number of maps to be generated for every axis.
    :return: tuple of 2 ndarrays (mapx: [nimages, H, W, C], mapy: [nimages, H, W, C])
    """
    mapsx = []
    mapsy = []
    for _ in range(nimages):
        map_x = np.tile(np.arange(shape[1]), [shape[0], 1])
        map_y = np.tile(np.arange(shape[0])[:, np.newaxis], [1, shape[1]])
        if mode == "identity":
            pass
        elif mode == "xflip":
            map_x = shape[1] - map_x
        elif mode == "yflip":
            map_y = shape[0] - map_y
        elif mode == "xyflip":
            map_x = shape[1] - map_x
            map_y = shape[0] - map_y
        elif mode == "random":
            map_x = rng.uniform(low=0, high=map_x.shape[1] + 0, size=shape)
            map_y = rng.uniform(low=0, high=map_y.shape[0] + 0, size=shape)
        else:
            raise ValueError("Unknown map mode.")
        mapsx.append(map_x)
        mapsy.append(map_y)
    return np.array(mapsx, dtype=np.float32), np.array(mapsy, dtype=np.float32)


def _cv_remap(img, mapx, mapy):
    return cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)


@pipeline_def
def remap_pipe(remap_op, maps_data, img_size):
    """
    Returns either a reference pipeline or a pipeline under test.

    If the remap_op argument is 'dali', this function returns a DALI pipeline under test.
    If the remap_op argument is 'cv', this function returns a reference DALI pipeline.

    :param remap_op: 'dali' or 'cv'.
    :param maps_data: List of ndarrays, which contains data for the remap parameters (maps).
    :param img_size: Shape of the remap parameters, but without the channels value (only spatial).
    :return: DALI Pipeline
    """
    img, _ = fn.readers.file(file_root=data_dir)
    img = fn.decoders.image(img)
    img = fn.resize(img, size=img_size)
    mapx, mapy = fn.external_source(source=maps_data, batch=True, cycle=True, num_outputs=2)
    if remap_op == "dali":
        return fn.experimental.remap(
            img.gpu(),
            mapx.gpu(),
            mapy.gpu(),
            interp=DALIInterpType.INTERP_NN,
            device="gpu",
            pixel_origin="center",
        )
    elif remap_op == "cv":
        return fn.python_function(img, mapx, mapy, function=_cv_remap)
    else:
        raise ValueError("Unknown remap operator.")


class RemapTest(unittest.TestCase):
    def setUp(self):
        self.img_size = (480, 640)
        self.batch_size = 64
        self.common_dali_pipe_params = {
            "batch_size": self.batch_size,
            "num_threads": 1,
            "device_id": 0,
        }

    @params("identity", "xflip", "yflip", "xyflip", "random")
    def test_remap(self, map_mode):
        maps = [update_map(mode=map_mode, shape=self.img_size, nimages=self.batch_size)]
        dpipe = remap_pipe("dali", maps, self.img_size, **self.common_dali_pipe_params)
        cpipe = remap_pipe(
            "cv",
            maps,
            self.img_size,
            exec_async=False,
            exec_pipelined=False,
            **self.common_dali_pipe_params,
        )
        self._compare_pipelines_pixelwise(dpipe, cpipe, N_iterations=2, eps=0.01)

    def benchmark_remap_against_cv(self, map_mode):
        import torch.cuda.nvtx as nvtx

        nvtx.range_push("Benchmark against OpenCV")
        maps = [update_map(mode=map_mode, shape=self.img_size, nimages=self.batch_size)]
        dpipe = remap_pipe(
            "dali",
            maps,
            self.img_size,
            exec_async=False,
            exec_pipelined=False,
            **self.common_dali_pipe_params,
            prefetch_queue_depth=1,
        )
        cpipe = remap_pipe(
            "cv",
            maps,
            self.img_size,
            exec_async=False,
            exec_pipelined=False,
            **self.common_dali_pipe_params,
            prefetch_queue_depth=1,
        )
        dpipe.build()
        cpipe.build()
        dtime = self._measure_time(dpipe.run)
        ctime = self._measure_time(cpipe.run)
        nvtx.range_pop()
        print(f"DALI Pipeline average time: {dtime}. OpenCV Pipeline average time: {ctime}.")

    def benchmark_remap_isolated(self, map_mode):
        import torch.cuda.nvtx as nvtx

        nvtx.range_push("Benchmark isolated")
        maps = [update_map(mode=map_mode, shape=self.img_size, nimages=self.batch_size)]
        dpipe = remap_pipe(
            "dali", maps, self.img_size, **self.common_dali_pipe_params, prefetch_queue_depth=1
        )
        dpipe.build()
        avg_time = self._measure_time(dpipe.run)
        nvtx.range_pop()
        print(f"DALI Pipeline average execution time: {avg_time} seconds.")

    def _compare_pipelines_pixelwise(self, pipe1, pipe2, N_iterations, eps=0.01):
        pipe1.build()
        pipe2.build()
        for _ in range(N_iterations):
            out1 = tuple(out.as_cpu() for out in pipe1.run())
            out2 = tuple(out.as_cpu() for out in pipe2.run())
            self.assertTrue(
                len(out1) == len(out2),
                f"Numbers of outputs in the pipelines does not match: {len(out1)} vs {len(out2)}.",
            )
            for i in range(len(out1)):
                for sample1, sample2 in zip(out1[i], out2[i]):
                    s1 = np.array(sample1)
                    s2 = np.array(sample2)
                    self.assertTrue(
                        s1.shape == s2.shape,
                        f"Sample shapes do not match: {s1.shape} vs {s2.shape}",
                    )
                    noutliers = self._count_outlying_pixels(s1, s2)
                    size = np.prod(s1.shape[:-1])
                    self.assertTrue(
                        noutliers / size < eps,
                        f"Test failed. Actual error: {noutliers / size}, expected: {eps}.",
                    )

    @staticmethod
    def _measure_time(func, n_iterations=30):
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            func()
            stop = time.perf_counter()
            times.append(stop - start)
        return np.mean(np.array(times))

    @staticmethod
    def _count_outlying_pixels(sample1, sample2):
        eq = sample1 != sample2
        return np.count_nonzero(np.sum(eq, axis=2))
