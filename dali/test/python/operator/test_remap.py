# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali as dali
import nvidia.dali.fn as fn
import os.path
import unittest
import time
from nose2.tools import params
from nvidia.dali.pipeline.experimental import pipeline_def
from nvidia.dali.types import DALIInterpType

test_data_root = os.environ['DALI_EXTRA_PATH']
data_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')

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
        map_x = np.zeros(shape, dtype=np.float32)
        map_y = np.zeros(shape, dtype=np.float32)
        if mode == 'identity':
            for i in range(map_x.shape[0]):
                map_x[i, :] = [x for x in range(map_x.shape[1])]
            for j in range(map_y.shape[1]):
                map_y[:, j] = [y for y in range(map_y.shape[0])]
        elif mode == 'xflip':
            for i in range(map_x.shape[0]):
                map_x[i, :] = [x for x in range(map_x.shape[1])]
            for j in range(map_y.shape[1]):
                map_y[:, j] = [map_y.shape[0] - y for y in range(map_y.shape[0])]
        elif mode == 'yflip':
            for i in range(map_x.shape[0]):
                map_x[i, :] = [map_x.shape[1] - x for x in range(map_x.shape[1])]
            for j in range(map_y.shape[1]):
                map_y[:, j] = [y for y in range(map_y.shape[0])]
        elif mode == 'xyflip':
            for i in range(map_x.shape[0]):
                map_x[i, :] = [map_x.shape[1] - x for x in range(map_x.shape[1])]
            for j in range(map_y.shape[1]):
                map_y[:, j] = [map_y.shape[0] - y for y in range(map_y.shape[0])]
        elif mode == 'random':
            map_x = rng.uniform(low=0, high=map_x.shape[1] + 0, size=map_x.shape)
            map_y = rng.uniform(low=0, high=map_y.shape[0] + 0, size=map_y.shape)
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
    if remap_op == 'dali':
        return fn.experimental.remap(img.gpu(), mapx.gpu(), mapy.gpu(),
                                     interp=DALIInterpType.INTERP_NN, device='gpu',
                                     pixel_origin="center")
    elif remap_op == 'cv':
        return fn.python_function(img, mapx, mapy, function=_cv_remap)
    else:
        raise ValueError("Unknown remap operator.")


class RemapTest(unittest.TestCase):
    def setUp(self):
        self.img_size = (480, 640)
        self.batch_size = 3
        self.common_dali_pipe_params = {
            "batch_size": self.batch_size,
            "num_threads": 3,
            "device_id": 0,
            # "exec_async": False,
            # "exec_pipelined": False,
        }

    @params('identity', 'xflip', 'yflip', 'xyflip', 'random')
    def test_remap(self, map_mode):
        maps = [update_map(mode=map_mode, shape=self.img_size, nimages=self.batch_size)]
        dpipe = remap_pipe('dali', maps, self.img_size, **self.common_dali_pipe_params)
        cpipe = remap_pipe('cv', maps, self.img_size, exec_async=False, exec_pipelined=False, **self.common_dali_pipe_params)
        self._compare_pipelines_pixelwise(dpipe, cpipe, N_iterations=2, eps=.01)

    def _compare_pipelines_pixelwise(self, pipe1, pipe2, N_iterations, eps=.01):
        pipe1.build()
        pipe2.build()
        for _ in range(N_iterations):
            out1 = pipe1.run()
            out2 = pipe2.run()
            self.assertTrue(
                len(out1) == len(out2),
                f"Numbers of outputs in the pipelines does not match: {len(out1)} vs {len(out2)}.")
            for i in range(len(out1)):
                out1_data = out1[i].as_cpu() \
                    if isinstance(out1[i][0], dali.backend_impl.TensorGPU) else out1[i]
                out2_data = out2[i].as_cpu() \
                    if isinstance(out2[i][0], dali.backend_impl.TensorGPU) else out2[i]
                for sample1, sample2 in zip(out1_data, out2_data):
                    s1 = np.array(sample1)
                    s2 = np.array(sample2)
                    self.assertTrue(s1.shape == s2.shape,
                                    f"Sample shapes do not match: {s1.shape} vs {s2.shape}")
                    noutliers = self._count_outlying_pixels(s1, s2)
                    size = np.prod(s1.shape[:-1])
                    self.assertTrue(
                        noutliers / size < eps,
                        f"Test failed. Actual error: {noutliers / size}, expected: {eps}.")

    @params('random')
    def test_benchmark_remap_vs_opencv(self, map_mode):
        maps = [update_map(mode=map_mode, shape=self.img_size, nimages=self.batch_size)]
        dpipe = remap_pipe('dali', maps, self.img_size, **self.common_dali_pipe_params)
        cpipe = remap_pipe('cv', maps, self.img_size, exec_async=False, exec_pipelined=False, **self.common_dali_pipe_params)
        dpipe.build()
        cpipe.build()
        dpipe.run()
        cpipe.run()
        n_iterations=1000
        dtime = self._measure_time(dpipe.run,n_iterations)
        ctime = self._measure_time(cpipe.run,n_iterations)
        print(f"Average DALI Remap pipeline execution time: {dtime/n_iterations}")
        print(f"Average OpenCV Remap pipeline execution time: {ctime/n_iterations}")
        self.assertTrue(dtime < ctime)

    @staticmethod
    def _measure_time(func, n_iterations=1000):
        start = time.time()
        for _ in range(n_iterations):
            func()
        stop = time.time()
        return stop - start

    @staticmethod
    def _count_outlying_pixels(sample1, sample2):
        diff = sample1 - sample2
        diff = diff.reshape(-1, diff.shape[-1])
        sum = 0
        for px in diff:
            if not np.all(px == np.zeros(3)):
                sum += 1
        return sum
