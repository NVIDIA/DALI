# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali as dali
import nvidia.dali.fn as fn
import cv2
import numpy as np
import multiprocessing as mp
import test_utils
from nose2.tools import params

NUM_THREADS = mp.cpu_count()
DEV_ID = 0
SEED = 1313


def cv2_median_blur(dst, img, ksize, layout):
    if layout[-1] == "C":
        cv2.medianBlur(img, ksize=ksize, dst=dst)
    else:
        for c in range(img.shape[0]):
            cv2.medianBlur(img[c, :, :], ksize, dst=dst[c, :, :])


def ref_func(img, ksize, layout):
    ksize = ksize[0]
    dst = np.zeros_like(img)
    if layout[0] == "F":
        for f in range(0, img.shape[0]):
            cv2_median_blur(dst[f, :, :, :], img[f, :, :, :], ksize, layout)
    else:
        cv2_median_blur(dst, img, ksize, layout)
    return dst


@dali.pipeline_def(
    num_threads=NUM_THREADS, device_id=DEV_ID, exec_pipelined=False, exec_async=False
)
def reference_pipe(data_src, layout, ksize_src):
    img = fn.external_source(source=data_src, batch=True, layout=layout)
    ksize = fn.external_source(source=ksize_src)
    return fn.python_function(
        img,
        ksize,
        function=lambda im, ks: ref_func(im, ks, layout=layout),
        batch_processing=False,
        output_layouts=layout,
    )


@dali.pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def median_blur_pipe(data_src, layout, ksize_src):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device="gpu")
    ksize = fn.external_source(source=ksize_src)
    ksize = fn.cat(ksize, ksize)
    return fn.experimental.median_blur(img, window_size=ksize)


@dali.pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def median_blur_cksize_pipe(data_src, layout, ksize):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device="gpu")
    return fn.experimental.median_blur(img, window_size=ksize)


# OpenCV requires ksize to be odd and greater than 1
def ksize_src(bs, lo, hi, seed):
    np_rng = np.random.default_rng(seed=seed)

    def gen_ksize():
        return np_rng.integers(lo // 2, hi // 2 + 1, size=(1), dtype=np.int32) * 2 + 1

    while True:
        ksize = [gen_ksize() for _ in range(bs)]
        yield ksize


@params(
    (32, "HWC", np.uint8, 3, 9),
    (32, "CHW", np.float32, 4, 5),
    (32, "HWC", np.uint16, 1, 5),
    (4, "FHWC", np.float32, 3, 5),
    (4, "FCHW", np.uint8, 1, 9),
)
def test_median_blur_vs_ocv(bs, layout, dtype, channels, max_ksize):
    cdim = layout.find("C")
    min_shape = [64 for c in layout]
    min_shape[cdim] = channels
    max_shape = [256 for c in layout]
    max_shape[cdim] = channels
    if layout[0] == "F":
        min_shape[0] = 8
        max_shape[0] = 32

    data1 = test_utils.RandomlyShapedDataIterator(
        batch_size=bs, min_shape=min_shape, max_shape=max_shape, dtype=dtype, seed=SEED
    )
    data2 = test_utils.RandomlyShapedDataIterator(
        batch_size=bs, min_shape=min_shape, max_shape=max_shape, dtype=dtype, seed=SEED
    )
    ksize1 = ksize_src(bs, 3, max_ksize, SEED)
    ksize2 = ksize_src(bs, 3, max_ksize, SEED)
    pipe1 = median_blur_pipe(
        data_src=data1, layout=layout, ksize_src=ksize1, batch_size=bs, prefetch_queue_depth=1
    )
    pipe2 = reference_pipe(data_src=data2, layout=layout, ksize_src=ksize2, batch_size=bs)
    test_utils.compare_pipelines(pipe1, pipe2, batch_size=bs, N_iterations=10)


@params(
    (32, "HWC", np.uint8, 3, (7, 7)),
    (32, "CHW", np.float32, 4, 3),
    (4, "FCHW", np.uint8, 1, (9, 9)),
)
def test_median_blur_const_ksize_vs_ocv(bs, layout, dtype, channels, ksize):
    cdim = layout.find("C")
    min_shape = [64 for c in layout]
    min_shape[cdim] = channels
    max_shape = [256 for c in layout]
    max_shape[cdim] = channels
    if layout[0] == "F":
        min_shape[0] = 8
        max_shape[0] = 32

    data1 = test_utils.RandomlyShapedDataIterator(
        batch_size=bs, min_shape=min_shape, max_shape=max_shape, dtype=dtype, seed=SEED
    )
    data2 = test_utils.RandomlyShapedDataIterator(
        batch_size=bs, min_shape=min_shape, max_shape=max_shape, dtype=dtype, seed=SEED
    )
    if isinstance(ksize, tuple):
        cv_ksize = ksize[0]
    else:
        cv_ksize = ksize
    ksize1 = ksize_src(bs, cv_ksize, cv_ksize, SEED)
    pipe1 = median_blur_cksize_pipe(
        data_src=data1, layout=layout, ksize=ksize, batch_size=bs, prefetch_queue_depth=1
    )
    pipe2 = reference_pipe(data_src=data2, layout=layout, ksize_src=ksize1, batch_size=bs)
    test_utils.compare_pipelines(pipe1, pipe2, batch_size=bs, N_iterations=10)
