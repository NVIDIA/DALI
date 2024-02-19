# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
SEED = 134


def ocv_border_mode(border_mode):
    if border_mode == "constant":
        return cv2.BORDER_CONSTANT
    elif border_mode == "replicate":
        return cv2.BORDER_REPLICATE
    elif border_mode == "reflect":
        return cv2.BORDER_REFLECT
    elif border_mode == "reflect_101":
        return cv2.BORDER_REFLECT_101
    elif border_mode == "wrap":
        return cv2.BORDER_WRAP
    else:
        raise ValueError("Invalid border mode")


def cv2_morph(dst, img, ksize, anchor, layout, border_mode, morph_type):
    morph = cv2.dilate if morph_type == "dilate" else cv2.erode
    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    def morph_func(img, dst):
        morph(img, struct_element, anchor=anchor, borderType=ocv_border_mode(border_mode), dst=dst)

    if layout[-1] == "C":
        morph_func(img, dst)
    else:
        for c in range(img.shape[0]):
            morph_func(img[c, :, :], dst[c, :, :])


def ref_func(img, ksize, anchor, layout, border_mode, morph_type):
    dst = np.zeros_like(img)
    if layout[0] == "F":
        for f in range(0, img.shape[0]):
            cv2_morph(
                dst[f, :, :, :], img[f, :, :, :], ksize, anchor, layout, border_mode, morph_type
            )
    else:
        cv2_morph(dst, img, ksize, anchor, layout, border_mode, morph_type)
    return dst


@dali.pipeline_def(
    num_threads=NUM_THREADS, device_id=DEV_ID, exec_pipelined=False, exec_async=False
)
def reference_pipe(data_src, layout, ksize_src, anchor_src, border_mode, morph_type):
    img = fn.external_source(source=data_src, batch=True, layout=layout)
    ksize = fn.external_source(source=ksize_src)
    anchor = fn.external_source(source=anchor_src)
    return fn.python_function(
        img,
        ksize,
        anchor,
        output_layouts=layout,
        function=lambda im, ks, anch: ref_func(
            im, ks, anch, layout=layout, border_mode=border_mode, morph_type=morph_type
        ),
        batch_processing=False,
    )


@dali.pipeline_def(num_threads=NUM_THREADS, device_id=DEV_ID)
def morphology_pipe(data_src, layout, ksize_src, anchor_src, border_mode, morph_type):
    img = fn.external_source(source=data_src, batch=True, layout=layout, device="gpu")
    ksize = fn.external_source(source=ksize_src)
    anchor = fn.external_source(source=anchor_src)
    if morph_type == "dilate":
        return fn.experimental.dilate(img, mask_size=ksize, anchor=anchor, border_mode=border_mode)
    else:
        return fn.experimental.erode(img, mask_size=ksize, anchor=anchor, border_mode=border_mode)


def ksize_src(bs, lo, hi, seed):
    np_rng = np.random.default_rng(seed=seed)

    def gen_ksize():
        return np_rng.integers(lo, hi, size=(2), dtype=np.int32)

    while True:
        ksize = [gen_ksize() for _ in range(bs)]
        yield ksize


def anchor_src(bs, seed):
    np_rng = np.random.default_rng(seed=seed)

    def gen_anchor():
        use_default = np_rng.choice([True, False])
        if use_default:
            return np.array([-1, -1], dtype=np.int32)
        else:
            return np_rng.integers(0, 3, size=(2), dtype=np.int32)

    while True:
        anchor = [gen_anchor() for _ in range(bs)]
        yield anchor


@params(
    ("dilate", 32, "HWC", np.uint8, 3, 9, "constant"),
    ("dilate", 32, "CHW", np.float32, 1, 4, "constant"),
    ("dilate", 32, "HWC", np.uint16, 1, 5, "reflect"),
    ("dilate", 4, "FHWC", np.float32, 3, 5, "reflect_101"),
    ("dilate", 4, "FCHW", np.uint8, 4, 9, "replicate"),
    ("erode", 32, "HWC", np.uint8, 3, 9, "constant"),
    ("erode", 32, "CHW", np.float32, 1, 4, "constant"),
    ("erode", 32, "HWC", np.uint16, 1, 5, "reflect"),
    ("erode", 4, "FHWC", np.float32, 3, 5, "reflect_101"),
    ("erode", 4, "FCHW", np.uint8, 4, 9, "replicate"),
)
def test_dilate_vs_ocv(morph_type, bs, layout, dtype, channels, max_ksize, border_mode):
    cdim = layout.find("C")
    min_shape = [64 for c in layout]
    min_shape[cdim] = channels
    max_shape = [128 for c in layout]
    max_shape[cdim] = channels
    if layout[0] == "F":
        min_shape[0] = 8
        max_shape[0] = 32

    val_range = (0, 1.0) if dtype == np.float32 else None
    data1 = test_utils.RandomlyShapedDataIterator(
        batch_size=bs,
        min_shape=min_shape,
        max_shape=max_shape,
        dtype=dtype,
        seed=SEED,
        val_range=val_range,
    )
    data2 = test_utils.RandomlyShapedDataIterator(
        batch_size=bs,
        min_shape=min_shape,
        max_shape=max_shape,
        dtype=dtype,
        seed=SEED,
        val_range=val_range,
    )

    ksize1 = ksize_src(bs, 3, max_ksize, SEED)
    ksize2 = ksize_src(bs, 3, max_ksize, SEED)
    anchor1 = anchor_src(bs, SEED)
    anchor2 = anchor_src(bs, SEED)

    pipe1 = morphology_pipe(
        data_src=data1,
        layout=layout,
        ksize_src=ksize1,
        anchor_src=anchor1,
        border_mode=border_mode,
        morph_type=morph_type,
        batch_size=bs,
        prefetch_queue_depth=1,
    )
    pipe2 = reference_pipe(
        data_src=data2,
        layout=layout,
        ksize_src=ksize2,
        anchor_src=anchor2,
        border_mode=border_mode,
        morph_type=morph_type,
        batch_size=bs,
    )
    test_utils.compare_pipelines(pipe1, pipe2, batch_size=bs, N_iterations=1)
