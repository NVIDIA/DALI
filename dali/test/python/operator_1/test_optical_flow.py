# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import numpy as np
import random
import cv2
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali import fn, types
from test_utils import get_dali_extra_path, get_arch, is_of_supported
from nose_utils import raises, assert_raises, SkipTest

test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, "db", "imgproc")


def get_mapping(shape):
    h, w = shape
    x = np.arange(w, dtype=np.float32) + 0.5
    y = np.arange(h, dtype=np.float32) + 0.5
    xy = np.transpose([np.tile(x, h), np.repeat(y, w)]).reshape([h, w, 2])
    center = np.array([[[w * 0.5, h * 0.5]]])
    d = xy - center
    dnorm = np.linalg.norm(d, ord=2, axis=2)
    dexp1 = dnorm * (7 / np.sqrt(w * w + h * h))
    dexp2 = dnorm * (9 / np.sqrt(w * w + h * h))
    mag = np.exp(-(dexp1**2)) - np.exp(-(dexp2**2))
    od = d + 0
    od[:, :, 0] = d[:, :, 0] * (1 - mag) + d[:, :, 1] * mag
    od[:, :, 1] = d[:, :, 1] * (1 - mag) + d[:, :, 0] * mag

    ofs = od - d

    return xy, ofs


def load_frames(sample_info=types.SampleInfo(0, 0, 0, 0), hint_grid=None):
    img = cv2.imread(os.path.join(images_dir, "alley.png"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if sample_info.idx_in_epoch % 2:
        img = cv2.resize(
            img, dsize=(img.shape[0] // 2, img.shape[1] // 2), interpolation=cv2.INTER_AREA
        )

    xy, ofs = get_mapping(img.shape[:2])
    remap = (xy + ofs - np.array([[[0.5, 0.5]]])).astype(np.float32)

    warped = cv2.remap(img, remap, None, interpolation=cv2.INTER_LINEAR)
    result = np.array([img, warped])

    if hint_grid is not None:
        result = [result]
        result.append(np.zeros(shape=result[0].shape, dtype=np.uint8))
    return result


@pipeline_def(batch_size=1, seed=16)
def of_pipeline(output_grid=1, hint_grid=1, use_temporal_hints=False):
    if hint_grid is not None:
        seq, hint = fn.external_source(
            lambda info: load_frames(info, hint_grid),
            layout=["FHWC", "FHWC"],
            batch=False,
            num_outputs=2,
        )

        of = fn.optical_flow(
            seq.gpu(),
            hint.gpu(),
            device="gpu",
            output_grid=output_grid,
            hint_grid=hint_grid,
            enable_temporal_hints=use_temporal_hints,
        )
    else:
        seq = fn.external_source(
            lambda info: load_frames(info, hint_grid), layout="FHWC", batch=False
        )
        of = fn.optical_flow(
            seq.gpu(),
            device="gpu",
            output_grid=output_grid,
            enable_temporal_hints=use_temporal_hints,
        )
    return seq, of


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    """

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    """

    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


interactive = False


def check_optflow(output_grid=1, hint_grid=1, use_temporal_hints=False):
    if not is_of_supported():
        raise SkipTest("Optical Flow is not supported on this platform")
    batch_size = 3
    pipe = of_pipeline(
        batch_size=batch_size,
        num_threads=3,
        device_id=0,
        output_grid=output_grid,
        hint_grid=hint_grid,
        use_temporal_hints=use_temporal_hints,
    )
    if get_arch() < 8:
        if output_grid != 4:
            assert_raises(
                RuntimeError, pipe.run, glob="grid size: * is not supported, supported are:"
            )
            raise SkipTest("Skipped as grid size is not supported for this arch")
        elif hint_grid not in [4, 8, None]:
            assert_raises(
                RuntimeError, pipe.run, glob="hint grid size: * is not supported, supported are:"
            )
            raise SkipTest("Skipped as hint grid size is not supported for this arch")

    for _ in range(2):
        out0, out1 = tuple(out.as_cpu() for out in pipe.run())
        for i in range(batch_size):
            seq = out0.at(i)
            out_field = out1.at(i)[0]
            _, ref_field = get_mapping(seq.shape[1:3])
            dsize = (out_field.shape[1], out_field.shape[0])
            ref_field = cv2.resize(ref_field, dsize=dsize, interpolation=cv2.INTER_AREA)
            if interactive:
                cv2.imshow("out", flow_to_color(out_field, None, True))
                cv2.imshow("ref", flow_to_color(ref_field, None, True))
                print(np.max(out_field))
                print(np.max(ref_field))
                cv2.imshow("dif", flow_to_color(ref_field - out_field, None, True))
                cv2.waitKey(0)
            err = np.linalg.norm(ref_field - out_field, ord=2, axis=2)
            assert np.mean(err) < 1  # average error of less than one pixel
            assert np.max(err) < 100  # no point more than 100px off
            assert np.sum(err > 1) / np.prod(err.shape) < 0.1  # 90% are within 1px
            assert np.sum(err > 2) / np.prod(err.shape) < 0.05  # 95% are within 2px


def test_optflow():
    for output_grid in [1, 2, 4]:
        hint_grid = random.choice([None, 1, 2, 4, 8])
        for use_temporal_hints in [True, False]:
            yield check_optflow, output_grid, hint_grid, use_temporal_hints


@raises(RuntimeError, "Output grid size: 3 is not supported, supported are:")
def test_wrong_out_grid_size():
    if not is_of_supported():
        raise SkipTest("Optical Flow is not supported on this platform")
    pipe = of_pipeline(num_threads=3, device_id=0, output_grid=3)
    pipe.run()


@raises(RuntimeError, "Hint grid size: 3 is not supported, supported are:")
def test_wrong_hint_grid_size():
    if not is_of_supported():
        raise SkipTest("Optical Flow is not supported on this platform")
    pipe = of_pipeline(num_threads=3, device_id=0, output_grid=4, hint_grid=3)
    pipe.run()
