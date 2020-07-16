# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import shutil
import sys
import cv2
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from random import shuffle
from test_utils import get_dali_extra_path

test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, 'db', 'imgproc')

def get_mapping(shape):
    h, w = shape
    x = np.arange(w, dtype=np.float32) + 0.5
    y = np.arange(h, dtype=np.float32) + 0.5
    xy = np.transpose([np.tile(x, h), np.repeat(y, w)]).reshape([h, w, 2])
    center = np.array([[[w*0.5, h*0.5]]])
    d = xy - center
    dnorm = np.linalg.norm(d, ord=2, axis=2)
    dexp1 = dnorm*(7/np.sqrt(w*w+h*h))
    dexp2 = dnorm*(9/np.sqrt(w*w+h*h))
    mag = np.exp(-dexp1 ** 2) - np.exp(-dexp2 ** 2)
    od = d + 0
    od[:,:,0] = d[:,:,0] * (1-mag) + d[:,:,1] * mag
    od[:,:,1] = d[:,:,1] * (1-mag) + d[:,:,0] * mag

    ofs = od - d

    return xy, ofs

def load_frames():
    img = cv2.imread(os.path.join(images_dir, 'alley.png'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    xy, ofs = get_mapping(img.shape[:2])
    remap = (xy + ofs - np.array([[[0.5,0.5]]])).astype(np.float32)

    warped = cv2.remap(img, remap, None, interpolation = cv2.INTER_LINEAR)
    result = np.array([img, warped])

    return [result]


class OFPipeline(Pipeline):
    def __init__(self, num_threads, device_id):
        super(OFPipeline, self).__init__(1, num_threads, device_id, seed=16)

        self.input = ops.ExternalSource()
        self.of_op = ops.OpticalFlow(device="gpu", output_format=4)

    def define_graph(self):
        seq = self.data = self.input(name="input")
        of = self.of_op(seq.gpu())
        return [seq, of]

    def iter_setup(self):
        self.feed_input(self.data, load_frames(), layout="DHWC")

def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

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
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

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

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

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

def test_optflow():
    pipe = OFPipeline(3, 0);
    pipe.build()
    out = pipe.run()
    seq = out[0].at(0)
    out_field = out[1].as_cpu().at(0)[0];
    _, ref_field = get_mapping(seq.shape[1:3])
    dsize = (out_field.shape[1], out_field.shape[0])
    ref_field = cv2.resize(ref_field, dsize = dsize, interpolation = cv2.INTER_AREA)
    if interactive:
        cv2.imshow("out", flow_to_color(out_field, None, True))
        cv2.imshow("ref", flow_to_color(ref_field, None, True))
        print(np.max(out_field))
        print(np.max(ref_field))
        cv2.imshow("dif", flow_to_color(ref_field - out_field, None, True))
        cv2.waitKey(0)
    err = np.linalg.norm(ref_field-out_field, ord=2, axis=2)
    assert(np.mean(err) < 1)   # average error of less than one pixel
    assert(np.max(err) < 100)  # no point more than 100px off
    assert(np.sum(err > 1) / np.prod(err.shape) < 0.1)  # 90% are within 1px
    assert(np.sum(err > 2) / np.prod(err.shape) < 0.05)  # 95% are within 2px

def main():
    test_optflow()

if __name__ == '__main__':
    interactive = True
    main()
