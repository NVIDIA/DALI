# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the experimental ``multi_crop`` Python helper.

Covers https://github.com/NVIDIA/DALI/issues/4735.
"""

import numpy as np

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.experimental.multi_crop import multi_crop
from nose_utils import assert_raises


def _make_image(h=32, w=32, c=3):
    img = np.arange(h * w * c, dtype=np.uint8).reshape(h, w, c)
    return img


def _multi_crop_pipe(image, anchors, crop_shape, stack=True):
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=None)
    with pipe:
        src = fn.external_source(source=lambda: [image], batch=False, layout="HWC")
        out = multi_crop(src, anchors=anchors, crop=crop_shape, axes=(0, 1), stack=stack)
        if stack:
            pipe.set_outputs(out)
        else:
            pipe.set_outputs(*out)
    pipe.build()
    return pipe


def test_multi_crop_stacked_shape():
    image = _make_image(32, 32, 3)
    anchors = [(0, 0), (0, 8), (8, 0), (4, 4)]
    pipe = _multi_crop_pipe(image, anchors, crop_shape=(16, 16), stack=True)
    (out,) = pipe.run()
    arr = out.at(0)
    assert arr.shape == (4, 16, 16, 3), arr.shape


def test_multi_crop_list_outputs():
    image = _make_image(20, 24, 3)
    anchors = [(0, 0), (4, 4), (2, 8)]
    pipe = _multi_crop_pipe(image, anchors, crop_shape=(10, 10), stack=False)
    outs = pipe.run()
    assert len(outs) == 3
    for batch in outs:
        assert batch.at(0).shape == (10, 10, 3)


def test_multi_crop_pixels_match_numpy():
    image = _make_image(16, 16, 3)
    anchors = [(0, 0), (4, 4), (0, 8)]
    crop = (8, 8)
    pipe = _multi_crop_pipe(image, anchors, crop_shape=crop, stack=True)
    (out,) = pipe.run()
    arr = out.at(0)
    for i, (y, x) in enumerate(anchors):
        expected = image[y : y + crop[0], x : x + crop[1], :]
        np.testing.assert_array_equal(arr[i], expected)


def test_multi_crop_relative_anchors():
    image = _make_image(20, 20, 3)
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=None)
    with pipe:
        src = fn.external_source(source=lambda: [image], batch=False, layout="HWC")
        out = multi_crop(
            src,
            rel_anchors=[(0.0, 0.0), (0.5, 0.5)],
            rel_crop=(0.5, 0.5),
            axes=(0, 1),
        )
        pipe.set_outputs(out)
    pipe.build()
    (out,) = pipe.run()
    arr = out.at(0)
    assert arr.shape == (2, 10, 10, 3), arr.shape
    np.testing.assert_array_equal(arr[0], image[0:10, 0:10, :])
    np.testing.assert_array_equal(arr[1], image[10:20, 10:20, :])


def test_multi_crop_validation_no_anchors():
    with assert_raises(ValueError, glob="exactly one of"):
        multi_crop(None, anchors=None, rel_anchors=None)


def test_multi_crop_validation_both_modes():
    with assert_raises(ValueError, glob="exactly one of"):
        multi_crop(None, anchors=[(0, 0)], rel_anchors=[(0.0, 0.0)])


def test_multi_crop_validation_missing_crop():
    with assert_raises(ValueError, glob="`crop` must be provided*"):
        multi_crop(None, anchors=[(0, 0)])


def test_multi_crop_validation_empty():
    with assert_raises(ValueError, glob="*at least one anchor*"):
        multi_crop(None, anchors=[], crop=(8, 8))
