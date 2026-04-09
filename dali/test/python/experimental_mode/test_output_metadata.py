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

import nvidia.dali.experimental.dynamic as ndd
from ndd_utils import eval_modes
from collections.abc import Iterable
from nose2.tools import params


def assert_correct_metadata(
    *tensors: ndd.Tensor,
    attributes: Iterable[str] = ("dtype", "ndim", "layout"),
):
    # First collect all metadata to avoid triggering transitive evaluations
    actual_metadata = []
    for tensor in tensors:
        assert tensor._invocation_result._invocation._results is None
        actual = {attr: getattr(tensor, attr) for attr in attributes}
        assert tensor._invocation_result._invocation._results is None
        actual_metadata.append(actual)

    for tensor, actual in zip(tensors, actual_metadata):
        tensor.evaluate()
        assert tensor._invocation_result._invocation._results is not None
        expected = {attr: getattr(tensor, attr) for attr in attributes}

        for attr in attributes:
            assert (
                actual[attr] == expected[attr]
            ), f"Expected {attr} to be {expected[attr]}, but got {actual[attr]}"


# --- Image operators ---


@eval_modes(ndd.EvalMode.deferred)
@params(
    (ndd.flip, dict(horizontal=1)),
    (ndd.crop, dict(crop=(2, 2))),
    (ndd.crop, dict(crop=(2, 2), dtype=ndd.float32)),
    (ndd.cast, dict(dtype=ndd.float32)),
    (ndd.resize, dict(resize_x=2, resize_y=2)),
    (ndd.hsv, dict(dtype=ndd.float32, hue=10.0)),
    (ndd.hsv, dict(hue=10.0)),
    (ndd.brightness_contrast, dict(brightness=1.2)),
    (ndd.crop_mirror_normalize, dict(crop=(2, 2))),
    (ndd.crop_mirror_normalize, dict(crop=(2, 2), dtype=ndd.float32)),
)
def test_image_ops(func, kwargs):
    tensor = ndd.zeros(shape=(4, 4, 3), dtype=ndd.uint8, layout="HWC")
    assert_correct_metadata(func(tensor, **kwargs))


# --- Random generators ---


@eval_modes(ndd.EvalMode.deferred)
@params(
    ndd.random.coin_flip,
    ndd.random.uniform,
)
def test_random_generators(func):
    assert_correct_metadata(func())


@eval_modes(ndd.EvalMode.deferred)
@params(
    (ndd.random.coin_flip, dict(shape=[4])),
    (ndd.random.uniform, dict(shape=[4])),
)
def test_random_generators_with_shape(func, kwargs):
    # ndim is excluded because the `shape` kwarg determines ndim at run time,
    # and the schema cannot infer it without executing the operator.
    assert_correct_metadata(func(**kwargs), attributes=("dtype", "layout"))


# --- Other operators ---


@eval_modes(ndd.EvalMode.deferred)
def test_bb_flip():
    bbox = ndd.zeros(shape=(1, 4), dtype=ndd.float32)
    assert_correct_metadata(ndd.bb_flip(bbox))


@eval_modes(ndd.EvalMode.deferred)
def test_random_bbox_crop():
    bboxes = ndd.zeros(shape=(1, 4), dtype=ndd.float32)
    anchor, shape, out_bboxes = ndd.random_bbox_crop(bboxes)
    assert_correct_metadata(anchor, shape, out_bboxes)


# --- Slicing ---


@eval_modes(ndd.EvalMode.deferred)
def test_slice():
    input = ndd.as_batch(
        [
            ndd.zeros(shape=(2, 3, 4), dtype=ndd.int8),
            ndd.zeros(shape=(3, 2, 5), dtype=ndd.int8),
        ],
        layout="XYZ",
    ).evaluate()
    assert_correct_metadata(input.slice[1])
    assert_correct_metadata(input.slice[:, 1, :])
    assert_correct_metadata(input.slice[0, 1, 2])
    assert_correct_metadata(input.slice[1:, 1:, 1:])


# --- New dimensions ---


@eval_modes(ndd.EvalMode.deferred)
def test_stack():
    input1 = ndd.as_batch([[[1], [2]], [[10], [20]]], layout="AB")
    input2 = ndd.as_batch([[[3], [4]], [[30], [40]]], layout="AB")

    out0 = ndd.stack(input1, input2, axis=0, axis_name="C")
    assert out0.layout == "CAB"
    assert_correct_metadata(out0)

    out1 = ndd.stack(input1, input2, axis=1, axis_name="C")
    assert out1.layout == "ACB"
    assert_correct_metadata(out1)

    out2 = ndd.stack(input1, input2, axis=2, axis_name="C")
    assert out2.layout == "ABC"
    assert_correct_metadata(out2)


@eval_modes(ndd.EvalMode.deferred)
def test_expand_dims():
    inp = ndd.as_batch([[1, 2], [10, 20]], layout="A")

    out0 = ndd.expand_dims(inp, axes=[0, 1], new_axis_names="BC")
    assert out0.layout == "BCA"
    assert_correct_metadata(out0)

    out1 = ndd.expand_dims(inp, axes=[0, 2], new_axis_names="BC")
    assert out1.layout == "BAC"
    assert_correct_metadata(out1)

    out2 = ndd.expand_dims(inp, axes=[1, 2], new_axis_names="BC")
    assert out2.layout == "ABC"
    assert_correct_metadata(out2)

    inp = ndd.as_batch([1, 2])
    out3 = ndd.expand_dims(inp, axes=[0, 1], new_axis_names="AB")
    assert out3.layout == "AB"
    assert_correct_metadata(out3)

    inp = ndd.as_batch([1, 2])
    out4 = ndd.expand_dims(inp, axes=[0, 1])
    assert out4.layout is None
    assert_correct_metadata(out4)


# --- Sequence ---


@eval_modes(ndd.EvalMode.deferred)
def test_per_frame_warp():
    input = ndd.as_batch(
        [
            ndd.zeros(shape=(300, 400, 3), dtype=ndd.uint8),
            ndd.zeros(shape=(200, 500, 3), dtype=ndd.uint8),
        ],
        layout="HWC",
    ).evaluate()

    matrices = ndd.per_frame(
        ndd.tensor(
            [
                [[1, 0, 0], [0, 1, 0]],
                [[0.9, 0.1, 0], [-0.1, 0.9, 0]],
            ]
        )
    )
    assert_correct_metadata(ndd.warp_affine(input, matrices))


@eval_modes(ndd.EvalMode.deferred)
def test_gaussian_blur_fchw():
    input = ndd.as_batch(
        [
            ndd.zeros(shape=(4, 3, 300, 400), dtype=ndd.uint8),
            ndd.zeros(shape=(4, 3, 200, 500), dtype=ndd.uint8),
        ],
        layout="FCHW",
    ).evaluate()

    sigma = ndd.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=ndd.float32, layout="F*")
    sigmas = ndd.as_batch([sigma] * 2).evaluate()
    sigmas = ndd.per_frame(sigmas)

    assert_correct_metadata(ndd.gaussian_blur(input, sigma=sigmas))
