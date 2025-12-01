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

import numpy as np

import nvidia.dali.tensors as _tensors
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.auto_aug.core import augmentation, signed_bin
from nvidia.dali.auto_aug.core._args import forbid_unused_kwargs
from nose2.tools import params

from test_utils import check_batch
from nose_utils import assert_warns, assert_raises


def sample_info(cb):
    def idx_in_batch_cb(sample_info):
        return np.array(cb(sample_info), dtype=np.int32)

    return fn.external_source(idx_in_batch_cb, batch=False)


def ref_param(
    mag_range, mag_range_num_elements, bins_batch, mag_signs_batch=None, mag_to_param=None
):
    if isinstance(mag_range, tuple):
        assert len(mag_range) == 2
        lo, hi = mag_range
        mag_range = np.linspace(lo, hi, mag_range_num_elements)
    magnitudes = [mag_range[mag_bin] for mag_bin in bins_batch]
    if mag_signs_batch is not None:
        assert len(mag_signs_batch) == len(magnitudes)
        magnitudes = [mag * ((-1) ** negate) for mag, negate in zip(magnitudes, mag_signs_batch)]
    mag_to_param = mag_to_param if mag_to_param is not None else np.array
    return np.array([mag_to_param(mag) for mag in magnitudes])


def test_magnitude_is_none():
    @augmentation
    def pass_through_sample(data, param):
        assert param is None, "If the `mag_range` is not specified the param should be None"
        return data

    data = types.Constant(42)
    assert pass_through_sample(data, magnitude_bin=42) is data


def test_lo_hi_mag_range():
    mag_range = (100, 110)
    batch_size = 11
    const_bin = 2

    @augmentation(mag_range=mag_range)
    def pass_through_mag(data, param):
        return param

    @pipeline_def(num_threads=4, device_id=0, batch_size=batch_size, seed=42)
    def pipeline():
        idx_in_batch = sample_info(lambda info: info.idx_in_batch)
        const_mag = pass_through_mag(
            types.Constant(42), magnitude_bin=const_bin, num_magnitude_bins=5
        )
        dyn_mag = pass_through_mag(
            types.Constant(42), magnitude_bin=idx_in_batch, num_magnitude_bins=11
        )
        return const_mag, dyn_mag

    p = pipeline()
    const_mag, dyn_mag = p.run()
    const_mag_ref = ref_param(mag_range, 5, [const_bin] * batch_size)
    dyn_mag_ref = ref_param(mag_range, 11, list(range(batch_size)))
    check_batch(const_mag, const_mag_ref, max_allowed_error=0)
    check_batch(dyn_mag, dyn_mag_ref, max_allowed_error=0)


def test_explicit_mag_range():
    mag_range = np.array([1, 1, 2, 3, 5, 8, 13, 21])
    batch_size = 8
    const_bin = 7

    @augmentation(mag_range=mag_range)
    def pass_through_mag(data, param):
        return param

    @pipeline_def(num_threads=4, device_id=0, batch_size=batch_size, seed=42)
    def pipeline():
        idx_in_batch = sample_info(lambda info: info.idx_in_batch)
        const_mag = pass_through_mag(types.Constant(42), magnitude_bin=const_bin)
        dyn_mag = pass_through_mag(types.Constant(42), magnitude_bin=idx_in_batch)
        return const_mag, dyn_mag

    p = pipeline()
    const_mag, dyn_mag = p.run()
    const_mag_ref = ref_param(mag_range, None, [const_bin] * batch_size)
    dyn_mag_ref = ref_param(mag_range, None, list(range(batch_size)))
    check_batch(const_mag, const_mag_ref, max_allowed_error=0)
    check_batch(dyn_mag, dyn_mag_ref, max_allowed_error=0)


@params(
    (((201, 260), 60, False, 0)),
    (((301, 330), 30, True, 29)),
    (((101, 150), 50, False, None)),
    (((701, 710), 10, True, None)),
)
def test_randomly_negate(mag_range, num_magnitude_bins, use_implicit_sign, const_mag):
    batch_size = 64

    @augmentation(mag_range=mag_range, randomly_negate=True)
    def pass_through_mag(data, param):
        return param

    @pipeline_def(num_threads=4, device_id=0, batch_size=batch_size, seed=42)
    def pipeline():
        magnitude_bin = (
            const_mag
            if const_mag is not None
            else sample_info(lambda info: info.idx_in_batch % num_magnitude_bins)
        )
        if not use_implicit_sign:
            magnitude_sign = sample_info(lambda info: info.idx_in_batch % 2)
            magnitude_bin = signed_bin(magnitude_bin, magnitude_sign)

        return pass_through_mag(
            types.Constant(42), magnitude_bin=magnitude_bin, num_magnitude_bins=num_magnitude_bins
        )

    if not use_implicit_sign:
        p = pipeline()
    else:
        warn_glob = "but unsigned `magnitude_bin` was passed to the augmentation call"
        with assert_warns(Warning, glob=warn_glob):
            p = pipeline()
    (magnitudes,) = p.run()
    magnitudes = [np.array(el) for el in magnitudes]
    if use_implicit_sign:
        # the implicit sign is random, just sanity check if
        # there are some positive and negative magnitudes
        assert any(el < 0 for el in magnitudes)
        assert any(el > 0 for el in magnitudes)
        magnitudes = [np.abs(el) for el in magnitudes]

    mag_sign = None if use_implicit_sign else [i % 2 for i in range(batch_size)]
    magnitude_bin = (
        [const_mag] * batch_size
        if const_mag is not None
        else [i % num_magnitude_bins for i in range(batch_size)]
    )
    ref_magnitudes = ref_param(
        mag_range, num_magnitude_bins, magnitude_bin, mag_signs_batch=mag_sign
    )
    check_batch(magnitudes, ref_magnitudes, max_allowed_error=0)


@params((4,), (None,))
def test_no_randomly_negate(const_mag):
    mag_range = (0, 10)
    num_magnitude_bins = 11
    batch_size = 32

    @augmentation(mag_range=mag_range)
    def pass_through_mag(data, param):
        return param

    @pipeline_def(num_threads=4, device_id=0, batch_size=batch_size, seed=42)
    def pipeline():
        magnitude_bin = (
            const_mag
            if const_mag is not None
            else sample_info(lambda info: info.idx_in_batch % num_magnitude_bins)
        )

        # make sure that the augmentation declared without `randomly_negate` ignores the signed_bin
        return pass_through_mag(
            types.Constant(42),
            magnitude_bin=signed_bin(magnitude_bin),
            num_magnitude_bins=num_magnitude_bins,
        )

    p = pipeline()
    (magnitudes,) = p.run()
    magnitude_bin = (
        [const_mag] * batch_size
        if const_mag is not None
        else [i % num_magnitude_bins for i in range(batch_size)]
    )
    ref_magnitudes = ref_param(mag_range, 11, magnitude_bin)
    check_batch(magnitudes, ref_magnitudes, max_allowed_error=0)


@params((((201, 211), 11, 7, np.uint16, "cpu")), (((101, 107), 7, None, np.float32, "gpu")))
def test_mag_to_param(mag_range, num_magnitude_bins, const_mag, dtype, param_device):
    batch_size = 31

    def mag_to_param(magnitude):
        return np.array([magnitude, magnitude + 2, 42], dtype=dtype)

    @augmentation(
        mag_range=mag_range,
        randomly_negate=True,
        mag_to_param=mag_to_param,
        param_device=param_device,
    )
    def pass_through_mag(data, param):
        return param

    @pipeline_def(num_threads=4, device_id=0, batch_size=batch_size, seed=42)
    def pipeline():
        mag_sign = sample_info(lambda info: info.idx_in_batch % 2)
        magnitude_bin = (
            const_mag
            if const_mag is not None
            else sample_info(lambda info: info.idx_in_batch % num_magnitude_bins)
        )

        return pass_through_mag(
            types.Constant(42),
            magnitude_bin=signed_bin(magnitude_bin, mag_sign),
            num_magnitude_bins=num_magnitude_bins,
        )

    p = pipeline()
    (magnitudes,) = p.run()
    if param_device == "cpu":
        assert isinstance(magnitudes, _tensors.TensorListCPU)
    else:
        assert isinstance(magnitudes, _tensors.TensorListGPU)
        magnitudes = magnitudes.as_cpu()
    magnitudes = [np.array(el) for el in magnitudes]

    mag_sign = [i % 2 for i in range(batch_size)]
    magnitude_bin = (
        [const_mag] * batch_size
        if const_mag is not None
        else [i % num_magnitude_bins for i in range(batch_size)]
    )
    ref_magnitudes = ref_param(
        mag_range,
        num_magnitude_bins,
        magnitude_bin,
        mag_signs_batch=mag_sign,
        mag_to_param=mag_to_param,
    )
    assert np.array(magnitudes).dtype == np.array(ref_magnitudes).dtype
    check_batch(magnitudes, ref_magnitudes, max_allowed_error=0)


def test_augmentation_setup_update():
    def dummy_mag_to_param(magnitude):
        return magnitude + 1

    initial = {
        "mag_range": (0, 10),
        "randomly_negate": True,
        "mag_to_param": dummy_mag_to_param,
        "param_device": "gpu",
        "name": "some_other_dummy_name",
    }

    @augmentation
    def default_aug(data, _):
        return data

    defaults = {attr: getattr(default_aug, attr) for attr in initial}
    defaults["name"] = "dummy"

    @augmentation(**initial)
    def dummy(data, _):
        return data

    for reset_attr in initial:
        reset_attr_aug = dummy.augmentation(**{reset_attr: None})
        for attr in initial:
            reset = getattr(reset_attr_aug, attr)
            ref = (defaults if attr == reset_attr else initial)[attr]
            assert reset == ref, f"{attr}: {reset}, {ref} ({reset_attr})"


def test_augmentation_nested_decorator_fail():
    @augmentation
    def dummy(data, _):
        return data

    with assert_raises(
        Exception, glob="The `@augmentation` was applied to already decorated Augmentation."
    ):
        augmentation(dummy, mag_range=(5, 10))


def test_mag_to_param_data_node_fail():
    def shear(magnitude):
        return fn.transforms.shear(shear=magnitude)

    @augmentation(mag_range=(0, 250), mag_to_param=shear)
    def illegal_shear(data, shear_mt):
        return fn.warp_affine(data, mt=shear_mt)

    @pipeline_def(num_threads=4, device_id=0, batch_size=8, seed=42)
    def pipeline():
        data = types.Constant(np.full((100, 100, 3), 42, dtype=np.uint8))
        return illegal_shear(data, magnitude_bin=5, num_magnitude_bins=10)

    glob_msg = "callback must return parameter that is `np.ndarray` or"
    with assert_raises(Exception, glob=glob_msg):
        pipeline()


@params((True, False), (False, True))
def test_mag_to_param_non_uniform_fail(non_uniform_shape, non_uniform_type):
    shape_lo = (2,)
    shape_hi = (3,)

    def mag_to_param(magnitude):
        shape = shape_lo if not non_uniform_shape or magnitude < 5 else shape_hi
        dtype = np.uint8 if not non_uniform_type or magnitude < 3 else np.uint16
        return np.full(shape, 42, dtype=dtype)

    @augmentation(mag_range=(0, 10), mag_to_param=mag_to_param)
    def pass_param(data, param):
        return param

    @pipeline_def(num_threads=4, device_id=0, batch_size=8, seed=42)
    def pipeline():
        data = types.Constant(np.full((100, 100, 3), 42, dtype=np.uint8))
        mag_bin = sample_info(lambda si: si.idx_in_batch)
        return pass_param(data, magnitude_bin=mag_bin, num_magnitude_bins=11)

    glob_msg = (
        f"augmentation must return the arrays of the same type and shape *"
        f"has shape {shape_hi if non_uniform_shape else shape_lo} and type "
        f"{'uint16' if non_uniform_type else 'uint8'}."
    )
    with assert_raises(Exception, glob=glob_msg):
        pipeline()


def test_lack_of_positional_args_fail():
    def no_args():
        pass

    def one_arg(arg):
        pass

    def one_kwarg_only(arg, *, kwarg_only):
        pass

    def kwarg_only(*, kwarg1, kwarg2):
        pass

    for i, fun in enumerate((no_args, kwarg_only, one_arg, one_kwarg_only)):
        msg = f"accepts {i // 2} positional argument(s), but the functions decorated"
        with assert_raises(Exception, glob=msg):
            augmentation(fun)


def test_no_required_kwargs():
    @augmentation
    def aug(data, param, extra, another_extra, extra_with_default=None):
        pass

    @pipeline_def(batch_size=3, num_threads=4, device_id=0, seed=42)
    def pipeline(aug, aug_kwargs):
        return aug(types.Constant(42), **aug_kwargs)

    pipeline(aug, {"extra": None, "another_extra": 42, "extra_with_default": 7})
    pipeline(aug, {"extra": None, "another_extra": 42})

    with assert_raises(Exception, glob="not provided to the call: another_extra"):
        pipeline(aug, {"extra": None})

    with assert_raises(Exception, glob="not provided to the call: extra"):
        pipeline(aug, {"another_extra": 42})

    with assert_raises(Exception, glob="not provided to the call: extra, another_extra"):
        pipeline(aug, {})


def test_unused_kwargs():
    @augmentation
    def no_extra(data, _):
        pass

    @augmentation
    def aug(data, param, one_param, another_param):
        pass

    @augmentation
    def another_aug(data, _, another_param, yet_another_param):
        pass

    augments = (no_extra, aug, another_aug)

    forbid_unused_kwargs(augments, {}, "dummy")
    forbid_unused_kwargs(
        augments, {"one_param": 1, "another_param": 2, "yet_another_param": 3}, "dummy"
    )

    with assert_raises(Exception, glob="The kwarg `amnother_param` is not used"):
        forbid_unused_kwargs(
            augments, {"one_param": 1, "amnother_param": 2, "yet_another_param": 3}, "dummy"
        )

    with assert_raises(Exception, glob="The kwargs `amnother_param, yemt_another_param` are"):
        forbid_unused_kwargs(
            augments, {"one_param": 1, "amnother_param": 2, "yemt_another_param": 3}, "dummy"
        )
