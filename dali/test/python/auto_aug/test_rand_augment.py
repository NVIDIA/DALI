# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools
import os

import numpy as np
from scipy.stats import chisquare
from nose2.tools import params

from nvidia.dali import fn, types
from nvidia.dali import pipeline_def
from nvidia.dali.auto_aug import rand_augment
from nvidia.dali.auto_aug.core import augmentation

from test_utils import get_dali_extra_path
from nose_utils import assert_raises

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')


@params(*tuple(enumerate(itertools.product((True, False), (True, False), (None, 0),
                                           (True, False)))))
def test_run_rand_aug(i, args):
    uniformly_resized, use_shape, fill_value, specify_translation_bounds = args
    batch_sizes = [1, 8, 7, 64, 13, 64, 128]
    ns = [1, 2, 3, 4]
    ms = [0, 15, 30]
    batch_size = batch_sizes[i % len(batch_sizes)]
    n = ns[i % len(ns)]
    m = ms[i % len(ms)]

    @pipeline_def(enable_conditionals=True, batch_size=batch_size, num_threads=4, device_id=0,
                  seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        if uniformly_resized:
            image = fn.resize(image, size=(244, 244))
        extra = {} if not use_shape else {"shape": fn.peek_image_shape(encoded_image)}
        if fill_value is not None:
            extra["fill_value"] = fill_value
        if specify_translation_bounds:
            if use_shape:
                extra["max_translate_rel"] = 0.9
            else:
                extra["max_translate_abs"] = 400
        image = rand_augment.rand_augment(image, n=n, m=m, **extra)
        return image

    p = pipeline()
    p.build()
    for _ in range(3):
        p.run()


@params(*tuple(itertools.product((True, False), (0, 1), ('height', 'width', 'both'))))
def test_translation(use_shape, offset_fraction, extent):
    # make sure the translation helper processes the args properly
    # note, it only uses translate_y (as it is in imagenet policy)
    shape = [300, 400]
    fill_value = 105
    params = {}
    if use_shape:
        param = offset_fraction
        param_name = "max_translate_rel"
    else:
        param_name = "max_translate_abs"
    assert extent in ('height', 'width', 'both'), f"{extent}"
    if extent == 'both':
        param = [shape[0] * offset_fraction, shape[1] * offset_fraction]
    elif extent == 'height':
        param = [shape[0] * offset_fraction, 0]
    elif extent == 'width':
        param = [0, shape[1] * offset_fraction]
    params[param_name] = param
    translate_x, translate_y = rand_augment._get_translations(use_shape=use_shape, **params)
    if extent == 'both':
        augments = [translate_x, translate_y]
    elif extent == 'height':
        augments = [translate_y]
    elif extent == 'width':
        augments = [translate_x]

    @pipeline_def(enable_conditionals=True, batch_size=3, num_threads=4, device_id=0, seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        image = fn.resize(image, size=shape)
        if use_shape:
            return rand_augment.apply_rand_augment(augments, image, n=1, m=30,
                                                   fill_value=fill_value, shape=shape)
        else:
            return rand_augment.apply_rand_augment(augments, image, n=1, m=30,
                                                   fill_value=fill_value)

    p = pipeline()
    p.build()
    output, = p.run()
    output = [np.array(sample) for sample in output.as_cpu()]
    for i, sample in enumerate(output):
        sample = np.array(sample)
        if offset_fraction == 1:
            assert np.all(sample == fill_value), f"sample_idx: {i}"
        else:
            background_count = np.sum(sample == fill_value)
            assert background_count / sample.size < 0.1, \
                f"sample_idx: {i}, {background_count / sample.size}"


@params(*tuple(enumerate(itertools.product(
    ['cpu', 'gpu'],
    [True, False],
    [1, 2, 3],
    [2, 3],
))))
def test_ops_selection_and_mags(case_idx, args):

    dev, use_sign, n, num_ops = args
    num_magnitude_bins = 9
    # the chisquare expects at least 5 elements in a bin and we can have around
    # (num_ops * (2**use_signs)) ** n ops
    batch_size = 2048
    magnitude_cases = list(range(num_magnitude_bins))
    m = magnitude_cases[case_idx % len(magnitude_cases)]

    def mag_to_param_with_op_id(op_id):

        def mag_to_param(magnitude):
            return np.array([op_id, magnitude], dtype=np.int32)

        return mag_to_param

    @augmentation(param_device=dev)
    def op(sample, op_id_mag_id):
        return fn.cat(sample, op_id_mag_id)

    augmentations = [
        op.augmentation(mag_range=(10 * i + 1, 10 * i + num_magnitude_bins),
                        mag_to_param=mag_to_param_with_op_id(i + 1), randomly_negate=use_sign
                        and i % 3 == 0) for i in range(num_ops)
    ]

    expected_counts = {}
    seq_prob = 1. / (num_ops**n)
    for aug_sequence in itertools.product(*([augmentations] * n)):
        possible_signs = [(-1, 1) if aug.randomly_negate else (1, ) for aug in aug_sequence]
        possible_signs = tuple(itertools.product(*possible_signs))
        prob = seq_prob / len(possible_signs)
        for signs in possible_signs:
            assert len(aug_sequence) == len(signs)
            outs = []
            for aug, sign in zip(aug_sequence, signs):
                mag = aug._get_magnitudes(num_magnitude_bins)[m]
                op_id_mag = aug.mag_to_param(mag * sign)
                outs.append(op_id_mag)
            expected_counts[tuple(el for out in outs for el in out)] = prob
    expected_counts = {output: p * batch_size for output, p in expected_counts.items()}

    @pipeline_def(enable_conditionals=True, batch_size=batch_size, num_threads=4, device_id=0,
                  seed=42)
    def pipeline():
        sample = types.Constant([], dtype=types.INT32)
        if dev == "gpu":
            sample = sample.gpu()
        sample = rand_augment.apply_rand_augment(augmentations, sample, n=n, m=m,
                                                 num_magnitude_bins=num_magnitude_bins)
        return fn.reshape(sample, shape=(-1, 2))

    p = pipeline()
    p.build()
    for i in range(3):
        output, = p.run()
        output = [np.array(s) for s in (output.as_cpu() if dev == "gpu" else output)]
        actual_count = {allowed_out: 0 for allowed_out in expected_counts}
        for sample in output:
            assert len(sample) == n, f"{i} {sample}"
            out = tuple(el for op_mag in sample for el in op_mag)
            actual_count[out] += 1
        actual = []
        expected = []
        for out in expected_counts:
            actual.append(actual_count[out])
            expected.append(expected_counts[out])
        stat = chisquare(actual, expected)
        assert 0.01 <= stat.pvalue <= 0.99, f"{stat} {actual} {expected}"


def test_wrong_params_fail():

    @pipeline_def(batch_size=4, device_id=0, num_threads=4, seed=42, enable_conditionals=True)
    def pipeline(n, m, num_magnitude_bins):
        sample = types.Constant(np.array([[[]]], dtype=np.uint8))
        return rand_augment.rand_augment(sample, n=n, m=m, num_magnitude_bins=num_magnitude_bins)

    with assert_raises(Exception,
                       glob="The number of operations to apply `n` must be a non-negative integer"):
        pipeline(n=None, m=1, num_magnitude_bins=11)

    with assert_raises(Exception, glob="The `num_magnitude_bins` must be a positive integer, got"):
        pipeline(n=1, m=1, num_magnitude_bins=None)

    with assert_raises(Exception, glob="`m` must be an integer from `[[]0, 14[]]` range. Got 15."):
        pipeline(n=1, m=15, num_magnitude_bins=15)

    with assert_raises(Exception, glob="The `augmentations` list cannot be empty"):

        @pipeline_def(batch_size=4, device_id=0, num_threads=4, seed=42, enable_conditionals=True)
        def no_aug_pipeline():
            sample = types.Constant(np.array([[[]]], dtype=np.uint8))
            return rand_augment.apply_rand_augment([], sample, 1, 20)

        no_aug_pipeline()

    with assert_raises(Exception, glob="The augmentation `translate_x` requires `shape` argument"):

        @pipeline_def(batch_size=4, device_id=0, num_threads=4, seed=42, enable_conditionals=True)
        def missing_shape():
            sample = types.Constant(np.array([[[]]], dtype=np.uint8))
            augments = rand_augment.get_rand_augment_suite(use_shape=True)
            return rand_augment.apply_rand_augment(augments, sample, 1, 20)

        missing_shape()

    with assert_raises(Exception, glob="The kwarg `shhape` is not used by any of the"):

        @pipeline_def(batch_size=4, device_id=0, num_threads=4, seed=42, enable_conditionals=True)
        def unused_kwarg():
            sample = types.Constant(np.array([[[]]], dtype=np.uint8))
            augments = rand_augment.get_rand_augment_suite(use_shape=True)
            return rand_augment.apply_rand_augment(augments, sample, 1, 20, shhape=42)

        unused_kwarg()
