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
from nose2.tools import params

from nvidia.dali import fn
from nvidia.dali.pipeline import experimental
from nvidia.dali.auto_aug import auto_augment
from nvidia.dali.auto_aug.core import augmentation, Policy

from test_utils import get_dali_extra_path
from nose_utils import assert_raises

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')


@params(*tuple(enumerate(itertools.product((True, False), (True, False), (None, 0),
                                           (True, False)))))
def test_run_auto_aug(i, args):
    uniformly_resized, use_shape, fill_value, specify_translation_bounds = args
    batch_sizes = [1, 8, 7, 64, 13, 64, 128]
    batch_size = batch_sizes[i % len(batch_sizes)]

    @experimental.pipeline_def(enable_conditionals=True, batch_size=batch_size, num_threads=4,
                               device_id=0, seed=43)
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
        image = auto_augment.auto_augment_image_net(image, **extra)
        return image

    p = pipeline()
    p.build()
    for _ in range(3):
        p.run()


def test_unused_arg_fail():

    @experimental.pipeline_def(enable_conditionals=True, batch_size=5, num_threads=4, device_id=0,
                               seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        image_net_policy = auto_augment.get_image_net_policy()
        return auto_augment.apply_auto_augment(image_net_policy, image, misspelled_kwarg=100)

    msg = "The kwarg `misspelled_kwarg` is not used by any of the augmentations."
    with assert_raises(Exception, glob=msg):
        pipeline()


def test_missing_shape_fail():

    @experimental.pipeline_def(enable_conditionals=True, batch_size=5, num_threads=4, device_id=0,
                               seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        image_net_policy = auto_augment.get_image_net_policy(use_shape=True)
        return auto_augment.apply_auto_augment(image_net_policy, image)

    msg = "`translate_y` * provide it as `shape` argument to `apply_auto_augment` call"
    with assert_raises(Exception, glob=msg):
        pipeline()


def test_clashing_names():

    def get_first_augment():

        @augmentation
        def clashing_name(sample, _):
            pass

        return clashing_name

    def get_second_augment():

        @augmentation
        def clashing_name(sample, _):
            pass

        return clashing_name

    one = get_first_augment()
    another = get_second_augment()
    policy = Policy(name="DummyPolicy", num_magnitude_bins=11, sub_policies=[[(one, 0.1, 5),
                                                                              (another, 0.4, 7)]])
    policy_str = str(policy)
    assert 'clashing_name__0' in policy_str
    assert 'clashing_name__1' in policy_str


@params((True, 0), (True, 1), (False, 0), (False, 1))
def test_translation(use_shape, offset_fraction):
    # make sure the translation helper processes the args properly
    max_extent = 1000
    fill_value = 217
    params = {}
    if use_shape:
        params["max_translate_rel"] = offset_fraction
    else:
        params["max_translate_abs"] = offset_fraction * max_extent
    translate_y = auto_augment._get_translate_y(use_shape=use_shape, **params)
    policy = Policy(f"Policy_{use_shape}_{offset_fraction}", num_magnitude_bins=21,
                    sub_policies=[[(translate_y, 1, 20)]])

    @experimental.pipeline_def(enable_conditionals=True, batch_size=16, num_threads=4, device_id=0,
                               seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        image = fn.crop(image, crop=(max_extent, max_extent), out_of_bounds_policy="trim_to_shape")
        if use_shape:
            return auto_augment.apply_auto_augment(policy, image, fill_value=fill_value,
                                                   shape=fn.peek_image_shape(encoded_image))
        else:
            return auto_augment.apply_auto_augment(policy, image, fill_value=fill_value)

    p = pipeline()
    p.build()
    output, = p.run()
    output = [np.array(sample) for sample in output.as_cpu()]
    for i, sample in enumerate(output):
        sample = np.array(sample)
        if offset_fraction == 1:
            assert np.all(sample == fill_value), f"sample_idx: {i}"
        else:
            assert np.sum(sample == fill_value) / sample.size < 0.1, f"sample_idx: {i}"
