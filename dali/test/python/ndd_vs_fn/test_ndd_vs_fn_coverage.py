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

import test_ndd_vs_fn_noinput
import test_ndd_vs_fn_image
from nvidia.dali.experimental.dynamic._ops import _all_ops


excluded_operators = [
    "_arithmetic_generic_op",  # Hidden operators are not part of this suite.
    "_conditional.merge",  # Hidden operators are not part of this suite.
    "_conditional.not_",  # Hidden operators are not part of this suite.
    "_conditional.split",  # Hidden operators are not part of this suite.
    "_conditional.validate_logical",  # Hidden operators are not part of this suite.
    "_shape",  # Hidden operators are not part of this suite.
    "_subscript_dim_check",  # Hidden operators are not part of this suite.
    "_tensor_subscript",  # Hidden operators are not part of this suite.
    "batch_permutation",  # BUG
    "bbox_rotate",  # BUG
    "decoders.image_random_crop",  # BUG
    "decoders.image_slice",  # BUG
    "decoders.inflate",  # TODO(mszolucha): Add inflate test.
    "experimental.decoders.image_random_crop",  # BUG
    "experimental.readers.fits",  # No input data in DALI_extra
    "io.file.read",  # BUG
    "permute_batch",  # BUG
    "random_resized_crop",  # BUG
    "readers.video_resize",  # BUG
    # Will be tested in following PRs:
    "audio_resample",
    "bb_flip",
    "bbox_paste",
    "box_encoder",
    "cast_like",
    "coord_flip",
    "debayer",
    "decoders.audio",
    "decoders.image",
    "decoders.image_crop",
    "decoders.numpy",
    "decoders.video",
    "element_extract",
    "expand_dims",
    "experimental.decoders.image",
    "experimental.decoders.image_crop",
    "experimental.decoders.image_slice",
    "experimental.inputs.video",
    "experimental.peek_image_shape",
    "experimental.readers.video",
    "experimental.remap",
    "filter",
    "full",
    "full_like",
    "jitter",
    "lookup_table",
    "mel_filter_bank",
    "mfcc",
    "noise.gaussian",
    "noise.salt_and_pepper",
    "noise.shot",
    "nonsilent_region",
    "one_hot",
    "optical_flow",
    "peek_image_shape",
    "power_spectrum",
    "preemphasis_filter",
    "random.beta",
    "random.choice",
    "random.coin_flip",
    "random.normal",
    "random.uniform",
    "random_bbox_crop",
    "random_crop_generator",
    "readers.caffe",
    "readers.caffe2",
    "readers.coco",
    "readers.file",
    "readers.mxnet",
    "readers.nemo_asr",
    "readers.numpy",
    "readers.tfrecord",
    "readers.video",
    "readers.webdataset",
    "reductions.std_dev",
    "reductions.variance",
    "roi_random_crop",
    "segmentation.random_mask_pixel",
    "segmentation.random_object_bbox",
    "segmentation.select_masks",
    "sequence_rearrange",
    "spectrogram",
    "squeeze",
    "to_decibels",
    "warp_affine",
]


def get_tested_operators():
    tested_operators = set()
    modules = [
        test_ndd_vs_fn_noinput,
        test_ndd_vs_fn_image,
    ]
    for module in modules:
        if hasattr(module, "tested_operators"):
            tested_operators.update(module.tested_operators)
    return tested_operators


def get_all_operators():
    ret = []
    for o in _all_ops:
        op_name = o._schema.ModulePath()
        op_name.append(o._fn_name)
        ret.append(".".join(op_name))
    return ret


def test_coverage():
    tested_operators = get_tested_operators()
    covered_operators = tested_operators.union(excluded_operators)
    all_operators = get_all_operators()

    untested_operators = [op for op in all_operators if op not in covered_operators]

    if untested_operators:
        print("\nOperators that are not covered:")
        for op in sorted(untested_operators):
            print(f"  - {op}")
        print(f"\nTotal not covered: {len(untested_operators)} out of {len(all_operators)}")
    else:
        print("All operators are tested!")

    assert len(untested_operators) == 0, f"Found {len(untested_operators)} untested operators"
