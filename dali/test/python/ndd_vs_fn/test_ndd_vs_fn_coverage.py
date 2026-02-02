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

from ndd_vs_fn_test_utils import sign_off
from nvidia.dali.experimental.dynamic._ops import _all_ops


excluded_operators = [
    # "batch_permutation",  # Random op - not handled yet
    # "bbox_rotate",  # requires floating point input (not image-like)
    # "decoders.image_random_crop",  # TODO(michalz): Add decoder tests
    # "decoders.image_slice",  # TODO(michalz): Add decoder tests
    # "decoders.inflate",  # TODO(mszolucha): Add inflate test.
    # "experimental.decoders.image_random_crop",  # TODO(michalz): Add decoder tests
    # "experimental.readers.fits",  # No input data in DALI_extra
    # "io.file.read",  # Special operator, needs handwritten test
    # "permute_batch",  # Special operator, needs handwritten test
    # "random_resized_crop",  # TODO(michalz): Add tests for operators with random state
    # "readers.video_resize",  # TODO(michalz): Needs handwritten test
    # "plugin.video.decoder",  # Still kind of experimental, skipping
    # # Will be tested in following PRs:
    # "audio_resample",
    # "bb_flip",
    # "bbox_paste",
    # "box_encoder",
    # "cast_like",
    # "coord_flip",
    # "debayer",
    # "decoders.audio",
    # "decoders.image",
    # "decoders.image_crop",
    # "decoders.numpy",
    # "decoders.video",
    # "element_extract",
    # "experimental.decoders.image",
    # "experimental.decoders.image_crop",
    # "experimental.decoders.image_slice",
    # "experimental.inputs.video",
    # "experimental.peek_image_shape",
    # "experimental.readers.video",
    # "experimental.remap",
    # "filter",
    # "full",
    # "full_like",
    # "lookup_table",
    # "mel_filter_bank",
    # "mfcc",
    # "noise.gaussian",
    # "noise.salt_and_pepper",
    # "noise.shot",
    # "nonsilent_region",
    # "one_hot",
    # "optical_flow",
    # "peek_image_shape",
    # "power_spectrum",
    # "preemphasis_filter",
    # "random.beta",
    # "random.choice",
    # "random.coin_flip",
    # "random.normal",
    # "random.uniform",
    # "random_bbox_crop",
    # "random_crop_generator",
    # "readers.caffe",
    # "readers.caffe2",
    # "readers.coco",
    # "readers.file",
    # "readers.mxnet",
    # "readers.nemo_asr",
    # "readers.numpy",
    # "readers.tfrecord",
    # "readers.video",
    # "readers.webdataset",
    # "reductions.std_dev",
    # "reductions.variance",
    # "roi_random_crop",
    # "segmentation.random_mask_pixel",
    # "segmentation.random_object_bbox",
    # "segmentation.select_masks",
    # "sequence_rearrange",
    # "spectrogram",
    # "squeeze",
    # "to_decibels",
    # "warp_affine",
]


def get_all_operators():
    ret = []
    for o in _all_ops:
        if o._schema.IsInternal() or o._schema.IsDocHidden() or o._schema_name.startswith("_"):
            continue  # skip internal/hidden operators
        op_name = o._schema.ModulePath()
        op_name.append(o._fn_name)
        ret.append(".".join(op_name))
    return ret


def test_coverage():
    covered_operators = sign_off.tested_ops
    eligible_operators = set(get_all_operators()).difference(excluded_operators)

    untested_operators = [op for op in eligible_operators if op not in covered_operators]

    if untested_operators:
        print("\nOperators that are not covered:")
        for op in sorted(untested_operators):
            print(f"  - {op}")
        print(f"\nTotal not covered: {len(untested_operators)} out of {len(eligible_operators)}")
        if len(excluded_operators):
            print(f"{len(excluded_operators)} operators were excluded from the test.")
    else:
        if len(excluded_operators):
            print("All eligible operators are tested.")
            print(f"{len(excluded_operators)} operators were excluded from the test.")
        else:
            print("All operators are tested.")

    assert len(untested_operators) == 0, f"Found {len(untested_operators)} untested operators"
