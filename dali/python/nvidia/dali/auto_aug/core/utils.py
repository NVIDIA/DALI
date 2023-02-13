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

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples.")


def remap_bins_to_signed_magnitudes(magnitudes, randomly_negate):

    def remap_bin_idx(bin_idx):
        magnitude = magnitudes[bin_idx // 2]
        if randomly_negate and bin_idx % 2:
            magnitude = -magnitude
        return magnitude

    return np.array([remap_bin_idx(bin_idx) for bin_idx in range(2 * len(magnitudes))])


def split_samples_among_ops(op_range_lo, op_range_hi, ops, selected_op_idx, op_kwargs):
    assert op_range_lo <= op_range_hi
    if op_range_lo == op_range_hi:
        return ops[op_range_lo](**op_kwargs)
    mid = (op_range_lo + op_range_hi) // 2
    if selected_op_idx <= mid:
        return split_samples_among_ops(op_range_lo, mid, ops, selected_op_idx, op_kwargs)
    else:
        return split_samples_among_ops(mid + 1, op_range_hi, ops, selected_op_idx, op_kwargs)


def select(ops, selected_op_idx, op_kwargs):
    return split_samples_among_ops(0, len(ops) - 1, ops, selected_op_idx, op_kwargs)
