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

from typing import List
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.auto_aug.core._augmentation import Augmentation


def split_samples_among_ops(
    op_range_lo: int,
    op_range_hi: int,
    ops: List[Augmentation],
    selected_op_idx: _DataNode,
    op_args,
    op_kwargs,
):
    assert op_range_lo <= op_range_hi
    if op_range_lo == op_range_hi:
        return ops[op_range_lo](*op_args, **op_kwargs)
    mid = (op_range_lo + op_range_hi) // 2
    if selected_op_idx <= mid:
        return split_samples_among_ops(op_range_lo, mid, ops, selected_op_idx, op_args, op_kwargs)
    else:
        return split_samples_among_ops(
            mid + 1, op_range_hi, ops, selected_op_idx, op_args, op_kwargs
        )


def select(ops: List[Augmentation], selected_op_idx: _DataNode, *op_args, **op_kwargs):
    """
    Applies the operator from the operators list based on the provided index as if by calling
    `ops[selected_op_idx](**op_kwargs)`.

    The `selected_op_idx` must be a batch of indices from [0, len(ops) - 1] range. The `op_kwargs`
    can contain other data nodes, they will be split into partial batches accordingly.
    """
    return split_samples_among_ops(0, len(ops) - 1, ops, selected_op_idx, op_args, op_kwargs)
