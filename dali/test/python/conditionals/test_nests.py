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

from nvidia.dali import fn, pipeline_def, types
import numpy as np

from test_utils import check_batch
from nose_utils import assert_raises


def test_select_impls():
    # Test recursive select that returns tuple of outputs (due to the ops having 2 outputs).
    # Without supporting nested structures it encountered a ((DataNode, DataNode),) branch output
    # and crashed.

    def _select_fwd(op_range_lo, op_range_hi, ops, selected_op_idx, op_args, op_kwargs):
        assert op_range_lo <= op_range_hi
        if op_range_lo == op_range_hi:
            return ops[op_range_lo](*op_args, **op_kwargs)
        mid = (op_range_lo + op_range_hi) // 2
        if selected_op_idx <= mid:
            ret = _select_fwd(op_range_lo, mid, ops, selected_op_idx, op_args, op_kwargs)
        else:
            ret = _select_fwd(mid + 1, op_range_hi, ops, selected_op_idx, op_args, op_kwargs)
        return ret

    def _select_unpack(op_range_lo, op_range_hi, ops, selected_op_idx, op_args, op_kwargs):
        assert op_range_lo <= op_range_hi
        if op_range_lo == op_range_hi:
            return ops[op_range_lo](*op_args, **op_kwargs)
        mid = (op_range_lo + op_range_hi) // 2
        if selected_op_idx <= mid:
            a, b = _select_unpack(op_range_lo, mid, ops, selected_op_idx, op_args, op_kwargs)
        else:
            a, b = _select_unpack(mid + 1, op_range_hi, ops, selected_op_idx, op_args, op_kwargs)
        return a, b

    def select(ops, selected_op_idx, *op_args, unpacking_select=False, **op_kwargs):
        if unpacking_select:
            return _select_unpack(0, len(ops) - 1, ops, selected_op_idx, op_args, op_kwargs)
        else:
            return _select_fwd(0, len(ops) - 1, ops, selected_op_idx, op_args, op_kwargs)

    def rotate(image, label):
        image = fn.rotate(image, angle=42)
        return image, label

    def color(image, label):
        image = fn.color_twist(image, saturation=0)
        return image, label

    @pipeline_def(enable_conditionals=True, num_threads=4, batch_size=8, device_id=0)
    def pipeline(unpacking_select):
        image = types.Constant(np.full((200, 300, 3), 42, dtype=np.uint8), device="cpu")
        label = types.Constant(np.array(1), device="cpu")
        ops = [rotate, color]
        op_idx = fn.random.uniform(values=list(range(len(ops))))
        image, label = select(
            ops, op_idx, image=image, label=label, unpacking_select=unpacking_select
        )
        return image, label

    pipe_unpacking = pipeline(unpacking_select=True)
    pipe_unpacking.run()

    pipe_forwarding = pipeline(unpacking_select=False)
    pipe_forwarding.run()


def test_dicts():
    @pipeline_def(enable_conditionals=True, num_threads=4, batch_size=8, device_id=0)
    def pipeline():
        pred = fn.external_source(source=lambda x: np.array(x.idx_in_batch % 2), batch=False)
        if pred:
            out = {"out": np.array(2)}
        else:
            out = {"out": np.array(1)}
        return out["out"]

    pipe = pipeline()
    (out,) = pipe.run()
    check_batch(out, [i % 2 + 1 for i in range(8)])

    @pipeline_def(enable_conditionals=True, num_threads=4, batch_size=8, device_id=0)
    def pipeline_op():
        pred = fn.external_source(source=lambda x: np.array(x.idx_in_batch % 2), batch=False)
        data = types.Constant(np.array(42), device="cpu")
        if pred:
            out = {"out": data - 1}
        else:
            out = {"out": data + 1}
        return out["out"]

    pipe_op = pipeline_op()
    (out,) = pipe_op.run()
    check_batch(out, [41 if i % 2 else 43 for i in range(8)])


def test_tuples():
    @pipeline_def(enable_conditionals=True, num_threads=4, batch_size=8, device_id=0)
    def pipeline():
        pred = fn.external_source(source=lambda x: np.array(x.idx_in_batch % 2), batch=False)
        data = types.Constant(np.array(42), device="cpu")
        if pred:
            out = (data, data + 10, data + 20)
        else:
            out = (np.array(-10), data, data * 2)
        a, b, c = out
        return a, b, c

    pipe = pipeline()
    (
        a,
        b,
        c,
    ) = pipe.run()
    check_batch(a, [42 if i % 2 else -10 for i in range(8)])
    check_batch(b, [52 if i % 2 else 42 for i in range(8)])
    check_batch(c, [62 if i % 2 else 84 for i in range(8)])


def test_nesting_error():
    @pipeline_def(enable_conditionals=True, num_threads=4, batch_size=8, device_id=0)
    def pipeline():
        pred = fn.external_source(source=lambda x: np.array(x.idx_in_batch % 2), batch=False)
        if pred:
            out = {"out": np.array(2), "mismatched": np.array(9999)}
        else:
            out = {"out": np.array(1)}
        return out

    with assert_raises(
        ValueError,
        glob=(
            "*Divergent data found in different branches of `if/else` control"
            " flow statement. Variables in all code paths are merged into common"
            " output batches. The values assigned to a given variable need to"
            " have the same nesting structure in every code path"
            " (both `if` branches).*"
            "*The two structures don't have the same nested structure*"
            "*The two dictionaries don't have the same set of keys."
            " First structure has keys type=list str=*'out', 'mismatched'*,"
            " while second structure has keys type=list str=*'out'*"
        ),
    ):
        _ = pipeline()
