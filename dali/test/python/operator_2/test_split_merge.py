# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import numpy as np

from test_utils import check_batch, RandomlyShapedDataIterator
from nose_utils import assert_raises
from nose2.tools import params

test_iters = 4


def to_batch(tl, batch_size):
    return [np.array(tl[i]) for i in range(batch_size)]


@pipeline_def
def rotate_pipe(dev):
    input = fn.external_source(name="input", device=dev)
    return fn.rotate(input, angle=15)


@pipeline_def
def flip_pipe(dev):
    input = fn.external_source(name="input", device=dev)
    return fn.flip(input, horizontal=True)


@pipeline_def
def conditional_split_merge_pipe(dev):
    input = fn.external_source(name="input", device=dev)
    pred = fn.external_source(name="predicate")
    true_branch, false_branch = fn._conditional.split(input, predicate=pred)
    true_rotated = fn.rotate(true_branch, angle=15)
    false_flipped = fn.flip(false_branch, horizontal=True)
    return fn._conditional.merge(true_rotated, false_flipped, predicate=pred)


def check_conditional_split_merge(dev, pred_gen):
    bs = 10
    kwargs = {
        "batch_size": bs,
        "num_threads": 4,
        "device_id": 0,
        "prefetch_queue_depth": 1,  # so that it's easier to use external source
    }
    pipe_sm = conditional_split_merge_pipe(dev, **kwargs)
    pipe_true = rotate_pipe(dev, **kwargs)
    pipe_false = flip_pipe(dev, **kwargs)
    data_iter = RandomlyShapedDataIterator(bs, min_shape=(20, 20, 3), max_shape=(40, 30, 3))
    data_iter = iter(data_iter)
    for _ in range(test_iters):
        predicate = [pred_gen(i) for i in range(bs)]
        data = next(data_iter)
        data_true = [data[i] for i in range(bs) if predicate[i]]
        data_false = [data[i] for i in range(bs) if not predicate[i]]
        pipe_sm.feed_input("input", data)
        pipe_sm.feed_input("predicate", predicate)
        if data_true:
            pipe_true.feed_input("input", data_true)
            (out_true,) = pipe_true.run()
        else:
            out_true = []
        if data_false:
            pipe_false.feed_input("input", data_false)
            (out_false,) = pipe_false.run()
        else:
            out_false = []
        (out,) = pipe_sm.run()
        out_baseline = []
        idx_true = 0
        idx_false = 0
        for p in predicate:
            if p:
                out_baseline.append(out_true[idx_true])
                idx_true = idx_true + 1
            else:
                out_baseline.append(out_false[idx_false])
                idx_false = idx_false + 1
        if dev == "gpu":
            out = [out[i].as_cpu() for i in range(bs)]
            out_baseline = [out_baseline[i].as_cpu() for i in range(bs)]
        check_batch(out, out_baseline, bs)


def test_conditional_split_merge():
    rng = np.random.default_rng()
    for dev in ["cpu", "gpu"]:
        for pred_gen in [
            lambda x: np.array(x < 3),
            lambda x: np.array(x % 2 == 0),
            lambda x: np.array(x % 3 == 0),
            lambda _: np.array(False),
            lambda _: rng.choice([np.array(True), np.array(False)]),
        ]:
            yield check_conditional_split_merge, dev, pred_gen


@pipeline_def
def conditional_split_merge_reinterpret_pipe(dtype, layout, shape):
    batch_size = Pipeline.current().max_batch_size
    input = fn.external_source(
        source=[[np.full((10, 10, 3), 42, dtype=np.int32) for _ in range(batch_size)]], cycle=True
    )
    pred = fn.external_source(
        source=[[np.array(i % 2 == 0, dtype=bool) for i in range(batch_size)]], cycle=True
    )
    true_branch, false_branch = fn._conditional.split(input, predicate=pred)
    false_changed = fn.reinterpret(false_branch, dtype=dtype, layout=layout, shape=shape)
    return fn._conditional.merge(true_branch, false_changed, predicate=pred)


def run_conditional_split_merge_reinterpret(dtype, layout, shape):
    bs = 10
    kwargs = {
        "batch_size": bs,
        "num_threads": 4,
        "device_id": 0,
        "prefetch_queue_depth": 1,  # so that it's easier to use external source
    }
    pipe = conditional_split_merge_reinterpret_pipe(dtype, layout, shape, **kwargs)
    pipe.run()


@params(
    (types.UINT32, None, None, "types*"),
    (None, "HWC", None, "layouts*"),
    (None, None, [10, -1], "sample dimensions*"),
)
def test_fail_conditional_split_merge(dtype, layout, shape, err_glob):
    base = (
        "Divergent data found in different branches of conditional operation. All paths in "
        "conditional operation are merged into one batch which must have consistent type, "
        "number of dimensions, layout and other metadata. Found distinct "
    )

    with assert_raises(RuntimeError, glob=base + err_glob):
        run_conditional_split_merge_reinterpret(dtype, layout, shape)
