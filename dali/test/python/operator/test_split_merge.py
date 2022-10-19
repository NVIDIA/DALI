# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

import numpy as np

from test_utils import check_batch, RandomlyShapedDataIterator

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
def split_merge_pipe(dev):
    input = fn.external_source(name="input", device=dev)
    pred = fn.external_source(name="predicate")
    true_branch, false_branch = fn.experimental._split(input, predicate=pred)
    true_rotated = fn.rotate(true_branch, angle=15)
    false_flipped = fn.flip(false_branch, horizontal=True)
    return fn.experimental._merge(true_rotated, false_flipped, predicate=pred)


def check_split_merge(dev, pred_gen):
    bs = 10
    kwargs = {
        "batch_size": bs,
        "num_threads": 4,
        "device_id": 0,
        "prefetch_queue_depth": 1  # so that it's easier to use external source
    }
    pipe_sm = split_merge_pipe(dev, **kwargs)
    pipe_true = rotate_pipe(dev, **kwargs)
    pipe_false = flip_pipe(dev, **kwargs)
    pipe_sm.build()
    pipe_true.build()
    pipe_false.build()
    data_iter = RandomlyShapedDataIterator(bs, min_shape=(20, 20, 3), max_shape=(40, 30, 3))
    data_iter = iter(data_iter)
    for _ in range(test_iters):
        predicate = [pred_gen(i) for i in range(bs)]
        data = next(data_iter)
        data_true = [data[i] for i in range(bs) if predicate[i]]
        data_false = [data[i] for i in range(bs) if not predicate[i]]
        pipe_sm.feed_input("input", data)
        pipe_sm.feed_input("predicate", predicate)
        pipe_true.feed_input("input", data_true)
        pipe_false.feed_input("input", data_false)
        out, = pipe_sm.run()
        out_true, = pipe_true.run() if data_true else ([], )
        out_false, = pipe_false.run() if data_false else ([], )
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


def test_split_merge():
    rng = np.random.default_rng()
    for dev in ["cpu", "gpu"]:
        for pred_gen in [
                lambda x: np.array(x < 3), lambda x: np.array(x % 2 == 0),
                lambda x: np.array(x % 3 == 0), lambda _: np.array(False),
                lambda _: rng.choice([np.array(True), np.array(False)])
        ]:
            yield check_split_merge, dev, pred_gen
