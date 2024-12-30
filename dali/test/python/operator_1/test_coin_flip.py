# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline


def random_shape(max_shape, diff=100):
    # Produces a random shape close to the max shape
    for s in max_shape:
        assert s > diff
    return np.array([np.random.randint(s - diff, s) for s in max_shape], dtype=np.int32)


def check_coin_flip(
    device="cpu", batch_size=32, max_shape=[1e5], p=None, use_shape_like_input=False
):
    pipe = Pipeline(batch_size=batch_size, device_id=0, num_threads=3, seed=123456)
    with pipe:

        def shape_gen_f():
            return random_shape(max_shape)

        shape_arg = None
        inputs = []
        shape_out = None
        if max_shape is not None:
            if use_shape_like_input:
                shape_like_in = dali.fn.external_source(
                    lambda: np.zeros(shape_gen_f()), device=device, batch=False
                )
                inputs += [shape_like_in]
                shape_out = shape_like_in.shape(device=device)
            else:
                shape_arg = dali.fn.external_source(shape_gen_f, batch=False)
                shape_out = shape_arg
        outputs = [dali.fn.random.coin_flip(*inputs, device=device, probability=p, shape=shape_arg)]
        if shape_out is not None:
            outputs += [shape_out]
        pipe.set_outputs(*outputs)
    outputs = tuple(out.as_cpu() for out in pipe.run())
    data_out = outputs[0]
    shapes_out = None
    if max_shape is not None:
        shapes_out = outputs[1]
    p = p if p is not None else 0.5
    for i in range(batch_size):
        data = np.array(data_out[i])
        assert np.logical_or(data == 0, data == 1).all()
        if max_shape is not None:
            sample_shape = np.array(shapes_out[i])
            assert (data.shape == sample_shape).all()
            total = len(data)
            positive = np.count_nonzero(data)
            np.testing.assert_allclose(p, positive / total, atol=0.005)  # +/- -.5%


def test_coin_flip():
    batch_size = 8
    for device in ["cpu", "gpu"]:
        for max_shape, use_shape_like_in in [([100000], False), ([100000], True), (None, False)]:
            for probability in [None, 0.7, 0.5, 0.0, 1.0]:
                yield check_coin_flip, device, batch_size, max_shape, probability, use_shape_like_in
