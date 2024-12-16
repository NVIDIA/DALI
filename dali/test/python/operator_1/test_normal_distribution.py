# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import scipy.stats as st
import random

test_types = [types.INT8, types.INT16, types.INT32, types.FLOAT, types.FLOAT64]


def random_shape(max_shape):
    return np.array([1 if s == 1 else np.random.randint(1, s) for s in max_shape], dtype=np.int32)


def random_shape_or_empty(max_shape):
    empty_sh = random.choice([True, False])
    if empty_sh:
        return np.array([200, 0, 3], np.int32)
    else:
        return np.array(
            [1 if s == 1 else np.random.randint(1, s) for s in max_shape], dtype=np.int32
        )


def check_normal_distribution(
    device,
    dtype,
    shape=None,
    use_shape_like_input=False,
    variable_shape=False,
    mean=0.0,
    stddev=1.0,
    variable_dist_params=False,
    shape_gen_f=None,
    niter=3,
    batch_size=3,
    device_id=0,
    num_threads=3,
):
    pipe = Pipeline(
        batch_size=batch_size, device_id=device_id, num_threads=num_threads, seed=123456
    )
    with pipe:
        shape_like_in = None
        shape_arg = None
        assert shape is None or shape_gen_f is None
        if variable_shape:
            if shape_gen_f is None:

                def shape_gen_f():
                    return random_shape(shape)

            if use_shape_like_input:
                shape_like_in = fn.external_source(
                    lambda: np.zeros(shape_gen_f()), device=device, batch=False
                )
                shape_out = shape_like_in.shape(device=device)
            else:
                shape_arg = fn.external_source(shape_gen_f, batch=False)
                shape_out = shape_arg
        else:
            if use_shape_like_input:
                shape_like_in = np.zeros(shape)
            else:
                shape_arg = shape
            # Can't make an empty list constant
            shape_out = types.Constant(
                shape if shape is not None and shape != () else (1,), dtype=types.INT32
            )

        mean_arg = None
        stddev_arg = None
        if variable_dist_params:
            mean_arg = fn.external_source(
                lambda: np.array(np.random.uniform(low=-100.0, high=100.0), dtype=np.float32),
                device="cpu",
                batch=False,
            )
            stddev_arg = fn.external_source(
                lambda: np.array(np.random.uniform(low=1.0, high=100.0), dtype=np.float32),
                device="cpu",
                batch=False,
            )
        else:
            mean_arg = mean
            stddev_arg = stddev
        inputs = [shape_like_in] if shape_like_in is not None else []
        out = fn.random.normal(
            *inputs, device=device, shape=shape_arg, mean=mean_arg, stddev=stddev_arg, dtype=dtype
        )
        pipe.set_outputs(out, shape_out, mean_arg, stddev_arg)
    for i in range(niter):
        outputs = pipe.run()
        out, shapes, means, stddevs = tuple(outputs[i].as_cpu() for i in range(len(outputs)))
        for sample_idx in range(batch_size):
            sample = np.array(out[sample_idx])
            if sample.shape == ():
                continue
            sample_shape = np.array(shapes[sample_idx])
            mean = np.array(means[sample_idx])
            stddev = np.array(stddevs[sample_idx])
            assert (sample.shape == sample_shape).all(), f"{sample.shape} != {sample_shape}"

            data = sample.flatten()
            data_len = len(data)

            # Checking sanity of the data
            if data_len >= 100 and dtype in [types.FLOAT, types.FLOAT64]:
                # Empirical rule:
                # ~68% of the observations within one standard deviation
                # ~95% of the observations within one standard deviation
                # ~99.7% of the observations within one standard deviation
                within_1stddevs = np.where(
                    (data > (mean - 1 * stddev)) & (data < (mean + 1 * stddev))
                )
                p1 = len(within_1stddevs[0]) / data_len
                within_2stddevs = np.where(
                    (data > (mean - 2 * stddev)) & (data < (mean + 2 * stddev))
                )
                p2 = len(within_2stddevs[0]) / data_len
                within_3stddevs = np.where(
                    (data > (mean - 3 * stddev)) & (data < (mean + 3 * stddev))
                )
                p3 = len(within_3stddevs[0]) / data_len
                assert p3 > 0.9, f"{p3}"  # leave some room
                assert p2 > 0.8, f"{p2}"  # leave some room
                assert p1 > 0.5, f"{p1}"  # leave some room

                # It's not 100% mathematically correct, but makes do in case of this test
                _, pvalues_anderson, _ = st.anderson(data, dist="norm")
                assert pvalues_anderson[2] > 0.5


def test_normal_distribution():
    niter = 3
    batch_size = 3
    for device in ("cpu", "gpu"):
        for dtype in test_types:
            for mean, stddev, variable_dist_params in [
                (0.0, 1.0, False),
                (111.0, 57.0, False),
                (0.0, 0.0, True),
            ]:
                for shape in [(100,), (10, 20, 30), (1, 2, 3, 4, 5, 6)]:
                    use_shape_like_in = False if shape is None else random.choice([True, False])
                    variable_shape = random.choice([True, False])
                    shape_arg = None
                    if variable_shape:

                        def shape_gen_f():
                            return random_shape(shape)

                    else:
                        shape_arg = shape
                        shape_gen_f = None
                    yield (
                        check_normal_distribution,
                        device,
                        dtype,
                        shape_arg,
                        use_shape_like_in,
                        variable_shape,
                        mean,
                        stddev,
                        variable_dist_params,
                        shape_gen_f,
                        niter,
                        batch_size,
                    )


def test_normal_distribution_scalar_and_one_elem():
    niter = 3
    batch_size = 3
    mean = 100.0
    stddev = 20.0
    for device in ("cpu", "gpu"):
        for dtype in [types.FLOAT, types.INT16]:
            for shape in [None, (), (1,)]:
                yield (
                    check_normal_distribution,
                    device,
                    dtype,
                    shape,
                    False,
                    False,
                    mean,
                    stddev,
                    False,
                    None,
                    niter,
                    batch_size,
                )


def test_normal_distribution_empty_shapes():
    niter = 3
    batch_size = 20
    dtype = types.FLOAT
    mean = 100.0
    stddev = 20.0
    max_shape = (200, 300, 3)
    for device in ("cpu", "gpu"):
        yield check_normal_distribution, device, dtype, (
            0,
        ), False, False, mean, stddev, False, None, niter, batch_size
        yield (
            check_normal_distribution,
            device,
            dtype,
            None,
            False,
            False,
            mean,
            stddev,
            False,
            lambda: random_shape_or_empty(max_shape),
            niter,
            batch_size,
        )
