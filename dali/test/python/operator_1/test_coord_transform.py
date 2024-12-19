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

import random
import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn

from test_utils import check_batch, dali_type
from sequences_test_utils import ArgData, ArgDesc, sequence_suite_helper, ArgCb


def make_param(kind, shape):
    if kind == "input":
        return fn.random.uniform(range=(0, 1), shape=shape)
    elif kind == "scalar input":
        return fn.reshape(fn.random.uniform(range=(0, 1)), shape=[])
    elif kind == "vector":
        return np.random.rand(*shape).astype(np.float32)
    elif kind == "scalar":
        return np.random.rand()
    else:
        return None


def clip(value, type=None):
    try:
        info = np.iinfo(type)
        return np.clip(value, info.min, info.max)
    except AttributeError:
        return value


def make_data_batch(batch_size, in_dim, type):
    np.random.seed(1234)
    batch = []
    lo = 0
    hi = 1
    if np.issubdtype(type, np.integer):
        info = np.iinfo(type)
        # clip range to +/- 1000000 to prevent excessively large epsilons
        lo = max(info.min / 2, -1000000)
        hi = min(info.max / 2, 1000000)

    for i in range(batch_size):
        batch.append(
            (np.random.rand(np.random.randint(0, 10000), in_dim) * (hi - lo) + lo).astype(type)
        )
    return batch


def get_data_source(batch_size, in_dim, type):
    return lambda: make_data_batch(batch_size, in_dim, type)


def _run_test(device, batch_size, out_dim, in_dim, in_dtype, out_dtype, M_kind, T_kind):
    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=4, device_id=0, seed=1234)
    with pipe:
        X = fn.external_source(
            source=get_data_source(batch_size, in_dim, in_dtype), device=device, layout="NX"
        )
        M = None
        T = None
        MT = None
        if T_kind == "fused":
            MT = make_param(M_kind, [out_dim, in_dim + 1])
        else:
            M = make_param(M_kind, [out_dim, in_dim])
            T = make_param(T_kind, [out_dim])

        Y = fn.coord_transform(
            X,
            MT=MT.flatten().tolist() if isinstance(MT, np.ndarray) else MT,
            M=M.flatten().tolist() if isinstance(M, np.ndarray) else M,
            T=T.flatten().tolist() if isinstance(T, np.ndarray) else T,
            dtype=dali_type(out_dtype),
        )
        if M is None:
            M = 1
        if T is None:
            T = 0
        if MT is None:
            MT = 0

        M, T, MT = (
            (
                x
                if isinstance(x, dali.data_node.DataNode)
                else dali.types.Constant(x, dtype=dali.types.FLOAT)
            )
            for x in (M, T, MT)
        )

        pipe.set_outputs(X, Y, M, T, MT)

    for iter in range(3):
        outputs = pipe.run()
        outputs = [x.as_cpu() if hasattr(x, "as_cpu") else x for x in outputs]
        ref = []
        scale = 1
        for idx in range(batch_size):
            X = outputs[0].at(idx)
            if T_kind == "fused":
                MT = outputs[4].at(idx)
                if MT.size == 1:
                    M = MT
                    T = 0
                else:
                    M = MT[:, :-1]
                    T = MT[:, -1]
            else:
                M = outputs[2].at(idx)
                T = outputs[3].at(idx)

            if M.size == 1:
                Y = X.astype(np.float32) * M + T
            else:
                Y = np.matmul(X.astype(np.float32), M.transpose()) + T

            if np.issubdtype(out_dtype, np.integer):
                info = np.iinfo(out_dtype)
                Y = Y.clip(info.min, info.max)

            ref.append(Y)
            scale = max(scale, np.max(np.abs(Y)) - np.min(np.abs(Y))) if Y.size > 0 else 1
        avg = 1e-6 * scale
        eps = 1e-6 * scale
        if out_dtype != np.float32:  # headroom for rounding
            avg += 0.33
            eps += 0.5
        check_batch(outputs[1], ref, batch_size, eps, eps, expected_layout="NX")


def test_all():
    for device in ["cpu", "gpu"]:
        for M_kind in [None, "vector", "scalar", "input", "scalar input"]:
            for T_kind in [None, "vector", "scalar", "input", "scalar input"]:
                for batch_size in [1, 3]:
                    yield (
                        _run_test,
                        device,
                        batch_size,
                        3,
                        3,
                        np.float32,
                        np.float32,
                        M_kind,
                        T_kind,
                    )

    for device in ["cpu", "gpu"]:
        for in_dtype in [np.uint8, np.uint16, np.int16, np.int32, np.float32]:
            for out_dtype in set([in_dtype, np.float32]):
                for batch_size in [1, 8]:
                    yield (
                        _run_test,
                        device,
                        batch_size,
                        3,
                        3,
                        in_dtype,
                        out_dtype,
                        "input",
                        "input",
                    )

    for device in ["cpu", "gpu"]:
        for M_kind in ["input", "vector", None]:
            for in_dim in [1, 2, 3, 4, 5, 6]:
                if M_kind == "vector" or M_kind == "input":
                    out_dims = [1, 2, 3, 4, 5, 6]
                else:
                    out_dims = [in_dim]
                for out_dim in out_dims:
                    yield (
                        _run_test,
                        device,
                        2,
                        out_dim,
                        in_dim,
                        np.float32,
                        np.float32,
                        M_kind,
                        "vector",
                    )

    for device in ["cpu", "gpu"]:
        for MT_kind in ["vector", "input", "scalar"]:
            for in_dim in [1, 2, 3, 4, 5, 6]:
                if MT_kind == "vector" or MT_kind == "input":
                    out_dims = [1, 2, 3, 4, 5, 6]
                else:
                    out_dims = [in_dim]
                for out_dim in out_dims:
                    yield (
                        _run_test,
                        device,
                        2,
                        out_dim,
                        in_dim,
                        np.float32,
                        np.float32,
                        MT_kind,
                        "fused",
                    )


def _test_empty_input(device):
    pipe = dali.pipeline.Pipeline(batch_size=2, num_threads=4, device_id=0, seed=1234)
    with pipe:
        X = fn.external_source(
            source=[[np.zeros([0, 3]), np.zeros([0, 3])]], device="cpu", layout="AB"
        )
        Y = fn.coord_transform(X, M=(1, 2, 3, 4, 5, 6), T=(1, 2))
        pipe.set_outputs(Y)
    o = pipe.run()
    assert o[0].layout() == "AB"
    assert len(o[0]) == 2
    for i in range(len(o[0])):
        assert o[0].at(0).size == 0


def test_empty_input():
    for device in ["cpu", "gpu"]:
        yield _test_empty_input, device


def test_sequences():
    rng = random.Random(42)
    np_rng = np.random.default_rng(12345)
    max_batch_size = 64
    max_num_frames = 50
    num_points = 30
    num_iters = 4

    def points():
        return np.float32(np_rng.uniform(-100, 250, (num_points, 2)))

    def rand_range(limit):
        return range(rng.randint(1, limit) + 1)

    def m(sample_desc):
        angles = np_rng.uniform(-np.pi, np.pi, 2)
        scales = np_rng.uniform(0, 5, 2)
        c = np.cos(angles[0])
        s = np.sin(angles[1])
        return np.array([[c * scales[0], -s], [s, c * scales[1]]], dtype=np.float32)

    def t(sample_desc):
        return np.float32(np_rng.uniform(-100, 250, 2))

    def mt(sample_desc):
        return np.append(m(sample_desc), t(sample_desc).reshape(-1, 1), axis=1)

    input_cases = [
        (fn.coord_transform, {}, [ArgCb("M", m, True)]),
        (fn.coord_transform, {}, [ArgCb("T", t, True)]),
        (fn.coord_transform, {}, [ArgCb("MT", mt, True)]),
        (fn.coord_transform, {}, [ArgCb("MT", mt, False)]),
        (fn.coord_transform, {}, [ArgCb("M", m, True), ArgCb("T", t, True)]),
        (fn.coord_transform, {}, [ArgCb("M", m, False), ArgCb("T", t, True)]),
    ]

    input_seq_data = [
        [
            np.array([points() for _ in rand_range(max_num_frames)], dtype=np.float32)
            for _ in rand_range(max_batch_size)
        ]
        for _ in range(num_iters)
    ]

    main_input = ArgData(desc=ArgDesc(0, "F", "", "F**"), data=input_seq_data)

    yield from sequence_suite_helper(rng, [main_input], input_cases, num_iters)

    input_broadcast_cases = [
        (fn.coord_transform, {}, [ArgCb(0, lambda _: points(), False, "cpu")], ["cpu"]),
        (fn.coord_transform, {}, [ArgCb(0, lambda _: points(), False, "gpu")], ["cpu"]),
    ]

    input_mt_data = [
        [
            np.array([mt(None) for _ in rand_range(max_num_frames)], dtype=np.float32)
            for _ in rand_range(max_batch_size)
        ]
        for _ in range(num_iters)
    ]

    main_input = ArgData(desc=ArgDesc("MT", "F", "", "F**"), data=input_mt_data)

    yield from sequence_suite_helper(rng, [main_input], input_broadcast_cases, num_iters)
