# Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.fn as fn
from nose_utils import nottest
from nvidia.dali import pipeline_def
from test_utils import np_type_to_dali
import itertools
from nose2.tools import params


def ref_cast(x, dtype):
    if np.issubdtype(dtype, np.integer):
        lo = np.iinfo(dtype).min
        hi = np.iinfo(dtype).max
        if np.issubdtype(x.dtype, np.floating):
            x = np.round(x)
        return x.clip(lo, hi).astype(dtype)
    else:
        return x.astype(dtype)


def random_shape(rng, ndim: int, max_size: int):
    if ndim == 0:
        return []
    max_size = int(max_size ** (1 / ndim))
    return list(rng.integers(1, max_size, [ndim]))


def replace_with_empty_volumes(rng, input, empty_volume_policy):
    """Replaces samples with 0-volumed ones if possible.

    Parameters
    ----------
    rng :
        rng
    input : List of np.array
        Batch to process
    empty_volume_policy : str
        one of "left", "right, "middle", "mixed", "all", to indicate if the batch suffix, prefix,
        infix or all of them should be randomly replaced with 0-volumed samples

    Returns
    -------
    List of np.array
    """
    if empty_volume_policy is None:
        return input
    if len(input[0].shape) == 0:
        return input
    if empty_volume_policy == "mixed":
        left = replace_with_empty_volumes(rng, input, "left")
        left_and_mid = replace_with_empty_volumes(rng, left, "middle")
        return replace_with_empty_volumes(rng, left_and_mid, "right")
    if empty_volume_policy == "all":
        start = 0
        end = len(input)
    elif empty_volume_policy == "left":
        start = 0
        end = rng.integers(1, len(input) // 3)
    elif empty_volume_policy == "right":
        start = rng.integers(len(input) * 2 // 3, len(input) - 1)
        end = len(input)
    elif empty_volume_policy == "middle":
        start = rng.integers(1 + len(input) // 3, len(input) * 2 // 3)
        end = rng.integers(start + 1, len(input) - 1)
    for i in range(start, end):
        shape = list(input[i].shape)
        shape[0] = 0
        input[i] = np.zeros(dtype=input[i].dtype, shape=shape)
    return input


def generate(
    rng,
    ndim: int,
    batch_size: int,
    in_dtype: np.dtype,
    out_dtype: np.dtype,
    empty_volume_policy: str,
):
    lo, hi = -1000, 1000
    if np.issubdtype(out_dtype, np.integer):
        lo = np.iinfo(out_dtype).min
        hi = np.iinfo(out_dtype).max
        if hi < np.iinfo(np.int64).max:
            r = hi - lo
            hi += r // 2
            lo -= r // 2
        if np.issubdtype(in_dtype, np.integer):
            lo = max(np.iinfo(in_dtype).min, lo)
            hi = min(np.iinfo(in_dtype).max, hi)
        else:
            lo = max(-np.finfo(in_dtype).max, lo)
            hi = min(np.finfo(in_dtype).max, hi)

    max_size = 100000 // batch_size
    out = [
        rng.uniform(lo, hi, size=random_shape(rng, ndim, max_size)).astype(in_dtype)
        for _ in range(batch_size)
    ]
    out = replace_with_empty_volumes(rng, out, empty_volume_policy)
    if np.issubdtype(in_dtype, np.floating) and np.issubdtype(out_dtype, np.integer):
        for x in out:
            # avoid exactly halfway numbers - rounding is different for CPU and GPU
            halfway = x[x - np.floor(x) == 0.5]
            x[x - np.floor(x) == 0.5] = np.nextafter(halfway, np.Infinity)
    return out


rng = np.random.default_rng(1234)


@nottest
def _test_operator_cast(ndim, batch_size, in_dtype, out_dtype, device, empty_volume_policy=None):
    def src():
        return generate(rng, ndim, batch_size, in_dtype, out_dtype, empty_volume_policy)

    @pipeline_def(
        batch_size=batch_size,
        num_threads=4,
        device_id=None if device == "cpu" else 0,
    )
    def cast_pipe():
        inp = fn.external_source(src)
        inp_dev = inp.gpu() if device == "gpu" else inp
        return inp, fn.cast(inp_dev, dtype=np_type_to_dali(out_dtype))

    pipe = cast_pipe()
    for _ in range(10):
        inp, out = pipe.run()
        if device == "gpu":
            out = out.as_cpu()
        ref = [ref_cast(np.array(x), out_dtype) for x in inp]

        # work around a bug in numpy: when the argument is a scalar fp32 or fp16, nextafter
        # promotes it to fp64, resulting in insufficient epsilon - we want an epsilon of the
        # type specified in out_dtype
        eps = (
            0
            if np.issubdtype(out_dtype, np.integer)
            else (np.nextafter(out_dtype([1]), 2) - 1.0)[0]
        )

        for i in range(batch_size):
            if not np.allclose(out[i], ref[i], eps):
                matI = np.array(inp[i])
                matO = np.array(out[i])
                matR = ref[i]
                mask = np.logical_not(np.isclose(matO, matR, eps))
                print(f"At sample {i}:\nI:\n{matI}\nO\n{matO}\nR\n{matR}")
                print(f"Differences at {mask}:\nI:\n{matI[mask]}\nO\n{matO[mask]}\nR\n{matR[mask]}")
                print(f"Result: {np.count_nonzero(mask)} wrong values out of {mask.size}.")
                assert np.array_equal(out[i], ref[i])


def test_operator_cast():
    types = [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float16,
        np.float32,
    ]
    for device in ["cpu", "gpu"]:
        for in_type in types:
            for out_type in types:
                ndim = rng.integers(0, 4)
                batch_size = rng.integers(1, 11)
                yield _test_operator_cast, ndim, batch_size, in_type, out_type, device


def test_operator_cast_empty_volumes():
    types = [np.uint8, np.int32, np.float32]
    for device in ["cpu", "gpu"]:
        for in_type in types:
            for out_type in types:
                ndim = rng.integers(0, 4)

                batch_size = rng.integers(12, 64)
                for empty_volume_policy in [
                    rng.choice(["left", "right", "middle", "mixed"]),
                    "all",
                ]:
                    yield (
                        _test_operator_cast,
                        ndim,
                        batch_size,
                        in_type,
                        out_type,
                        device,
                        empty_volume_policy,
                    )


@params(
    *itertools.product(
        (("cpu", "cpu"), ("gpu", "cpu"), ("gpu", "gpu")),
        (np.uint8, np.int32, np.float32),
        (np.uint8, np.int32, np.float32),
    )
)
def test_cast_like(devices, dtype_in, dtype_out):
    @pipeline_def(batch_size=1, num_threads=4, device_id=0)
    def cast_pipe():
        device_left, device_right = devices
        data0 = fn.random.uniform(
            range=[0, 255], dtype=np_type_to_dali(dtype_in), device=device_left
        )
        data1 = fn.random.uniform(
            range=[0, 255], dtype=np_type_to_dali(dtype_out), device=device_right
        )
        return fn.cast_like(data0, data1)

    p = cast_pipe()
    (out,) = p.run()
    expected_type = np_type_to_dali(dtype_out)
    assert out.dtype == expected_type, f"{out.dtype} != {expected_type}"
