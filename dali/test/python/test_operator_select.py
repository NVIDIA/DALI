# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from numpy import random
from numpy.core.fromnumeric import shape
from nvidia.dali import Pipeline
import nvidia.dali as dali
from nvidia.dali.external_source import external_source
import nvidia.dali.fn as fn
import numpy as np
import random
import test_utils

random.seed(1234)
np.random.seed(1234)

def generate_data(ndim, ninp, type, max_batch_size):
    batch_size = np.random.randint(1, max_batch_size+1)
    inp_sel = np.random.randint(0 if ndim == 0 else -1, ninp, (batch_size,), dtype=np.int32)
    dtype = test_utils.dali_type_to_np(type)
    if ndim > 0:
        max_extent = max(10, int(100000 ** (1/ndim)))
        random_shape = lambda: np.random.randint(1, max_extent, (ndim,))
    else:
        random_shape = lambda: ()
    if type in (dali.types.FLOAT, dali.types.FLOAT64, dali.types.FLOAT16):
      rnd = lambda: np.random.random(random_shape()).astype(dtype=dtype)
    else:
      rnd = lambda: np.random.randint(0, 100, size=random_shape(), dtype=dtype)
    data = [[rnd() for _ in range(batch_size)] for _ in range(ninp)]
    return data + [inp_sel]

def make_layout(ndim, start='a'):
    out = ''
    for i in range(ndim):
        out += chr(ord(start) + i)
    return out

def _test_select(ndim, ninp, type, max_batch_size, device, layout):
    pipe = Pipeline(max_batch_size, 3, None if device == "cpu" else 0)
    with pipe:
        *inputs_cpu, idx = fn.external_source(
            source=lambda: generate_data(ndim, ninp, type, max_batch_size),
            num_outputs=ninp+1)
        inputs_cpu = list(inputs_cpu)
        input_layout = make_layout(ndim)
        inputs_cpu[ninp - 1] = fn.reshape(inputs_cpu[ninp-1], layout=input_layout)
        inputs = [x.gpu() for x in inputs_cpu] if device == "gpu" else inputs_cpu
        out = fn.select(*inputs, input_idx=idx, device=device, layout=layout)
        pipe.set_outputs(out, *inputs_cpu, idx)
    pipe.build()
    expected_layout = layout if layout is not None else input_layout
    for iter in range(5):
        out_tl, *input_tls, idx_tl = pipe.run()
        assert out_tl.layout() == expected_layout
        if device == "gpu":
            out_tl = out_tl.as_cpu()
        batch_size = len(out_tl)
        for inp in input_tls:
            assert len(inp) == batch_size, "Inconsistent batch size"
        for i in range(batch_size):
            idx = int(idx_tl.at(i))
            if idx < 0:
                assert out_tl.at(i).size == 0
            else:
                assert np.array_equal(out_tl.at(i), input_tls[idx].at(i))

def test_select():
    for device in ["cpu", "gpu"]:
        for ndim in (0, 1, 2, 3):
            for ninp in (1, 2, 3, 4):
                type = random.choice([dali.types.FLOAT, dali.types.UINT8, dali.types.INT32])
                layout = make_layout(ndim, 'p') if np.random.randint(0, 2) else None
                yield _test_select, ndim, ninp, type, 10, device, layout


def test_error_inconsistent_ndim():
    def check(device):
        pipe = Pipeline(1, 3, 0)
        pipe.set_outputs(fn.select(np.float32([0,1]), np.float32([[2,3,4]]), input_idx=1))
        pipe.build()
        try:
            pipe.run()
            assert False, "Expected an exception"
        except RuntimeError as e:
            assert "same number of dimensions" in str(e), "Unexpected exception"

    for device in ["gpu", "cpu"]:
        yield check, device

def test_error_inconsistent_type():
    def check(device):
        pipe = Pipeline(1, 3, 0)
        pipe.set_outputs(fn.select(np.float32([0,1]), np.int32([2,3,4]), input_idx=1))
        pipe.build()
        try:
            pipe.run()
            assert False, "Expected an exception"
        except RuntimeError as e:
            assert "same type" in str(e), "Unexpected exception"

    for device in ["gpu", "cpu"]:
        yield check, device

def test_error_input_out_of_range():
    def check(device):
        pipe = Pipeline(1, 3, 0)
        pipe.set_outputs(fn.select(np.float32([0,1]), np.float32([2,3,4]), input_idx=2))
        pipe.build()
        try:
            pipe.run()
            assert False, "Expected an exception"
        except RuntimeError as e:
            assert "Invalid input index" in str(e), "Unexpected exception"

    for device in ["gpu", "cpu"]:
        yield check, device


def test_error_empty_scalar():
    def check(device):
        pipe = Pipeline(1, 3, 0)
        pipe.set_outputs(fn.select(np.array(0, dtype=np.float32), input_idx=-1))
        pipe.build()
        try:
            pipe.run()
            assert False, "Expected an exception"
        except RuntimeError as e:
            assert "Invalid input index" in str(e), "Unexpected exception"

    for device in ["gpu", "cpu"]:
        yield check, device

def test_error_bad_layout():
    def check(device):
        pipe = Pipeline(1, 3, 0)
        pipe.set_outputs(fn.select(np.float32([[[0,1]]]), np.float32([[[2,3,4]]]), input_idx=0, layout="ab"))
        pipe.build()
        try:
            pipe.run()
            assert False, "Expected an exception"
        except RuntimeError as e:
            assert "valid layout" in str(e), "Unexpected exception"

    for device in ["gpu", "cpu"]:
        yield check, device
