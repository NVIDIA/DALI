# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function
from __future__ import division
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.tensors import TensorListGPU
import numpy as np
from nose.tools import assert_equals
import itertools

from test_utils import check_batch

batch_size = 4

# Shape of the samples
shape = (256, 256)

# A number used to test constant inputs
magic_number = 42

# We cannot have constant x constant operations with DALI op.
input_kinds = list(itertools.product(["cpu", "gpu"], ["cpu", "gpu", "const"])) +  list(
        itertools.product(["const"], ["cpu", "gpu"]))

# float16 is marked as TODO in backend for gpu
input_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64]

np_types_to_dali = {
    np.int8:    types.INT8,
    np.int16:   types.INT16,
    np.int32:   types.INT32,
    np.int64:   types.INT64,
    np.uint8:   types.UINT8,
    np.uint16:  types.UINT16,
    np.uint32:  types.UINT32,
    np.uint64:  types.UINT64,
    np.float16: types.FLOAT16,
    np.float32: types.FLOAT,
    np.float64: types.FLOAT64,
}

sane_operations = [((lambda x, y: x + y), "+"), ((lambda x, y: x - y), "-"),
                   ((lambda x, y: x * y), "*")]

def as_cpu(tl):
    if isinstance(tl, TensorListGPU):
        return tl.as_cpu()
    return tl


def max_dtype(kind, left_dtype, right_dtype):
    return np.dtype(kind + str(max(left_dtype.itemsize, right_dtype.itemsize)))

def float_bin_promote(left_dtype, right_dtype):
    if 'f' in left_dtype.kind and not 'f' in right_dtype.kind:
        return left_dtype
    if not 'f' in left_dtype.kind and 'f' in right_dtype.kind:
        return right_dtype
    return max_dtype('f', left_dtype, right_dtype)

def signed_unsigned_bin_promote(signed_type, unsigned_type):
    if signed_type.itemsize > unsigned_type.itemsize:
        return np.dtype('i' + str(signed_type.itemsize))
    itemsize = min(unsigned_type.itemsize * 2, 8)
    return np.dtype('i' + str(itemsize))

def bin_promote_dtype(left_dtype, right_dtype):
    if left_dtype == right_dtype:
        return left_dtype
    if 'f' in left_dtype.kind or 'f' in right_dtype.kind:
        return float_bin_promote(left_dtype, right_dtype)
    if 'i' in left_dtype.kind and 'i' in right_dtype.kind:
        return max_dtype('i', left_dtype, right_dtype)
    if 'u' in left_dtype.kind and 'u' in right_dtype.kind:
        return max_dtype('u', left_dtype, right_dtype)
    if 'i' in left_dtype.kind:
        return signed_unsigned_bin_promote(left_dtype, right_dtype)
    return signed_unsigned_bin_promote(right_dtype, left_dtype)

def hack_builtin_types(input_type):
    if type(input_type) == int:
        return np.int32
    elif type(input_type) == float:
        return np.float32
    else:
        return input_type

def bin_promote(left_type, right_type):
    left_dtype = np.dtype(hack_builtin_types(left_type))
    right_dtype = np.dtype(hack_builtin_types(right_type))
    return bin_promote_dtype(left_dtype, right_dtype).type

# For __truediv__ we promote integer results to float, otherwise proceed like with bin op
def div_promote(left_type, right_type):
    left_dtype = np.dtype(hack_builtin_types(left_type))
    right_dtype = np.dtype(hack_builtin_types(right_type))
    if 'f' not in left_dtype.kind and 'f' not in right_dtype.kind:
        return np.float32
    return float_bin_promote(left_dtype, right_dtype).type


def int_generator(shape, type):
    iinfo = np.iinfo(type)
    result = np.random.randint(iinfo.min / 2, iinfo.max / 2, shape, type)
    zero_mask = result == 0
    return result + zero_mask


class ExternalInputIterator(object):
    def __init__(self, batch_size, left_type, right_type):
        self.batch_size = batch_size
        self.left_type = left_type
        self.right_type = right_type
        self.gen_l = self.get_generator(self.left_type)
        self.gen_r = self.get_generator(self.right_type)
        self.shape = shape

    def __iter__(self):
        return self

    def __next__(self):
        left = []
        right = []
        for sample in range(self.batch_size):
            left.append(self.gen_l(self.shape))
            right.append(self.gen_r(self.shape))
        return (left, right)

    def get_generator(self, type):
        if type in [np.float16, np.float32, np.float64]:
            return lambda shape : type(np.random.rand(shape[0], shape[1]))
        else:
            return lambda shape : int_generator(shape, type)

    next = __next__


class ExprOpPipeline(Pipeline):
    def __init__(self, kinds, types, iterator, op, batch_size, num_threads, device_id):
        super(ExprOpPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.left_source = ops.ExternalSource()
        self.right_source = ops.ExternalSource()
        self.kinds = kinds
        self.types = types
        self.iterator = iterator
        self.op = op

    def define_graph(self):
        self.left = self.left_source()
        self.right = self.right_source()
        left_kind, right_kind = self.kinds
        left_type, right_type = self.types
        l = self.get_operand(self.left, left_kind, left_type)
        r = self.get_operand(self.right, right_kind, right_type)
        return self.left, self.right, self.op(l, r)

    def get_operand(self, operand, kind, operand_type):
        if kind == "const":
            return types.Constant(magic_number, np_types_to_dali[operand_type])
        elif kind == "cpu":
            return operand
        elif kind == "gpu":
            return operand.gpu()

    def iter_setup(self):
        (l, r) = self.iterator.next()
        self.feed_input(self.left, l)
        self.feed_input(self.right, r)

def get_numpy_input(input, kind, type):
    if kind == "const":
        return type(magic_number)
    else:
        return input.astype(type)

# Regular arithmetic ops that can be validated as straight numpy
def check_arithm_op(kinds, types, op, _):
    left_type, right_type = types
    left_kind, right_kind = kinds
    target_type = bin_promote(left_type, right_type)
    iterator = iter(ExternalInputIterator(batch_size, left_type, right_type))
    pipe = ExprOpPipeline(kinds, types, iterator, op, batch_size = batch_size, num_threads = 2,
            device_id = 0)
    pipe.build()
    pipe_out = pipe.run()
    l = as_cpu(pipe_out[0]).as_array()
    r = as_cpu(pipe_out[1]).as_array()
    out = as_cpu(pipe_out[2]).as_array()
    assert_equals(out.dtype, target_type)
    l_np = get_numpy_input(l, left_kind, target_type)
    r_np = get_numpy_input(r, right_kind, target_type)
    if 'f' in np.dtype(target_type).kind:
        np.testing.assert_allclose(out, op(l_np, r_np),
            rtol=1e-07 if target_type != np.float16 else 0.005)
    else:
        np.testing.assert_array_equal(out, op(l_np, r_np))

def test_arithmetic_ops():
    for kinds in input_kinds:
        for (op, op_desc) in sane_operations:
            for types_in in itertools.product(input_types, input_types):
                yield check_arithm_op, kinds, types_in, op, op_desc


# The div operator that always returns floating point values
def check_arithm_fdiv(kinds, types):
    left_type, right_type = types
    left_kind, right_kind = kinds
    target_type = div_promote(left_type, right_type)
    iterator = iter(ExternalInputIterator(batch_size, left_type, right_type))
    pipe = ExprOpPipeline(kinds, types, iterator, (lambda x, y : x / y), batch_size = batch_size,
            num_threads = 2, device_id = 0)
    pipe.build()
    pipe_out = pipe.run()
    l = as_cpu(pipe_out[0]).as_array()
    r = as_cpu(pipe_out[1]).as_array()
    out = as_cpu(pipe_out[2]).as_array()
    assert_equals(out.dtype, target_type)
    l_np = get_numpy_input(l, left_kind, target_type)
    r_np = get_numpy_input(r, right_kind, target_type)
    np.testing.assert_allclose(out, l_np / r_np,
        rtol=1e-07 if target_type != np.float16 else 0.005)

def test_arithmetic_float_division():
    for kinds in input_kinds:
        for types_in in itertools.product(input_types, input_types):
            yield check_arithm_fdiv, kinds, types_in

# The div operator behaves like C/C++ one
def check_arithm_div(kinds, types):
    left_type, right_type = types
    left_kind, right_kind = kinds
    target_type = bin_promote(left_type, right_type)
    iterator = iter(ExternalInputIterator(batch_size, left_type, right_type))
    pipe = ExprOpPipeline(kinds, types, iterator, (lambda x, y : x // y), batch_size = batch_size,
            num_threads = 2, device_id = 0)
    pipe.build()
    pipe_out = pipe.run()
    l = as_cpu(pipe_out[0]).as_array()
    r = as_cpu(pipe_out[1]).as_array()
    out = as_cpu(pipe_out[2]).as_array()
    assert_equals(out.dtype, target_type)
    l_np = get_numpy_input(l, left_kind, target_type)
    r_np = get_numpy_input(r, right_kind, target_type)
    if 'f' in np.dtype(target_type).kind:
        np.testing.assert_allclose(out, l_np / r_np,
            rtol=1e-07 if target_type != np.float16 else 0.005)
    else:
        # Approximate validation, as np does something different than C
        result = np.floor_divide(np.abs(l_np), np.abs(r_np))
        neg = ((l_np < 0) & (r_np > 0)) | ((l_np > 0) & (r_np < 0))
        pos = ~neg
        result = result * (pos * 1 - neg * 1)
        np.testing.assert_array_equal(out, result)

def test_arithmetic_division():
    for kinds in input_kinds:
        for types_in in itertools.product(input_types, input_types):
            yield check_arithm_div, kinds, types_in