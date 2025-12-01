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

from nvidia.dali.pipeline import Pipeline, DataNode
from nvidia.dali import pipeline_def, fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.math as math
import numpy as np
from nose_utils import attr, raises, assert_raises, assert_equals
from nose2.tools import params
import itertools

from test_utils import np_type_to_dali


def list_product(*args):
    return list(itertools.product(*args))


# Some test in this file are marked as `slow`. They cover all possible type and input kind
# combinations. The rest of the test cover only subset of selected cases to allow
# running time reduction.

batch_size = 4


# Shape of the samples, currently forces the sample to have be covered by more than 1 tile
def shape_big(arg_idx):
    return [(1024, 1024)] * batch_size


# For the coverage of all type combinations we use smaller batch
def shape_small(arg_idx):
    return [(42, 3), (4, 16), (8, 2), (1, 64)]


# A number used to test constant inputs
magic_number = 7

unary_input_kinds = [
    "cpu",
    "gpu",
    "cpu_scalar",
    "gpu_scalar",
    "cpu_scalar_legacy",
    "gpu_scalar_legacy",
]

all_input_kinds = [
    "cpu",
    "gpu",
    "cpu_scalar",
    "gpu_scalar",
    "cpu_scalar_legacy",
    "gpu_scalar_legacy",
    "const",
]

# We cannot have 'Constant x Constant' operations with DALI op.
# And scalar is still a represented by a Tensor, so 'Scalar x Constant' is the same
# as 'Tensor x Constant'.
bin_input_kinds = list_product(["cpu", "gpu"], all_input_kinds) + list_product(
    ["cpu_scalar", "gpu_scalar", "const"], ["cpu", "gpu"]
)

ternary_input_kinds = list_product(all_input_kinds, all_input_kinds, all_input_kinds)
ternary_input_kinds.remove(("const", "const", "const"))

integer_types = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

# float16 is marked as TODO in backend for gpu
float_types = [np.float32, np.float64]

input_types = integer_types + float_types

selected_input_types = [np.bool_, np.int32, np.uint8, np.float32]
selected_input_arithm_types = [np.int32, np.uint8, np.float32]

selected_bin_input_kinds = [
    ("cpu", "cpu"),
    ("gpu", "gpu"),
    ("cpu", "cpu_scalar"),
    ("gpu", "gpu_scalar"),
    ("const", "cpu"),
    ("const", "gpu"),
]

selected_ternary_input_kinds = [
    ("cpu", "cpu", "cpu"),
    ("gpu", "gpu", "gpu"),
    ("cpu", "const", "const"),
    ("gpu", "const", "const"),
    ("gpu", "cpu", "cpu_scalar"),
    ("cpu_scalar", "cpu_scalar", "cpu_scalar"),
]

bench_ternary_input_kinds = [
    ("cpu", "cpu", "cpu"),
    ("gpu", "gpu", "gpu"),
    ("cpu", "const", "const"),
    ("gpu", "const", "const"),
    ("cpu", "cpu", "const"),
    ("gpu", "gpu", "const"),
]

unary_operations = [((lambda x: +x), "+"), ((lambda x: -x), "-")]


def sane_pow(x, y):
    if np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer):
        # numpy likes to rise errors, we prefer to have integers to negative powers result in 0.
        return np.where(y >= 0, np.power(x, y, where=y >= 0), 0)
    else:
        return np.power(x, y)


# For math functions we used limited ranges to not have too many NaNs or exceptions in the test.


def pos_range(*types):
    return [(1, 20) if np.issubdtype(t, np.integer) else (0.5, 20.0) for t in types]


# The range that is supposed to be [-1, 1], but we extend it a bit.


def one_range(*types):
    return [(-2, 2) if np.issubdtype(t, np.integer) else (-1.5, 1.5) for t in types]


def limited_range(*types):
    # Limit the range so we do not end with comparing just the infinities in results.
    return [(-30, 30) for _ in types]


def pow_range(*_):
    return [(-15, 15), (-4, 4)]


def default_range(*types):
    return [None for _ in types]


math_function_operations = [
    ((lambda x: math.sqrt(x)), (lambda x: np.sqrt(x)), "sqrt", pos_range, 1e-6),
    ((lambda x: math.rsqrt(x)), (lambda x: 1.0 / np.sqrt(x)), "rsqrt", pos_range, 1e-5),
    ((lambda x: math.cbrt(x)), (lambda x: np.cbrt(x)), "cbrt", default_range, 1e-6),
    ((lambda x: math.exp(x)), (lambda x: np.exp(x)), "exp", limited_range, 1e-6),
    ((lambda x: math.log(x)), (lambda x: np.log(x)), "log", pos_range, 1e-6),
    ((lambda x: math.log2(x)), (lambda x: np.log2(x)), "log2", pos_range, 1e-6),
    ((lambda x: math.log10(x)), (lambda x: np.log10(x)), "log10", pos_range, 1e-6),
    ((lambda x: math.fabs(x)), (lambda x: np.fabs(x)), "fabs", default_range, 1e-6),
    ((lambda x: math.floor(x)), (lambda x: np.floor(x)), "floor", default_range, 1e-6),
    ((lambda x: math.ceil(x)), (lambda x: np.ceil(x)), "ceil", default_range, 1e-6),
    ((lambda x: math.sin(x)), (lambda x: np.sin(x)), "sin", default_range, 1e-6),
    ((lambda x: math.cos(x)), (lambda x: np.cos(x)), "cos", default_range, 1e-6),
    ((lambda x: math.tan(x)), (lambda x: np.tan(x)), "tan", default_range, 1e-6),
    ((lambda x: math.asin(x)), (lambda x: np.arcsin(x)), "asin", one_range, 1e-6),
    ((lambda x: math.acos(x)), (lambda x: np.arccos(x)), "acos", one_range, 1e-6),
    ((lambda x: math.atan(x)), (lambda x: np.arctan(x)), "atan", default_range, 1e-6),
    ((lambda x: math.sinh(x)), (lambda x: np.sinh(x)), "sinh", default_range, 1e-6),
    ((lambda x: math.cosh(x)), (lambda x: np.cosh(x)), "cosh", default_range, 1e-6),
    ((lambda x: math.tanh(x)), (lambda x: np.tanh(x)), "tanh", default_range, 1e-6),
    ((lambda x: math.asinh(x)), (lambda x: np.arcsinh(x)), "asinh", limited_range, 1e-6),
    ((lambda x: math.acosh(x)), (lambda x: np.arccosh(x)), "acosh", pos_range, 1e-6),
    ((lambda x: math.atanh(x)), (lambda x: np.arctanh(x)), "atanh", one_range, 1e-6),
]

sane_operations = [
    ((lambda x, y: x + y), "+", default_range),
    ((lambda x, y: x - y), "-", default_range),
    ((lambda x, y: x * y), "*", default_range),
    (((lambda x, y: x**y), sane_pow), "**", pow_range),
    (((lambda x, y: math.pow(x, y)), sane_pow), "pow", pow_range),
    (((lambda x, y: math.min(x, y)), (lambda x, y: np.minimum(x, y))), "min", default_range),
    (((lambda x, y: math.max(x, y)), (lambda x, y: np.maximum(x, y))), "max", default_range),
]

floaty_operations = [
    (((lambda x, y: x / y), (lambda x, y: x / y)), "/", default_range),
    (((lambda x, y: math.fpow(x, y)), sane_pow), "fpow", pow_range),
    (((lambda x, y: math.atan2(x, y)), (lambda x, y: np.arctan2(x, y))), "atan2", default_range),
]

bitwise_operations = [
    ((lambda x, y: x & y), "&"),
    ((lambda x, y: x | y), "|"),
    ((lambda x, y: x ^ y), "^"),
]

comparisons_operations = [
    ((lambda x, y: x == y), "=="),
    ((lambda x, y: x != y), "!="),
    ((lambda x, y: x < y), "<"),
    ((lambda x, y: x <= y), "<="),
    ((lambda x, y: x > y), ">"),
    ((lambda x, y: x >= y), ">="),
]

# The observable behavior for hi < lo is the same as numpy
ternary_operations = [
    (((lambda v, lo, hi: math.clamp(v, lo, hi)), (lambda v, lo, hi: np.clip(v, lo, hi))), "clamp")
]


def max_dtype(kind, left_dtype, right_dtype):
    return np.dtype(kind + str(max(left_dtype.itemsize, right_dtype.itemsize)))


def float_bin_promote(left_dtype, right_dtype):
    if "f" in left_dtype.kind and "f" not in right_dtype.kind:
        return left_dtype
    if "f" not in left_dtype.kind and "f" in right_dtype.kind:
        return right_dtype
    return max_dtype("f", left_dtype, right_dtype)


def signed_unsigned_bin_promote(signed_type, unsigned_type):
    # Treat the booleans as smaller than anything
    if unsigned_type.kind == "b":
        return signed_type
    if signed_type.itemsize > unsigned_type.itemsize:
        return np.dtype("i" + str(signed_type.itemsize))
    itemsize = min(unsigned_type.itemsize * 2, 8)
    return np.dtype("i" + str(itemsize))


def bin_promote_dtype(left_dtype, right_dtype):
    if left_dtype == right_dtype:
        return left_dtype
    if "f" in left_dtype.kind or "f" in right_dtype.kind:
        return float_bin_promote(left_dtype, right_dtype)
    if "b" in left_dtype.kind and "b" in right_dtype.kind:
        return np.dtype(np.bool_)
    if "i" in left_dtype.kind and "i" in right_dtype.kind:
        return max_dtype("i", left_dtype, right_dtype)
    # Check if both types are either 'b' (bool) or 'u' (unsigned), 'b' op 'b' is checked above
    if set([left_dtype.kind, right_dtype.kind]) <= set("bu"):
        return max_dtype("u", left_dtype, right_dtype)
    # One of the types is signed
    if "i" in left_dtype.kind:
        return signed_unsigned_bin_promote(left_dtype, right_dtype)
    return signed_unsigned_bin_promote(right_dtype, left_dtype)


def hack_builtin_types(input_type):
    if type(input_type) is int:
        return np.int32
    elif type(input_type) is float:
        return np.float32
    else:
        return input_type


def bin_promote(left_type, right_type):
    left_dtype = np.dtype(hack_builtin_types(left_type))
    right_dtype = np.dtype(hack_builtin_types(right_type))
    return bin_promote_dtype(left_dtype, right_dtype).type


def div_promote(left_type, right_type):
    # For __truediv__ we promote integer results to float, otherwise proceed like with bin op
    left_dtype = np.dtype(hack_builtin_types(left_type))
    right_dtype = np.dtype(hack_builtin_types(right_type))
    if "f" not in left_dtype.kind and "f" not in right_dtype.kind:
        return np.float32
    return float_bin_promote(left_dtype, right_dtype).type


def int_generator(shape, type, no_zeros, limited_range):
    iinfo = np.iinfo(type)
    if limited_range is not None:
        low, high = limited_range
        low = max(iinfo.min, low)
        high = min(iinfo.max, high)
    else:
        low, high = iinfo.min / 2, iinfo.max / 2
    result = np.random.randint(low, high, shape, type)
    zero_mask = result == 0
    if no_zeros:
        return result + zero_mask
    return result


def bool_generator(shape, no_zeros):
    result = np.random.choice(a=[True, False], size=shape)
    zero_mask = result == False  # noqa:E712 Trust me, it's intended math op comparison with False
    if no_zeros:
        return result | zero_mask
    return result


def float_generator(shape, type, _, limited_range):
    if limited_range is not None:
        low, high = limited_range
    else:
        low, high = 0.0, 1.0
    if isinstance(shape, int):
        return type(low + np.random.rand(shape) * (high - low))
    elif len(shape) == 2:
        return type(low + np.random.rand(*shape) * (high - low))
    else:
        return type([low + np.random.rand() * (high - low)])


class ExternalInputIterator(object):
    """
    Generates inputs of required shapes and types
    The number of inputs is based on the length of tuple `types`, if types is a single element
    it is considered we should generate 1 output.
    If the kind contains 'scalar', than the result is batch of scalar tensors.
    the "shape" of `kinds` arguments should match the `types` argument - single elements or tuples
    of the same arity.
    """

    def __init__(
        self, batch_size, shape_gen, types, kinds, disallow_zeros=None, limited_range=None
    ):
        try:
            self.length = len(types)
        except TypeError:
            types = (types,)
            kinds = (kinds,)
            self.length = 1
        if not disallow_zeros:
            disallow_zeros = (False,) * self.length
        if limited_range is None:
            limited_range = (None,) * self.length
        self.batch_size = batch_size
        self.types = types
        self.gens = []
        self.shapes = []
        for i in range(self.length):
            self.gens += [self.get_generator(self.types[i], disallow_zeros[i], limited_range[i])]
            if "scalar" not in kinds[i]:
                self.shapes += [shape_gen(i)]
            elif "scalar_legacy" in kinds[i]:
                self.shapes += [[(1,)] * batch_size]
            else:
                self.shapes += [[]]  # empty shape, special 0D scalar

    def __iter__(self):
        return self

    def __next__(self):
        out = ()
        for i in range(self.length):
            batch = []
            # Handle 0D scalars
            if self.shapes[i] == []:
                batch = self.gens[i](self.batch_size)
            else:
                for sample in range(self.batch_size):
                    batch.append(self.gens[i](self.shapes[i][sample]))
            out = out + (batch,)
        return out

    def get_generator(self, type, no_zeros, limited_range):
        if type == np.bool_:
            return lambda shape: bool_generator(shape, no_zeros)
        elif type in [np.float16, np.float32, np.float64]:
            return lambda shape: float_generator(shape, type, no_zeros, limited_range)
        else:
            return lambda shape: int_generator(shape, type, no_zeros, limited_range)

    next = __next__


class ExprOpPipeline(Pipeline):
    def __init__(self, kinds, types, iterator, op, batch_size, num_threads, device_id):
        super(ExprOpPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        try:
            self.length = len(types)
        except TypeError:
            types = (types,)
            kinds = (kinds,)
            self.length = 1
        self.external_source = []
        for i in range(self.length):
            self.external_source.append(ops.ExternalSource())
        self.kinds = kinds
        self.types = types
        self.iterator = iterator
        self.op = op

    def define_graph(self):
        self.source = []
        inputs = []
        for i in range(self.length):
            self.source.append(self.external_source[i]())
            inputs.append(self.get_operand(self.source[i], self.kinds[i], self.types[i]))
        return tuple(self.source) + (self.op(*inputs),)

    def get_operand(self, operand, kind, operand_type):
        if kind == "const":
            return types.Constant(magic_number, np_type_to_dali(operand_type))
        elif "cpu" in kind:
            return operand
        elif "gpu" in kind:
            return operand.gpu()

    def iter_setup(self):
        inputs = self.iterator.next()
        for i in range(len(inputs)):
            self.feed_input(self.source[i], inputs[i])


def get_numpy_input(input, kind, orig_type, target_type):
    """
    :param orig_type: the original type of used input
    :param target_type: the type of the result after type promotions
    """
    if kind == "const":
        return target_type(orig_type(magic_number))
    else:
        if "scalar" in kind:
            return input.astype(target_type).reshape(input.shape)
        else:
            return input.astype(target_type)


def extract_un_data(pipe_out, sample_id, kind, target_type):
    input = np.array(pipe_out[0][sample_id].as_cpu())
    out = np.array(pipe_out[1][sample_id].as_cpu())
    assert_equals(out.dtype, target_type)
    in_np = get_numpy_input(input, kind, input.dtype.type, target_type)
    return in_np, out


def extract_data(pipe_out, sample_id, kinds, target_type):
    """
    Extract output for given sample_id from the pipeline
    Expand the data based on the kinds parameter and optionally cast it into target type
    as numpy does types promotions a bit differently.
    """
    arity = len(kinds)
    inputs = []
    for i in range(arity):
        dali_in = np.array(pipe_out[i][sample_id].as_cpu())
        numpy_in = get_numpy_input(
            dali_in,
            kinds[i],
            dali_in.dtype.type,
            target_type if target_type is not None else dali_in.dtype.type,
        )
        inputs.append(numpy_in)
    out = np.array(pipe_out[arity][sample_id].as_cpu())
    return tuple(inputs) + (out,)


def check_unary_op(kind, type, op, shape, _):
    # Regular arithmetic ops that can be validated as straight numpy
    iterator = iter(ExternalInputIterator(batch_size, shape, type, kind))
    pipe = ExprOpPipeline(
        kind, type, iterator, op, batch_size=batch_size, num_threads=2, device_id=0
    )
    pipe_out = pipe.run()
    for sample in range(batch_size):
        in_np, out = extract_un_data(pipe_out, sample, kind, type)
        if "f" in np.dtype(type).kind:
            np.testing.assert_allclose(out, op(in_np), rtol=1e-07 if type != np.float16 else 0.005)
        else:
            np.testing.assert_array_equal(out, op(in_np))


def test_unary_arithmetic_ops():
    for kinds in unary_input_kinds:
        for op, op_desc in unary_operations:
            for types_in in input_types:
                if types_in != np.bool_:
                    yield check_unary_op, kinds, types_in, op, shape_small, op_desc


def test_unary_arithmetic_ops_big():
    for kinds in unary_input_kinds:
        for op, op_desc in unary_operations:
            yield check_unary_op, kinds, np.int8, op, shape_big, op_desc


def check_math_function_op(kind, type, op, np_op, shape, get_range, op_desc, eps):
    is_integer = type not in [np.float16, np.float32, np.float64]
    limted_range = get_range(type)
    iterator = iter(
        ExternalInputIterator(batch_size, shape, type, kind, limited_range=limted_range)
    )
    pipe = ExprOpPipeline(
        kind, type, iterator, op, batch_size=batch_size, num_threads=2, device_id=0
    )
    pipe_out = pipe.run()
    out_type = np.float32 if is_integer else type
    for sample in range(batch_size):
        in_np, out = extract_un_data(pipe_out, sample, kind, out_type)
        np.testing.assert_allclose(
            out, np_op(in_np.astype(out_type)), rtol=eps if type != np.float16 else 0.005
        )


def test_math_function_ops():
    for kinds in unary_input_kinds:
        for op, np_op, op_desc, get_range, eps in math_function_operations:
            for types_in in input_types:
                if types_in != np.bool_:
                    yield (
                        check_math_function_op,
                        kinds,
                        types_in,
                        op,
                        np_op,
                        shape_small,
                        get_range,
                        op_desc,
                        eps,
                    )


def test_math_function_ops_big():
    for kinds in unary_input_kinds:
        for op, np_op, op_desc, get_range, eps in math_function_operations:
            for types_in in [np.int8]:
                yield (
                    check_math_function_op,
                    kinds,
                    types_in,
                    op,
                    np_op,
                    shape_big,
                    get_range,
                    op_desc,
                    eps,
                )


def check_arithm_op(kinds, types, op, shape, get_range, op_desc):
    # Regular arithmetic ops that can be validated as straight numpy
    if isinstance(op, tuple):
        dali_op, numpy_op = op
    else:
        dali_op = numpy_op = op
    left_type, right_type = types
    target_type = bin_promote(left_type, right_type)
    iterator = iter(
        ExternalInputIterator(
            batch_size, shape, types, kinds, limited_range=get_range(left_type, right_type)
        )
    )
    pipe = ExprOpPipeline(
        kinds, types, iterator, dali_op, batch_size=batch_size, num_threads=2, device_id=0
    )
    pipe_out = pipe.run()
    for sample in range(batch_size):
        l_np, r_np, out = extract_data(pipe_out, sample, kinds, target_type)
        assert_equals(out.dtype, target_type)
        if "f" in np.dtype(target_type).kind:
            np.testing.assert_allclose(
                out, numpy_op(l_np, r_np), rtol=1e-06 if target_type != np.float16 else 0.005
            )
        else:
            np.testing.assert_array_equal(out, numpy_op(l_np, r_np))


def check_ternary_op(kinds, types, op, shape, _):
    # Regular arithmetic ops that can be validated as straight numpy
    if isinstance(op, tuple):
        dali_op, numpy_op = op
    else:
        dali_op = numpy_op = op
    target_type = bin_promote(bin_promote(types[0], types[1]), types[2])
    iterator = iter(ExternalInputIterator(batch_size, shape, types, kinds))
    pipe = ExprOpPipeline(
        kinds, types, iterator, dali_op, batch_size=batch_size, num_threads=2, device_id=0
    )
    pipe_out = pipe.run()
    for sample in range(batch_size):
        x, y, z, out = extract_data(pipe_out, sample, kinds, target_type)
        assert_equals(out.dtype, target_type)
        if "f" in np.dtype(target_type).kind:
            np.testing.assert_allclose(
                out, numpy_op(x, y, z), rtol=1e-07 if target_type != np.float16 else 0.005
            )
        else:
            np.testing.assert_array_equal(out, numpy_op(x, y, z))


def test_arithmetic_ops_big():
    for kinds in bin_input_kinds:
        for op, op_desc, get_range in sane_operations:
            for types_in in [(np.int8, np.int8)]:
                yield check_arithm_op, kinds, types_in, op, shape_big, get_range, op_desc


def test_arithmetic_ops_selected():
    for kinds in selected_bin_input_kinds:
        for op, op_desc, get_range in sane_operations:
            for types_in in itertools.product(selected_input_types, selected_input_types):
                if types_in != (np.bool_, np.bool_) or op_desc == "*":
                    yield check_arithm_op, kinds, types_in, op, shape_small, get_range, op_desc


@attr("slow")
def slow_test_arithmetic_ops():
    for kinds in bin_input_kinds:
        for op, op_desc, get_range in sane_operations:
            for types_in in itertools.product(input_types, input_types):
                if types_in != (np.bool_, np.bool_) or op_desc == "*":
                    yield check_arithm_op, kinds, types_in, op, shape_small, get_range, op_desc


def test_ternary_ops_big():
    for kinds in selected_ternary_input_kinds:
        for op, op_desc in ternary_operations:
            for types_in in [
                (np.int32, np.int32, np.int32),
                (np.int32, np.int8, np.int16),
                (np.int32, np.uint8, np.float32),
            ]:
                yield check_ternary_op, kinds, types_in, op, shape_big, op_desc


def test_ternary_ops_selected():
    for kinds in selected_ternary_input_kinds:
        for op, op_desc in ternary_operations:
            for types_in in itertools.product(
                selected_input_arithm_types,
                selected_input_arithm_types,
                selected_input_arithm_types,
            ):
                yield check_ternary_op, kinds, types_in, op, shape_small, op_desc


# Only selected types, otherwise it takes too long


@attr("slow")
def slow_test_ternary_ops_kinds():
    for kinds in ternary_input_kinds:
        for op, op_desc in ternary_operations:
            for types_in in [
                (np.int32, np.int32, np.int32),
                (np.float32, np.int32, np.int32),
                (np.uint8, np.float32, np.float32),
                (np.int32, np.float32, np.float32),
            ]:
                yield check_ternary_op, kinds, types_in, op, shape_small, op_desc


@attr("slow")
def slow_test_ternary_ops_types():
    for kinds in selected_ternary_input_kinds:
        for op, op_desc in ternary_operations:
            for types_in in list_product(input_types, input_types, input_types):
                if types_in == (np.bool_, np.bool_, np.bool_):
                    continue
                yield check_ternary_op, kinds, types_in, op, shape_small, op_desc


def test_bitwise_ops_selected():
    for kinds in selected_bin_input_kinds:
        for op, op_desc in bitwise_operations:
            for types_in in itertools.product(selected_input_types, selected_input_types):
                if types_in[0] in integer_types and types_in[1] in integer_types:
                    yield check_arithm_op, kinds, types_in, op, shape_small, default_range, op_desc


@attr("slow")
def slow_test_bitwise_ops():
    for kinds in bin_input_kinds:
        for op, op_desc in bitwise_operations:
            for types_in in itertools.product(input_types, input_types):
                if types_in[0] in integer_types and types_in[1] in integer_types:
                    yield check_arithm_op, kinds, types_in, op, shape_small, default_range, op_desc


def check_comparsion_op(kinds, types, op, shape, _):
    # Comparisons - should always return bool
    iterator = iter(ExternalInputIterator(batch_size, shape, types, kinds))
    pipe = ExprOpPipeline(
        kinds, types, iterator, op, batch_size=batch_size, num_threads=2, device_id=0
    )
    pipe_out = pipe.run()
    for sample in range(batch_size):
        l_np, r_np, out = extract_data(pipe_out, sample, kinds, None)
        assert_equals(out.dtype, np.bool_)
        np.testing.assert_array_equal(out, op(l_np, r_np), err_msg=f"{l_np} op\n{r_np} =\n{out}")


def test_comparison_ops_selected():
    for kinds in selected_bin_input_kinds:
        for op, op_desc in comparisons_operations:
            for types_in in itertools.product(selected_input_types, selected_input_types):
                yield check_comparsion_op, kinds, types_in, op, shape_small, op_desc


@attr("slow")
def slow_test_comparison_ops():
    for kinds in bin_input_kinds:
        for op, op_desc in comparisons_operations:
            for types_in in itertools.product(input_types, input_types):
                yield check_comparsion_op, kinds, types_in, op, shape_small, op_desc


# The div operator that always returns floating point values


def check_arithm_binary_float(kinds, types, op, shape, get_range, _):
    if isinstance(op, tuple):
        dali_op, numpy_op = op
    else:
        dali_op = numpy_op = op
    left_type, right_type = types
    target_type = div_promote(left_type, right_type)
    iterator = iter(
        ExternalInputIterator(
            batch_size,
            shape,
            types,
            kinds,
            (False, True),
            limited_range=get_range(left_type, right_type),
        )
    )
    pipe = ExprOpPipeline(
        kinds, types, iterator, dali_op, batch_size=batch_size, num_threads=2, device_id=0
    )
    pipe_out = pipe.run()
    for sample in range(batch_size):
        l_np, r_np, out = extract_data(pipe_out, sample, kinds, target_type)
        assert_equals(out.dtype, target_type)
        np.testing.assert_allclose(
            out,
            numpy_op(l_np, r_np),
            rtol=1e-06 if target_type != np.float16 else 0.005,
            err_msg=f"{l_np} op\n{r_np} =\n{out}",
        )


def test_arithmetic_binary_float_big():
    for kinds in bin_input_kinds:
        for types_in in [(np.int8, np.int8)]:
            for op, op_desc, get_range in floaty_operations:
                yield check_arithm_binary_float, kinds, types_in, op, shape_big, get_range, op_desc


def test_arithmetic_binary_float_selected():
    for kinds in selected_bin_input_kinds:
        for types_in in itertools.product(selected_input_types, selected_input_types):
            for op, op_desc, get_range in floaty_operations:
                if types_in != (np.bool_, np.bool_):
                    yield (
                        check_arithm_binary_float,
                        kinds,
                        types_in,
                        op,
                        shape_small,
                        get_range,
                        op_desc,
                    )


@attr("slow")
def slow_test_arithmetic_binary_float():
    for kinds in bin_input_kinds:
        for types_in in itertools.product(input_types, input_types):
            for op, op_desc, get_range in floaty_operations:
                if types_in != (np.bool_, np.bool_):
                    yield (
                        check_arithm_binary_float,
                        kinds,
                        types_in,
                        op,
                        shape_small,
                        get_range,
                        op_desc,
                    )


def check_arithm_div(kinds, types, shape):
    # The div operator behaves like C/C++ one
    left_type, right_type = types
    target_type = bin_promote(left_type, right_type)
    iterator = iter(ExternalInputIterator(batch_size, shape, types, kinds, (False, True)))
    pipe = ExprOpPipeline(
        kinds,
        types,
        iterator,
        (lambda x, y: x // y),
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
    )
    pipe_out = pipe.run()
    for sample in range(batch_size):
        l_np, r_np, out = extract_data(pipe_out, sample, kinds, target_type)
        assert_equals(out.dtype, target_type)
        if "f" in np.dtype(target_type).kind:
            np.testing.assert_allclose(
                out, l_np / r_np, rtol=1e-07 if target_type != np.float16 else 0.005
            )
        else:
            # Approximate validation, as np does something different than C
            result = np.floor_divide(np.abs(l_np), np.abs(r_np))
            neg = ((l_np < 0) & (r_np > 0)) | ((l_np > 0) & (r_np < 0))
            pos = ~neg
            result = result * (pos * 1 - neg * 1)
            np.testing.assert_array_equal(out, result)


def test_arithmetic_division_big():
    for kinds in bin_input_kinds:
        for types_in in [(np.int8, np.int8)]:
            yield check_arithm_div, kinds, types_in, shape_big


def test_arithmetic_division_selected():
    for kinds in selected_bin_input_kinds:
        for types_in in itertools.product(selected_input_types, selected_input_types):
            if types_in != (np.bool_, np.bool_):
                yield check_arithm_div, kinds, types_in, shape_small


@attr("slow")
def slow_test_arithmetic_division():
    for kinds in bin_input_kinds:
        for types_in in itertools.product(input_types, input_types):
            if types_in != (np.bool_, np.bool_):
                yield check_arithm_div, kinds, types_in, shape_small


def check_raises(kinds, types, op, shape):
    if isinstance(op, tuple):
        dali_op = op[0]
    else:
        dali_op = op
    iterator = iter(ExternalInputIterator(batch_size, shape, types, kinds))
    pipe = ExprOpPipeline(
        kinds, types, iterator, dali_op, batch_size=batch_size, num_threads=2, device_id=0
    )
    pipe.run()


def check_raises_re(kinds, types, op, shape, _, msg):
    with assert_raises(RuntimeError, regex=msg):
        check_raises(kinds, types, op, shape)


@raises(
    TypeError,
    glob=(
        '"DataNode" was used in conditional context - it might have been used '
        "in truth evaluation for `if` statement, logical expression or cast to a "
        "boolean. To use conditional execution via `if` statements you need to "
        "specify `enable_conditionals=True` in `@nvidia.dali.pipeline_def` "
        "decorator. You can read more about conditional execution in specific "
        "section of the Pipeline documentation. Bool conversion can be achieved "
        "with the `cast` operator."
    ),
)
def check_raises_te(kinds, types, op, shape, _):
    check_raises(kinds, types, op, shape)


# Arithmetic operations between booleans that are not allowed
bool_disallowed = [
    ((lambda x, y: x + y), "+"),
    ((lambda x, y: x - y), "-"),
    ((lambda x, y: x / y), "/"),
    ((lambda x, y: x / y), "//"),
    ((lambda x, y: x**y), "**"),
]


def test_bool_disallowed():
    error_msg = (
        "Input[s]? to arithmetic operator `[\\S]*` cannot be [a]?[ ]?boolean[s]?."
        " Consider using bitwise operator[s]?"
    )
    for kinds in unary_input_kinds:
        for op, _, op_desc, _, _ in math_function_operations:
            yield check_raises_re, kinds, np.bool_, op, shape_small, op_desc, error_msg
    for kinds in bin_input_kinds:
        for op, op_desc in bool_disallowed:
            yield check_raises_re, kinds, (np.bool_, np.bool_), op, shape_small, op_desc, error_msg
    for kinds in selected_ternary_input_kinds:
        for op, op_desc in ternary_operations:
            yield (
                check_raises_re,
                kinds,
                (np.bool_, np.bool_, np.bool_),
                op,
                shape_small,
                op_desc,
                error_msg,
            )


def test_bitwise_disallowed():
    error_msg = "Inputs to bitwise operator `[\\S]*` must be of integral type."
    for kinds in bin_input_kinds:
        for op, op_desc in bitwise_operations:
            for types_in in itertools.product(selected_input_types, selected_input_types):
                if types_in[0] in float_types or types_in[1] in float_types:
                    yield check_raises_re, kinds, types_in, op, shape_small, op_desc, error_msg


def test_prohibit_min_max():
    for kinds in bin_input_kinds:
        for op, op_desc in [(min, "min"), (max, "max")]:
            yield check_raises_te, kinds, (np.int32, np.int32), op, shape_small, op_desc


@raises(
    TypeError,
    glob=(
        '"DataNode" was used in conditional context - it might have been used '
        "in truth evaluation for `if` statement, logical expression or cast to a "
        "boolean. To use conditional execution via `if` statements you need to "
        "specify `enable_conditionals=True` in `@nvidia.dali.pipeline_def` "
        "decorator. You can read more about conditional execution in specific "
        "section of the Pipeline documentation. Bool conversion can be achieved "
        "with the `cast` operator."
    ),
)
def test_bool_raises():
    bool(DataNode("dummy"))


def test_binary_ops_broadcasting():
    def get_sh(arg_idx):
        shapes0 = [(43, 42, 3), (4, 3, 16), (8, 1, 2), (1, 2, 64)]
        shapes1 = [(1, 1, 3), (1, 1, 1), (1, 8, 2), (1, 2, 64)]
        if arg_idx == 0:
            return shapes0
        elif arg_idx == 1:
            return shapes1
        else:
            assert False

    for kinds in list_product(["cpu", "gpu"], ["cpu", "gpu"]):
        for op, op_desc, get_range in sane_operations:
            for types_in in itertools.product(selected_input_types, selected_input_types):
                if types_in != (np.bool_, np.bool_) or op_desc == "*":
                    yield check_arithm_op, kinds, types_in, op, get_sh, get_range, op_desc


def test_ternary_ops_broadcasting():
    def get_sh(arg_idx):
        shapes0 = [(43, 42, 3), (4, 3, 16), (8, 1, 2), (1, 2, 64)]
        shapes1 = [(1, 1, 3), (1, 1, 1), (1, 8, 2), (1, 2, 64)]
        shapes2 = [(43, 1, 3), (4, 1, 16), (8, 1, 2), (1, 1, 1)]
        if arg_idx == 0:
            return shapes0
        elif arg_idx == 1:
            return shapes1
        elif arg_idx == 2:
            return shapes2
        else:
            assert False

    for kinds in ("cpu", "cpu", "cpu"), ("gpu", "gpu", "gpu"):
        for op, op_desc in ternary_operations:
            for types_in in itertools.product(
                selected_input_arithm_types,
                selected_input_arithm_types,
                selected_input_arithm_types,
            ):
                yield check_ternary_op, kinds, types_in, op, get_sh, op_desc


def generate_layout_broadcasting_cases():
    rng = np.random.default_rng(4242)

    def get_input_dev(num_inputs):
        placement = rng.choice(["cpu", "gpu", "non_uniform"])
        if placement != "non_uniform":
            return (placement,) * num_inputs
        placement = [rng.choice(["cpu", "gpu"]) for _ in range(num_inputs - 1)]
        placement.append("gpu" if placement[-1] == "cpu" else "cpu")
        return tuple(placement)

    def get_input_types(num_inputs, integral_only):
        types = (np.int32, np.uint8)
        if not integral_only:
            types += (np.float32,)
        return tuple(rng.choice(types, size=(num_inputs,)))

    # The input layouts and the expected output layout.
    # A number N denotes an ND tensor without a layout.
    # `Exception` means that applying an operator with the arguments with
    # given layouts should raise an error.
    bin_layouts = [
        ((4, "C"), 4),
        (("C", 3), 3),
        (("C", 2), 2),
        (("C", 1), "C"),
        ((1, "C"), "C"),
        ((0, "C"), "C"),
        (("C", 0), "C"),
        (("ABCD", 0), "ABCD"),
        ((0, "ABCD"), "ABCD"),
        (("ABCD", 3), "ABCD"),
        ((1, "ABCD"), "ABCD"),
        (("ABCD", "D"), "ABCD"),
        (("D", "ABCD"), "ABCD"),
        (("ABCD", "CD"), "ABCD"),
        (("ABCD", "BCD"), "ABCD"),
        (("BCD", "ABCD"), "ABCD"),
        (("ABCD", "ABCD"), "ABCD"),
        (("ABCD", "ABC"), Exception()),
        (("X", "ABCD"), Exception()),
    ]

    ternary_layouts = [
        (("ABCD", "CD", "D"), "ABCD"),
        (("ABCD", "D", "CD"), "ABCD"),
        ((3, "ABCD", "CD"), "ABCD"),
        ((0, "ABCD", 0), "ABCD"),
        ((0, "BCD", 4), 4),
        ((3, 4, "CD"), 4),
        ((4, "ABCD", 4), "ABCD"),
        ((4, "A", "B"), Exception()),
    ]

    bin_ops = (
        floaty_operations[:5]
        + bitwise_operations[:3]
        + comparisons_operations[:2]
        + sane_operations
    )

    def tensor_desc(ndim_or_layout):
        if isinstance(ndim_or_layout, int):
            ndim = ndim_or_layout
            layout = None
        else:
            assert isinstance(ndim_or_layout, str)
            ndim = len(ndim_or_layout)
            layout = ndim_or_layout
        max_shape = (5, 7, 11, 13)
        shape = tuple() if ndim == 0 else max_shape[-ndim:]
        return shape, layout

    for num_inputs, layouts, op_lists in [
        (2, bin_layouts, bin_ops),
        (3, ternary_layouts, ternary_operations),
    ]:
        for i, (args_desc, out_desc) in enumerate(layouts):
            assert len(args_desc) == num_inputs
            op, op_name = op_lists[i % len(op_lists)][:2]
            op = op if not isinstance(op, tuple) else op[0]
            input_devs = get_input_dev(num_inputs)
            in_types = get_input_types(num_inputs, op_name in ("&|^"))
            args_desc = tuple(tensor_desc(arg) for arg in args_desc)
            if not isinstance(out_desc, Exception):
                out_desc = tensor_desc(out_desc)
            yield op_name, args_desc, out_desc, input_devs, in_types, op


@params(*tuple(generate_layout_broadcasting_cases()))
def test_layout_broadcasting(op_name, args_desc, out_desc, in_devs, in_types, op):
    assert len(args_desc) == len(in_devs)
    assert len(in_types) == len(in_devs)
    batch_size = 4

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def pipeline():
        in_nodes = [
            types.Constant(np.full(shape, 1, dtype=in_type), device=in_dev, layout=layout)
            for (shape, layout), in_dev, in_type in zip(args_desc, in_devs, in_types)
        ]
        return op(*in_nodes)

    p = pipeline()
    if isinstance(out_desc, Exception):
        with assert_raises(Exception, glob="They must be equal or one must be a suffix"):
            p.run()
    else:
        (o,) = p.run()
        expected_shape, expected_layout = out_desc
        expected_layout = expected_layout or ""
        assert o.layout() == expected_layout, f"got `{o.layout()}`, expected `{expected_layout}`"
        out_shape = o.shape()
        assert len(out_shape) == batch_size, f"got `{len(out_shape)}`, expected `{batch_size}`"
        for sample_shape in out_shape:
            assert (
                sample_shape == expected_shape
            ), f"got `{sample_shape}`, expected `{expected_shape}`"


def test_broadcasting_dimensionality_limits():
    def impl(device, shape_a, shape_b):
        @pipeline_def(batch_size=1, num_threads=3, device_id=0)
        def pipe():
            a = fn.random.uniform(range=[-1, 1], shape=shape_a)
            b = fn.random.uniform(range=[-1, 1], shape=shape_b)
            return a + b

        p = pipe()
        p.run()

    # ERROR
    error_msg = (
        "Broadcasting pattern too complex. Can't operate with simplified"
        + " shapes with more than 6 groups of dimensions. Got 10 groups. "
        + "For more details see https://docs.nvidia.com/deeplearning/dali/user-guide/docs/math.html"
    )
    shape_a_err = (2, 1, 2, 1, 2, 1, 2, 1, 2, 1)
    shape_b_err = (1, 2, 1, 2, 1, 2, 1, 2, 1, 2)
    for device in ["cpu", "gpu"]:
        with assert_raises(RuntimeError, glob=error_msg):
            impl(device, shape_a_err, shape_b_err)

    # NO ERROR (exactly 6 groups)
    shape_a_ok = (2, 1, 1, 1, 3, 1, 4, 5, 6, 1)
    shape_b_ok = (1, 2, 3, 4, 1, 5, 1, 1, 1, 6)
    for device in ["cpu", "gpu"]:
        impl(device, shape_a_ok, shape_b_ok)


def test_broadcasting_incompatible_shapes():
    def impl(device, shape_a, shape_b):
        @pipeline_def(batch_size=1, num_threads=3, device_id=0)
        def pipe():
            a = fn.random.uniform(range=[-1, 1], shape=shape_a)
            b = fn.random.uniform(range=[-1, 1], shape=shape_b)
            return a + b

        p = pipe()
        p.run()

    error_msg1 = (
        "Can't broadcast shapes:*"
        + "2 x 3 x 4 (d=2, belonging to sample_idx=0)\n"
        + "2 x 3 x 3 (d=2, belonging to sample_idx=0)"
    )
    shape_a1 = (2, 3, 4)
    shape_b1 = (2, 3, 3)
    for device in ["cpu", "gpu"]:
        with assert_raises(RuntimeError, glob=error_msg1):
            impl(device, shape_a1, shape_b1)

    error_msg2 = (
        "Can't broadcast shapes:*"
        + "1 x 4 (d=1, belonging to sample_idx=0)\n"
        + "3 (d=0, belonging to sample_idx=0)"
    )
    shape_a2 = (1, 4)
    shape_b2 = 3
    for device in ["cpu", "gpu"]:
        with assert_raises(RuntimeError, glob=error_msg2):
            impl(device, shape_a2, shape_b2)


def test_nested_datanode_error_math():
    @pipeline_def(device_id=None, batch_size=1, num_threads=4)
    def err_pipe():
        u = fn.random.uniform(range=[0, 1])
        v = fn.random.uniform(range=[0, 1])
        return math.max([u, v], 5)

    with assert_raises(
        TypeError, glob="input 0 of operator `max` must be*" "Got a `list` with nested *DataNode"
    ):
        _ = err_pipe()


@params(
    *(
        (x,)
        for x in ("+", "-", "*", "/", "//", "**", "&", "|", "^", "==", "!=", "<", ">", "<=", ">=")
    )
)
def test_nested_datanode_error_arithm(op):
    print(op)

    @pipeline_def(device_id=None, batch_size=1, num_threads=4)
    def err_pipe():
        u = fn.random.uniform(range=[0, 1])  # noqa(F841)
        v = fn.random.uniform(range=[0, 1])  # noqa(F841)
        return eval(f"u {op} [v]")

    with assert_raises(
        TypeError, glob=f"input 1 of operator `{op}` must be*" "Got a `list` with nested *DataNode"
    ):
        _ = err_pipe()


@params(("cpu",), ("gpu",))
def test_empty_batch(device):
    @pipeline_def(
        device_id=0 if device == "gpu" else None,
        batch_size=1,
        num_threads=4,
    )
    def empty_input_pipe():
        x = types.Constant(np.zeros((0, 3), dtype=np.float32), device=device)
        return x * 42

    p = empty_input_pipe()
    (o,) = p.run()
    assert tuple(o[0].shape()) == (0, 3)
