# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#pylint: disable=no-member
from __future__ import division
import sys
import copy
from itertools import count
import threading
from nvidia.dali import backend as b
from nvidia.dali.types import _type_name_convert_to_string, _type_convert_value, \
        _vector_element_type, _bool_types, _int_types, _int_like_types, _float_types, \
        DALIDataType, CUDAStream, ScalarConstant as _ScalarConstant, Constant as _Constant
from nvidia.dali.pipeline import Pipeline
from future.utils import with_metaclass
import nvidia.dali.libpython_function_plugin


cupy = None
def _setup_cupy():
    global cupy
    if cupy is None:
        import cupy as cupy


class _EdgeReference(object):
    def __init__(self, name, device="cpu", source=None):
        self.name = name
        self.device = device
        self.source = source

    # Note: Regardless of whether we want the cpu or gpu version
    # of a tensor, we keep the source argument the same so that
    # the pipeline can backtrack through the user-defined graph
    def gpu(self):
        return _EdgeReference(self.name, "gpu", self.source)

    def __add__(self, other):
        return _arithm_op("add", self, other)
    def __radd__(self, other):
        return _arithm_op("add", other, self)

    def __sub__(self, other):
        return _arithm_op("sub", self, other)
    def __rsub__(self, other):
        return _arithm_op("sub", other, self)

    def __mul__(self, other):
        return _arithm_op("mul", self, other)
    def __rmul__(self, other):
        return _arithm_op("mul", other, self)

    def __truediv__(self, other):
        return _arithm_op("fdiv", self, other)
    def __rtruediv__(self, other):
        return _arithm_op("fdiv", other, self)

    def __floordiv__(self, other):
        return _arithm_op("div", self, other)
    def __rfloordiv__(self, other):
        return _arithm_op("div", other, self)

    def __neg__(self):
        return _arithm_op("minus", self)

    # Shortucitng the execution, unary + is basically a no-op
    def __pos__(self):
        return self

    def __eq__(self, other):
        return _arithm_op("eq", self, other)

    def __ne__(self, other):
        return _arithm_op("neq", self, other)

    def __lt__(self, other):
        return _arithm_op("lt", self, other)

    def __le__(self, other):
        return _arithm_op("leq", self, other)

    def __gt__(self, other):
        return _arithm_op("gt", self, other)

    def __ge__(self, other):
        return _arithm_op("geq", self, other)

    def __and__(self, other):
        return _arithm_op("bitand", self, other)
    def __rand__(self, other):
        return _arithm_op("bitand", other, self)

    def __or__(self, other):
        return _arithm_op("bitor", self, other)
    def __ror__(self, other):
        return _arithm_op("bitor", other, self)

    def __xor__(self, other):
        return _arithm_op("bitxor", self, other)
    def __rxor__(self, other):
        return _arithm_op("bitxor", other, self)

_cpu_ops = set({})
_gpu_ops = set({})
_mixed_ops = set({})

def _numpydoc_formatter(name, type, doc, optional = False):
    indent = "\n" + " " * 4
    if optional:
        type += ", optional"
    return "`{}` : {}{}{}".format(name, type, indent, doc.replace("\n", indent))

def _get_kwargs(schema, only_tensor = False):
    """
    Get the keywords arguments from the schema.

    `schema`
        the schema in which to lookup arguments
    `only_tensor`: bool
        If True list only keyword arguments that can be passed as TensorLists (argument inputs)
        If False list all the arguments. False indicates that we list arguments to the
        constructor of the operator which does not accept TensorLists (argument inputs) - that
        fact will be reflected in specified type
    """
    ret = ""
    for arg in schema.GetArgumentNames():
        if not only_tensor or schema.IsTensorArgument(arg):
            arg_name_doc = arg
            dtype = schema.GetArgumentType(arg)
            type_name = _type_name_convert_to_string(dtype, is_tensor = only_tensor)
            if schema.IsArgumentOptional(arg):
                default_value_string = schema.GetArgumentDefaultValueString(arg)
                # Evaluating empty string results in an error
                # so we need to prevent that
                if default_value_string:
                    default_value = eval(default_value_string)
                else:
                    default_value = default_value_string
                type_name += (", optional, default = " +
                        repr(_type_convert_value(dtype, default_value)))
            doc = schema.GetArgumentDox(arg)
            ret += _numpydoc_formatter(arg, type_name, doc)
            ret += '\n'
    return ret

def _docstring_generator(cls):
    """
        Generate docstring for the class obtaining it from schema based on cls.__name__

        This lists all the Keyword args that can be used when creating operator
    """
    op_name = cls.__name__
    schema = b.GetSchema(op_name)
    ret = '\n'

    if schema.IsDeprecated():
        use_instead = schema.DeprecatedInFavorOf()
        ret += ".. warning::\n\n   This operator is now deprecated"
        if use_instead:
            ret +=". Use `" + use_instead + "` instead."
        ret += "\n\n"

    ret += schema.Dox()
    ret += '\n'

    if schema.IsSequenceOperator():
        ret += "\nThis operator expects sequence inputs.\n"
    elif schema.AllowsSequences():
        ret += "\nThis operator allows sequence inputs.\n"

    if schema.SupportsVolumetric():
        ret += "\nThis operator supports volumetric data.\n"

    if schema.IsNoPrune():
        ret += "\nThis operator will **not** be optimized out of the graph.\n"

    op_dev = []
    if op_name in _cpu_ops:
        op_dev.append("'cpu'")
    if op_name in _gpu_ops:
        op_dev.append("'gpu'")
    if op_name in _mixed_ops:
        op_dev.append("'mixed'")
    ret += """
Supported backends
"""
    for dev in op_dev:
        ret += " * " + dev + "\n"
    ret += "\n"

    ret += """
Keyword args
------------
"""
    ret += _get_kwargs(schema)
    return ret

def _docstring_prefix_from_inputs(op_name):
    """
        Generate start of the docstring for `__call__` of Operator `op_name`
        assuming the docstrings were provided for all inputs separatelly

        Returns the signature of `__call__` and list of `Args` in appropriate section
    """
    schema = b.GetSchema(op_name)
    # Signature
    ret = "__call__(" + schema.GetCallSignatureInputs() + ", **kwargs)\n"
    # __call__ docstring
    ret += "\nOperator call to be used in `define_graph` step.\n"
    # Args section
    ret += """
Args
----
"""
    for i in range(schema.MaxNumInput()):
        optional = i >= schema.MinNumInput()
        ret += _numpydoc_formatter(schema.GetInputName(i), schema.GetInputType(i), schema.GetInputDox(i), optional)
        ret += "\n"
    ret += "\n"
    return ret

def _docstring_prefix_auto(op_name):
    """
        Generate start of the docstring for `__call__` of Operator `op_name`
        with default values. Assumes there will be 0 or 1 inputs
    """
    schema = b.GetSchema(op_name)
    if schema.MaxNumInput() == 0:
        return """__call__(**kwargs)

Operator call to be used in `define_graph` step. This operator does not accept any TensorList inputs.
"""
    elif schema.MaxNumInput() == 1:
        return """__call__(data, **kwargs)

Operator call to be used in `define_graph` step.

Args
----
`data`: TensorList
    Input to the operator.
"""
    return ""


def _docstring_generator_call(op_name):
    """
        Generate full docstring for `__call__` of Operator `op_name`.
    """
    schema = b.GetSchema(op_name)
    if schema.HasCallDox():
        ret = schema.GetCallDox()
    elif schema.HasInputDox():
        ret =_docstring_prefix_from_inputs(op_name)
    elif schema.CanUseAutoInputDox():
        ret = _docstring_prefix_auto(op_name)
    else:
        ret = "Please refer to class :meth:`nvidia.dali.ops." + op_name + "` for full documentation.\n"
    if schema.AppendKwargsSection():
        # Kwargs section
        tensor_kwargs = _get_kwargs(schema, only_tensor = True)
        if tensor_kwargs:
            ret += """
Keyword Args
------------
"""
            ret += tensor_kwargs
    return ret

class _OpCounter(object):
    #pylint: disable=too-few-public-methods
    _lock = threading.Lock()
    _op_count = count(0)
    def __init__(self):
        with self._lock:
            self._id = next(self._op_count)

    @property
    def id(self):
        return self._id

def _instantiate_constant_node(device, constant):
    return _Constant(device = device, value = constant.value, dtype = constant.dtype)

class _OperatorInstance(object):
    def __init__(self, inputs, op, **kwargs):
        self._counter = _OpCounter()
        self._outputs = []
        self._op = op
        self._spec = op.spec.copy()
        self._relation_id = self._counter.id

        default_input_device = "gpu" if op.device == "gpu" else "cpu"
        if inputs is not None:
            inputs = list(inputs)
            for i in range(len(inputs)):
                inp = inputs[i]
                if isinstance(inp, _ScalarConstant):
                    inputs[i] = _instantiate_constant_node(default_input_device, inp)
            inputs = tuple(inputs)

        self._inputs = inputs

        if "name" in kwargs.keys():
            self._name = kwargs["name"]
        else:
            self._name = '__' + type(op).__name__ + "_" + str(self._counter.id)
        # Add inputs
        if inputs:
            for inp in inputs:
                if not isinstance(inp, _EdgeReference):
                    raise TypeError(
                        "Expected inputs of type '_EdgeReference'. Received input of type '{}'."
                        .format(type(inp).__name__))
                self._spec.AddInput(inp.name, inp.device)
        # Argument inputs
        for k in sorted(kwargs.keys()):
            if k not in ["name"]:
                arg_inp = kwargs[k]
                if arg_inp is None:
                    continue
                if isinstance(arg_inp, _ScalarConstant):
                    arg_inp = _instantiate_constant_node("cpu", arg_inp)
                if not isinstance(arg_inp, _EdgeReference):
                    raise TypeError(
                            ("Expected inputs of type " +
                            "'_EdgeReference'. Received " +
                            "input of type '{}'.")
                            .format(type(arg_inp).__name__))
                self._spec.AddArgumentInput(k, arg_inp.name)
                self._inputs = list(self._inputs) + [arg_inp]

        if self._op.schema.IsDeprecated():
            use_instead = self._op.schema.DeprecatedInFavorOf()
            msg = "WARNING: `{}` is now deprecated".format(type(self._op).__name__)
            if use_instead:
                msg +=". Use `" + use_instead + "` instead"
            print(msg)

    def check_args(self):
        self._op.schema.CheckArgs(self._spec)

    def generate_outputs(self):
        pipeline = Pipeline.current()
        # Add outputs
        if self._op.device == "gpu" or self._op.device == "mixed":
            output_device = "gpu"
        else:
            output_device = "cpu"

        num_output = self._op.schema.CalculateOutputs(self._spec) + self._op.schema.CalculateAdditionalOutputs(self._spec)

        if num_output == 0 and self._op.preserve:
            t_name = type(self._op).__name__ + "_id_" + str(self.id) + "_sink"
            pipeline.add_sink(_EdgeReference(t_name, output_device, self))
            return

        for i in range(num_output):
            t_name = type(self._op).__name__ + "_id_" + str(self.id) + "_output_" + str(i)
            t = _EdgeReference(t_name, output_device, self)
            self._spec.AddOutput(t.name, t.device)
            if self._op.preserve:
                pipeline.add_sink(t)
            self.append_output(t)

    @property
    def id(self):
        return self._counter.id

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def unwrapped_outputs(self):
        if len(self._outputs) == 1:
            return self._outputs[0]
        else:
            return self._outputs

    @property
    def spec(self):
        return self._spec

    @property
    def name(self):
        return self._name

    @property
    def relation_id(self):
        return self._relation_id

    @relation_id.setter
    def relation_id(self, value):
        self._relation_id = value

    def append_output(self, output):
        self._outputs.append(output)

class _DaliOperatorMeta(type):
    @property
    def __doc__(self):
        return _docstring_generator(self)

    def __new__(mcs, name, bases, attrs, **kwargs):
        """
            This is just a workaround for Python2.
            In Python3 it just works, when we do Operator.__call__.__doc__ = ...
            after creating it in python_op_factory().

            Here we intercept the creation of class by overloading the __new__ of
            the metaclass, and access the `__call__` atribute in `attrs`.
            We must pass the operator name using a workaround through attributes as there seems
            to be no way of passing kwargs in Python2 using six.with_metaclass.
            TODO(klecki): remove when Python2 is dropped
        """
        # Get the operator name and remove it from attributes
        # In some cases we use the direct name
        try:
            actual_operator_name = attrs['_name']
            del attrs['_name']
        except KeyError:
            actual_operator_name = name
        # Set the docstring for __call__, if it's present
        try:
            attrs['__call__'].__doc__ = _docstring_generator_call(actual_operator_name)
        except KeyError:
            pass
        op_instance = super(_DaliOperatorMeta, mcs).__new__(mcs, name, bases, attrs)
        return op_instance

def python_op_factory(name, op_device = "cpu"):
    class Operator(with_metaclass(_DaliOperatorMeta, object)):
        def __init__(self, **kwargs):
            self._spec = b.OpSpec(type(self).__name__)
            self._schema = b.GetSchema(type(self).__name__)

            # Get the device argument. We will need this to determine
            # the device that our outputs will be stored on
            if "device" in kwargs.keys():
                self._device = kwargs["device"]
                del kwargs["device"]
            else:
                self._device = op_device
            self._spec.AddArg("device", self._device)

            if "preserve" in kwargs.keys():
                self._preserve = kwargs["preserve"]
            else:
                self._preserve = False
            self._spec.AddArg("preserve", self._preserve)
            self._preserve = self._preserve or self._schema.IsNoPrune()

            # Store the specified arguments
            for key, value in kwargs.items():
                if value is None:
                  # None is not a valid value for any argument type, so treat it
                  # as if the argument was not supplied at all
                  continue

                dtype = self._schema.GetArgumentType(key)
                if isinstance(value, (list, tuple)):
                    if len(value) == 0:
                        self._spec.AddArgEmptyList(key, _vector_element_type(dtype))
                        continue
                converted_value = _type_convert_value(dtype, value)
                self._spec.AddArg(key, converted_value)

        # TODO(klecki): remove when Python2 is dropped
        _name = name

        @property
        def spec(self):
            return self._spec

        @property
        def schema(self):
            return self._schema

        @property
        def device(self):
            return self._device

        @property
        def preserve(self):
            return self._preserve

        def __call__(self, *inputs, **kwargs):
            if (len(inputs) > self._schema.MaxNumInput() or
                    len(inputs) < self._schema.MinNumInput()):
                raise ValueError(
                    ("Operator {} expects [{}, " +
                    "{}] inputs, but received {}")
                    .format(type(self).__name__,
                            self._schema.MinNumInput(),
                            self._schema.MaxNumInput(),
                            len(inputs)))
            # Build input sets, most of the time we only have one
            input_sets = []
            if self._detect_multiple_input_sets(inputs):
                arg_list_len = self._check_common_length(inputs)
                packed_inputs = self._unify_lists(inputs, arg_list_len)
                input_sets = self._repack_input_sets(packed_inputs)
            else:
                input_sets = [inputs]

            # Create OperatorInstance for every input set
            op_instances = []
            for input_set in input_sets:
                op_instances.append(_OperatorInstance(input_set, self, **kwargs))
                op_instances[-1].generate_outputs()

            # Tie the instances together
            relation_id = op_instances[0].id
            for op in op_instances:
                op.relation_id = relation_id

            # If we don't have multiple input sets, flatten the result
            if len(op_instances) == 1:
                return op_instances[0].unwrapped_outputs
            outputs = []
            for op in op_instances:
                outputs.append(op.outputs)
            return self._repack_output_sets(outputs)

        # Check if any of inputs is a list
        def _detect_multiple_input_sets(self, inputs):
            return any(isinstance(input, list) for input in inputs)

        # Check if all list representing multiple input sets have the same length and return it
        def _check_common_length(self, inputs):
            arg_list_len = max(self._safe_len(input) for input in inputs)
            for input in inputs:
                if isinstance(input, list):
                    if len(input) != arg_list_len:
                        raise ValueError(("All argument lists for Multpile Input Sets used " +
                                          "with operator {} must have the same length")
                                          .format(type(self).__name__))
            return arg_list_len

        def _safe_len(self, input):
            if isinstance(input, _EdgeReference):
                return 1
            else:
                return len(input)

        # Pack single _EdgeReferences into lists, so they are treated as Multiple Input Sets
        # consistently with the ones already present
        def _unify_lists(self, inputs, arg_list_len):
            result = ()
            for input in inputs:
                if isinstance(input, list):
                    result = result + (input,)
                else:
                    result = result + ([input] * arg_list_len,)
            return result

        # Zip the list from [[arg0, arg0', arg0''], [arg1', arg1'', arg1''], ...]
        # to [(arg0, arg1, ...), (arg0', arg1', ...), (arg0'', arg1'', ...)]
        def _repack_input_sets(self, inputs):
            return self._repack_list(inputs, tuple)

        # Unzip the list from [[out0, out1, out2], [out0', out1', out2'], ...]
        # to [[out0, out0', ...], [out1, out1', ...], [out2, out2', ...]]
        # Assume that all elements of input have the same length
        # If the inputs were 1-elem lists, return just a list, that is:
        # [[out0], [out0'], [out0''], ...] -> [out0, out0', out0'', ...]
        def _repack_output_sets(self, outputs):
            if len(outputs) > 1 and len(outputs[0]) == 1:
                output = []
                for elem in outputs:
                    output.append(elem[0])
                return output
            return self._repack_list(outputs, list)

        # Repack list from [[a, b, c], [a', b', c'], ....]
        # to [fn(a, a', ...), fn(b, b', ...), fn(c, c', ...)]
        # where fn can be `tuple` or `list`
        # Assume that all elements of input have the same length
        def _repack_list(self, sets, fn):
            output_list = []
            arg_list_len = len(sets[0])
            for i in range(arg_list_len):
                output_list.append(fn(input_set[i] for input_set in sets))
            return output_list


    Operator.__name__ = str(name)
    # The autodoc doesn't generate doc for something that doesn't match the module name
    if b.GetSchema(Operator.__name__).IsInternal():
        Operator.__module__ = Operator.__module__ + ".internal"

    # TODO(klecki): use this instead of __new__ in metaclass when Python2 is dropped
    # Operator.__call__.__doc__ = _docstring_generator_call(name)
    return Operator

def _load_ops():
    global _cpu_ops
    global _gpu_ops
    global _mixed_ops
    _cpu_ops = _cpu_ops.union(set(b.RegisteredCPUOps()))
    _gpu_ops = _gpu_ops.union(set(b.RegisteredGPUOps()))
    _mixed_ops = _mixed_ops.union(set(b.RegisteredMixedOps()))
    _cpu_gpu_ops = _cpu_ops.union(_gpu_ops).union(_mixed_ops)
    for op_name in _cpu_gpu_ops:
        if not hasattr(sys.modules[__name__], op_name):
            setattr(sys.modules[__name__], op_name,
                    python_op_factory(op_name, op_device = "cpu"))
_load_ops()

def Reload():
    _load_ops()

# custom wrappers around ops
class TFRecordReader(with_metaclass(_DaliOperatorMeta, object)):
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({'TFRecordReader'})

    def __init__(self, path, index_path, features, **kwargs):
        if isinstance(path, list):
            self._path = path
        else:
            self._path = [path]
        if isinstance(index_path, list):
            self._index_path = index_path
        else:
            self._index_path = [index_path]
        self._schema = b.GetSchema("_TFRecordReader")
        self._spec = b.OpSpec("_TFRecordReader")
        self._device = "cpu"

        self._spec.AddArg("path", self._path)
        self._spec.AddArg("index_path", self._index_path)

        for key, value in kwargs.items():
            self._spec.AddArg(key, value)

        self._features = features

    @property
    def spec(self):
        return self._spec

    @property
    def schema(self):
        return self._schema

    @property
    def device(self):
        return self._device

    def __call__(self, *inputs, **kwargs):
        # We do not handle multiple input sets for Reader as they do not have inputs
        if (len(inputs) > self._schema.MaxNumInput() or
                len(inputs) < self._schema.MinNumInput()):
            raise ValueError(
                ("Operator {} expects [{}, " +
                "{}] inputs, but received {}")
                .format(type(self).__name__,
                        self._schema.MinNumInput(),
                        self._schema.MaxNumInput(),
                        len(inputs)))

        op_instance = _OperatorInstance(inputs, self, **kwargs)
        outputs = {}
        feature_names = []
        features = []
        for i, (feature_name, feature) in enumerate(self._features.items()):
            t_name = "_TFRecordReader" + "_id_" + str(op_instance.id) + "_output_" + str(i)
            t = _EdgeReference(t_name, self._device, op_instance)
            op_instance.spec.AddOutput(t.name, t.device)
            op_instance.append_output(t)
            outputs[feature_name] = t
            feature_names.append(feature_name)
            features.append(feature)

        op_instance.spec.AddArg("feature_names", feature_names)
        op_instance.spec.AddArg("features", features)
        return outputs


class PythonFunctionBase(with_metaclass(_DaliOperatorMeta, object)):
    def __init__(self, impl_name, function, num_outputs=1, device='cpu', **kwargs):
        self._schema = b.GetSchema(impl_name)
        self._spec = b.OpSpec(impl_name)
        self._device = device
        self._impl_name = impl_name

        for key, value in kwargs.items():
            self._spec.AddArg(key, value)

        self.function = function
        self.num_outputs = num_outputs
        self._preserve = True

    @property
    def spec(self):
        return self._spec

    @property
    def schema(self):
        return self._schema

    @property
    def device(self):
        return self._device

    @property
    def preserve(self):
        return self._preserve

    def __call__(self, *inputs, **kwargs):
        pipeline = Pipeline.current()
        if pipeline.exec_async or pipeline.exec_pipelined:
            raise RuntimeError("PythonFunction can be used only in pipelines with `exec_async` and "
                               "`exec_pipelined` set to False.")
        if (len(inputs) > self._schema.MaxNumInput() or
                len(inputs) < self._schema.MinNumInput()):
            raise ValueError(
                ("Operator {} expects [{}, " +
                 "{}] inputs, but received {}")
                .format(type(self).__name__,
                        self._schema.MinNumInput(),
                        self._schema.MaxNumInput(),
                        len(inputs)))
        for inp in inputs:
            if not isinstance(inp, _EdgeReference):
                raise TypeError(
                      ("Expected inputs of type '_EdgeReference'. Received input of type '{}'. " +
                       "Python Operators do not support Multiple Input Sets.")
                      .format(type(inp).__name__))
        op_instance = _OperatorInstance(inputs, self, **kwargs)
        op_instance.spec.AddArg("function_id", id(self.function))
        op_instance.spec.AddArg("num_outputs", self.num_outputs)
        op_instance.spec.AddArg("device", self.device)
        if self.num_outputs == 0:
            t_name = self._impl_name + "_id_" + str(op_instance.id) + "_sink"
            t = _EdgeReference(t_name, self._device, op_instance)
            pipeline.add_sink(t)
            return
        outputs = []
        for i in range(self.num_outputs):
            t_name = self._impl_name + "_id_" + str(op_instance.id) + "_output_" + str(i)
            t = _EdgeReference(t_name, self._device, op_instance)
            op_instance.spec.AddOutput(t.name, t.device)
            op_instance.append_output(t)
            pipeline.add_sink(t)
            outputs.append(t)
        return outputs[0] if len(outputs) == 1 else outputs


def _dlpack_to_array(dlpack):
    return nvidia.dali.libpython_function_plugin.DLTensorToArray(dlpack)


def _dlpack_from_array(array):
    return nvidia.dali.libpython_function_plugin.ArrayToDLTensor(array)


class PythonFunction(PythonFunctionBase):
    global _cpu_ops
    global _gpu_ops
    _cpu_ops = _cpu_ops.union({'PythonFunction'})
    _gpu_ops = _gpu_ops.union({'PythonFunction'})

    @staticmethod
    def current_stream():
        """Get DALI's current CUDA stream."""
        return CUDAStream(nvidia.dali.libpython_function_plugin.current_dali_stream())

    @staticmethod
    def function_wrapper_per_sample(function, from_dlpack, to_dlpack, *dlpack_inputs):
        arrays = [from_dlpack(dlpack) for dlpack in dlpack_inputs]
        arr_out = function(*arrays)
        if arr_out is None:
            return
        if isinstance(arr_out, tuple) or isinstance(arr_out, list):
            return tuple(map(lambda t: to_dlpack(t), arr_out))
        else:
            return to_dlpack(arr_out)

    @staticmethod
    def function_wrapper_batch(function, from_dlpack, to_dlpack, *dlpack_inputs):
        arrays = [[from_dlpack(dlpack) for dlpack in dl_input] for dl_input in dlpack_inputs]
        arr_outs = function(*arrays)
        if arr_outs is None:
            return
        if isinstance(arr_outs, tuple) or isinstance(arr_outs, list):
            return tuple(map(lambda l: [to_dlpack(out) for out in l], arr_outs))
        else:
            return [to_dlpack(out) for out in arr_outs]

    @staticmethod
    def _function_wrapper_cpu(batch_processing, function, *dlpack_inputs):
        if batch_processing:
            return PythonFunction.function_wrapper_batch(function, _dlpack_to_array,
                                                         _dlpack_from_array, *dlpack_inputs)
        else:
            return PythonFunction.function_wrapper_per_sample(function, _dlpack_to_array,
                                                              _dlpack_from_array,
                                                              *dlpack_inputs)

    @staticmethod
    def _cupy_stream_wrapper(function, *inputs):
        stream = cupy.cuda.Stream(null=True)
        stream.ptr = PythonFunction.current_stream().ptr
        with stream:
            out = function(*inputs)
        stream.ptr = 0
        return out

    @staticmethod
    def _function_wrapper_gpu(batch_processing, function, *dlpack_inputs):
        def wrapped_func(*inputs):
            return PythonFunction._cupy_stream_wrapper(function, *inputs)
        if batch_processing:
            return PythonFunction.function_wrapper_batch(wrapped_func, cupy.fromDlpack,
                                                         lambda t: t.toDlpack(), *dlpack_inputs)
        else:
            return PythonFunction.function_wrapper_per_sample(wrapped_func, cupy.fromDlpack,
                                                              lambda t: t.toDlpack(),
                                                              *dlpack_inputs)

    def __init__(self, function, num_outputs=1, device='cpu', batch_processing=False, **kwargs):
        if device == 'gpu':
            _setup_cupy()
        func = (lambda *ts: PythonFunction._function_wrapper_cpu(batch_processing, function, *ts))\
               if device == 'cpu' else \
               (lambda *ts: PythonFunction._function_wrapper_gpu(batch_processing, function, *ts))
        super(PythonFunction, self).__init__(impl_name="DLTensorPythonFunctionImpl",
                                             function=func,
                                             num_outputs=num_outputs, device=device,
                                             synchronize_stream=False,
                                             batch_processing=batch_processing, **kwargs)


class DLTensorPythonFunction(PythonFunctionBase):
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({'DLTensorPythonFunction'})
    global _gpu_ops
    _gpu_ops = _gpu_ops.union({'DLTensorPythonFunction'})

    def __init__(self, function, num_outputs=1, device='cpu', synchronize_stream=True,
                 batch_processing=True, **kwargs):
        super(DLTensorPythonFunction, self).__init__(impl_name="DLTensorPythonFunctionImpl",
                                                     function=function, num_outputs=num_outputs,
                                                     device=device,
                                                     synchronize_stream=synchronize_stream,
                                                     batch_processing=batch_processing,
                                                     **kwargs)

def _load_arithm_ops():
    arithm_op_names = ["ArithmeticGenericOp"]
    for op_name in arithm_op_names:
      if not hasattr(sys.modules[__name__], op_name):
          setattr(sys.modules[__name__], op_name,
                  python_op_factory(op_name, op_device = "cpu"))

_load_arithm_ops()

def _choose_device(inputs):
    if any (input.device == "gpu" for input in inputs):
        return "gpu"
    return "cpu"

def _is_boolean_like(input):
    if type(input) == bool:
        return True
    if isinstance(input, _ScalarConstant):
        if input.dtype in _bool_types:
            return True
    return False

# Boolean and integer types are considered integer-like
def _is_integer_like(input):
    if _is_boolean_like(input):
        return True
    if type(input) == int:
        return True
    if isinstance(input, _ScalarConstant):
        if input.dtype in _int_like_types:
            return True
    return False

def _is_real_like(input):
    if type(input) == float:
        return True
    if isinstance(input, _ScalarConstant):
        if input.dtype in _float_types:
            return True
    return False

# <type> description required by ArithmeticGenericOp
def _to_type_desc(input):
    if type(input) == bool:
        return "bool"
    if type(input) == int:
        return "int32"
    if type(input) == float:
        return "float32" # TODO(klecki): current DALI limitation
    if isinstance(input, _ScalarConstant):
        dtype_to_desc = {
            DALIDataType.BOOL:    "bool",
            DALIDataType.INT8:    "int8",
            DALIDataType.INT16:   "int16",
            DALIDataType.INT32:   "int32",
            DALIDataType.INT64:   "int64",
            DALIDataType.UINT8:   "uint8",
            DALIDataType.UINT16:  "uint16",
            DALIDataType.UINT32:  "uint32",
            DALIDataType.UINT64:  "uint64",
            DALIDataType.FLOAT16: "float16",
            DALIDataType.FLOAT:   "float32",
            DALIDataType.FLOAT64: "float64",
        }
        return dtype_to_desc[input.dtype]
    raise TypeError(("Constant argument to arithmetic operation not supported. Got {}, expected "
            "a constant value of type 'bool', 'int', 'float' or 'nvidia.dali.types.Constant'.")
            .format(str(type(input))))


# Group inputs into categories_idxs, _EdgeReferences, integer constants and real constants
# The categories_idxs is a list that for an input `i` contains a tuple:
# (category of ith input, index of ith input in appropriate category)
def _group_inputs(inputs):
    categories_idxs = []
    edges = []
    integers = []
    reals = []
    for input in inputs:
        if isinstance(input, _EdgeReference):
            categories_idxs.append(("edge", len(edges)))
            edges.append(input)
        elif _is_integer_like(input):
            categories_idxs.append(("integer", len(integers)))
            integers.append(input)
        elif _is_real_like(input):
            categories_idxs.append(("real", len(reals)))
            reals.append(input)
        else:
            raise TypeError(("Argument to arithmetic operation not supported. Got {}, expected "
                    "a return value from other DALI Operator  or a constant value of type 'bool', 'int', "
                    "'float' or 'nvidia.dali.types.Constant'.").format(str(type(input))))
    if len(integers) == 0:
        integers = None
    if len(reals) == 0:
        reals = None
    return (categories_idxs, edges, integers, reals)

# Generate the list of <input> subexpression as specified
# by grammar for ArithmeticGenericOp
def _generate_input_desc(categories_idx, integers, reals):
    input_desc = ""
    for i, (category, idx) in enumerate(categories_idx):
        if category == "edge":
            input_desc += "&{}".format(idx)
        elif category == "integer":
            input_desc += "${}:{}".format(idx, _to_type_desc(integers[idx]))
        elif category == "real":
            input_desc += "${}:{}".format(idx, _to_type_desc(reals[idx]))
        if i < len(categories_idx) - 1:
            input_desc += " "
    return input_desc

# Create arguments for ArithmeticGenericOp and call it with supplied inputs.
# Select the `gpu` device if at least one of the inputs is `gpu`, otherwise `cpu`.
def _arithm_op(name, *inputs):
    categories_idxs, edges, integers, reals = _group_inputs(inputs)
    input_desc = _generate_input_desc(categories_idxs, integers, reals)
    expression_desc = "{}({})".format(name, input_desc)
    dev = _choose_device(edges)
    # Create "instance" of operator
    op = ArithmeticGenericOp(device = dev, expression_desc = expression_desc,
                             integer_constants = integers, real_constants = reals)
    # If we are on gpu, we must mark all inputs as gpu
    if dev == "gpu":
        dev_inputs = list(edge.gpu() for edge in edges)
    else:
        dev_inputs = edges
    # Call it immediately
    return op(*dev_inputs)

def cpu_ops():
    return _cpu_ops

def gpu_ops():
    return _gpu_ops

def mixed_ops():
    return _mixed_ops


def register_cpu_op(name):
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({name})


def register_gpu_op(name):
    global _gpu_ops
    _gpu_ops = _gpu_ops.union({name})
