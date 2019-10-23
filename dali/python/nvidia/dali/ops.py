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
from nvidia.dali.types import _type_name_convert_to_string, _type_convert_value, DALIDataType, \
                              CUDAStream
from nvidia.dali.pipeline import Pipeline
from future.utils import with_metaclass
import nvidia.dali.libpython_function_plugin


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

    # def __neg__(self):
    #     return _arithm_op("minus", self)
    # def __pos__(self):
    #     return _arithm_op("plus", self)

_cpu_ops = set({})
_gpu_ops = set({})
_mixed_ops = set({})
_support_ops = set({})

def _docstring_generator(cls):
    op_name = cls.__name__
    op_dev = []
    if op_name in _cpu_ops:
        op_dev.append("'CPU'")
    if op_name in _gpu_ops:
        op_dev.append("'GPU'")
    if op_name in _mixed_ops:
        op_dev.append("'mixed'")
    if op_name in _support_ops:
        op_dev.append("'support'")
    pre_doc = "This is a " + ", ".join(op_dev) + " operator\n\n"

    schema = b.GetSchema(op_name)
    # insert tag to easily link to the operator
    ret = '.. _' + op_name + ':\n\n'
    ret += pre_doc
    ret += schema.Dox()
    ret += '\n'
    if schema.IsSequenceOperator():
        ret += "\nThis operator expects sequence inputs\n"
    elif schema.AllowsSequences():
        ret += "\nThis operator allows sequence inputs\n"

    if schema.IsDeprecated():
        use_instead = schema.DeprecatedInFavorOf()
        ret += "\n.. warning::\n\n   This operator is now deprecated"
        if use_instead:
            ret +=". Use `" + use_instead + "` instead"
        ret += "\n"

    if schema.IsNoPrune():
        ret += "\nThis operator will **not** be optimized out of the graph.\n"

    ret += """
Parameters
----------
"""
    for arg in schema.GetArgumentNames():
        dtype = schema.GetArgumentType(arg)
        arg_name_doc = "`" + arg + "` : "
        ret += (arg_name_doc +
                _type_name_convert_to_string(dtype, schema.IsTensorArgument(arg)))
        if schema.IsArgumentOptional(arg):
            default_value_string = schema.GetArgumentDefaultValueString(arg)
            # Evaluating empty string results in an error
            # so we need to prevent that
            if default_value_string:
                default_value = eval(default_value_string)
            else:
                default_value = default_value_string
            ret += (", optional, default = " +
                    repr(_type_convert_value(dtype, default_value)))
        indent = '\n' + " " * len(arg_name_doc)
        ret += indent
        ret += schema.GetArgumentDox(arg).replace("\n", indent)
        ret += '\n'
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

class _OperatorInstance(object):
    def __init__(self, inputs, op, **kwargs):
        self._counter = _OpCounter()
        self._inputs = inputs
        self._outputs = []
        self._op = op
        self._spec = op.spec.copy()
        self._relation_id = self._counter.id
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
                if not isinstance(kwargs[k], _EdgeReference):
                    raise TypeError(
                            ("Expected inputs of type " +
                            "'_EdgeReference'. Received " +
                            "input of type '{}'.")
                            .format(type(kwargs[k]).__name__))
                self._spec.AddArgumentInput(k, kwargs[k].name)
                self._inputs = list(self._inputs) + [kwargs[k]]

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

                if isinstance(value, list):
                    if not value:
                        raise RuntimeError("List arguments need to have at least 1 element.")
                dtype = self._schema.GetArgumentType(key)
                converted_value = _type_convert_value(dtype, value)
                self._spec.AddArg(key, converted_value)

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
    return Operator

def _load_ops():
    global _cpu_ops
    global _gpu_ops
    global _mixed_ops
    global _support_ops
    _cpu_ops = _cpu_ops.union(set(b.RegisteredCPUOps()))
    _gpu_ops = _gpu_ops.union(set(b.RegisteredGPUOps()))
    _mixed_ops = _mixed_ops.union(set(b.RegisteredMixedOps()))
    _cpu_gpu_ops = _cpu_ops.union(_gpu_ops).union(_mixed_ops)
    _support_ops = _support_ops.union(set(b.RegisteredSupportOps()))
    for op_name in _cpu_gpu_ops:
        if not hasattr(sys.modules[__name__], op_name):
            setattr(sys.modules[__name__], op_name,
                    python_op_factory(op_name, op_device = "cpu"))
    # add support ops
    for op_name in _support_ops:
        if not hasattr(sys.modules[__name__], op_name):
            setattr(sys.modules[__name__], op_name,
                    python_op_factory(op_name, op_device = "support"))
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


class PythonFunction(PythonFunctionBase):
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({'PythonFunction'})

    @staticmethod
    def current_stream():
        """Get DALI's current CUDA stream."""
        return CUDAStream(nvidia.dali.libpython_function_plugin.current_dali_stream())

    def __init__(self, function, num_outputs=1, device='cpu', **kwargs):
        super(PythonFunction, self).__init__(impl_name="PythonFunctionImpl", function=function,
                                             num_outputs=num_outputs, device=device, **kwargs)


class DLTensorPythonFunction(PythonFunctionBase):
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({'DLTensorPythonFunction'})
    global _gpu_ops
    _gpu_ops = _gpu_ops.union({'DLTensorPythonFunction'})

    def __init__(self, function, num_outputs=1, device='cpu', synchronize_stream=True, **kwargs):
        super(DLTensorPythonFunction, self).__init__(impl_name="DLTensorPythonFunctionImpl",
                                                     function=function, num_outputs=num_outputs,
                                                     device=device,
                                                     synchronize_stream=synchronize_stream,
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

def _target_uniform_dev(inputs, dev):
    return inputs


def _arithm_op(name, *inputs):
        input_desc = ""
        for i, input in enumerate(inputs):
            input_desc += "&" + str(i)
            if i < len(inputs) - 1:
                input_desc += " "
        expression_desc = "{}({})".format(name, input_desc)
        dev = _choose_device(inputs)
        # Create "instance" of operator
        op = ArithmeticGenericOp(device = _choose_device(inputs), expression_desc = expression_desc)
        # If we are on gpu, we must mark all inputs as gpu
        if dev == "gpu":
            dev_inputs = list(input.gpu() for input in inputs)
        else:
            dev_inputs = inputs
        # Call it imediatelly
        return op(*dev_inputs)



def cpu_ops():
    return _cpu_ops

def gpu_ops():
    return _gpu_ops

def support_ops():
    return _support_ops

def mixed_ops():
    return _mixed_ops

def register_cpu_op(name):
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({name})
