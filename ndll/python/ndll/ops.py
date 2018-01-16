#pylint: disable=no-member
import sys
import copy
from itertools import count
import ndll.backend as b
from ndll.tensor import TensorReference

class OpCounter(object):
    #pylint: disable=too-few-public-methods
    _op_count = count(0)
    def __init__(self):
        self._id = next(self._op_count)

    @property
    def id(self):
        return self._id

def python_op_factory(name):
    class Operator(object):
        class OperatorInstance(object):
            def __init__(self, inputs, op):
                self._counter = OpCounter()
                self._inputs = inputs
                self._outputs = []
                self._op = op
                self._spec = op.spec.copy()
                # Add inputs
                for inp in inputs:
                    if not isinstance(inp, TensorReference):
                        raise TypeError(
                            """Expected inputs of type
                            TensorReference. Received
                            input type {}"""
                            .format(type(inp).__name__))
                    self._spec.AddInput(inp.name, inp.device)
                # Add outputs
                num_output = op.schema.CalculateOutputs(self._spec)
                for i in range(num_output):
                    t_name = type(op).__name__ + "_id_" + str(self.id) + "_output_" + str(i)
                    t = TensorReference(t_name, op.device, self)
                    self._spec.AddOutput(t.name, t.device)
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
            def spec(self):
                return self._spec

            def append_output(self, output):
                self._outputs.append(output)

        def __init__(self, **kwargs):
            self._spec = b.OpSpec(type(self).__name__)
            self._schema = b.GetSchema(type(self).__name__)

            # Unfortunately class docstrings are immutable, so
            # we append the operator docs to the operator constructor
            try:
                type(self).__init__.__func__.__doc__ = self._schema.Dox()
            except:
                # In Python 3 __func__ attribute does not exist
                type(self).__init__.__doc__ = self._schema.Dox()

            # Get the device argument. We will need this to determine
            # the device that our outputs will be stored on
            if "device" in kwargs.keys():
                self._device = kwargs["device"]
            else:
                self._device = "cpu"

            # Store the specified arguments
            for key, value in kwargs.items():
                self._spec.AddArg(key, value)

        @property
        def spec(self):
            return self._spec

        @property
        def schema(self):
            return self._schema

        @property
        def device(self):
            return self._device

        def __call__(self, *inputs):
            # TODO(tgale): Inputs come in as a list of
            # TensorReferences. Can we also support
            # kwargs based on the docstring?
            if (len(inputs) > self._schema.MaxNumInput() or
                    len(inputs) < self._schema.MinNumInput()):
                raise ValueError(
                    """Operator {} expects [{},
                    {}] inputs, but received {}"""
                    .format(type(self).__name__,
                            self._schema.MinNumInput(),
                            self._schema.MaxNumInput(),
                            len(inputs)))

            op_instance = self.OperatorInstance(inputs, self)

            if len(op_instance.outputs) == 1:
                return op_instance.outputs[0]
            return op_instance.outputs

    Operator.__name__ = str(name)
    return Operator

# TODO(tgale): Do this for all cpu/gpu ops. Figure
# out how we want to expose what devices are supported
_all_ops = set(b.RegisteredCPUOps()).union(set(b.RegisteredGPUOps()))
for op_name in _all_ops:
    setattr(sys.modules[__name__], op_name,
            python_op_factory(op_name))
