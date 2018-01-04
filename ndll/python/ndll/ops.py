from itertools import count
import ndll.backend as b
from ndll.tensor import TensorReference
import sys
import re

class OpCounter(object):
    _op_count = count(0)
    def __init__(self):
        self.id = next(self._op_count)
        
def python_op_factory(name):
    class Operator(object):
        def __init__(self, **kwargs):
            self._counter = OpCounter()
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
        def id(self):
            return self._counter.id
        
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
        def inputs(self):
            return self._inputs

        @property
        def outputs(self):
            return self._outputs
        
        def __call__(self, *inputs):
            # TODO(tgale): Inputs come in as a list of
            # TensorReferences. Can we also support
            # kwargs based on the docstring?
            self._inputs = inputs
            if (len(inputs) > self._schema.MaxNumInput() or
                len(inputs) < self._schema.MinNumInput()):
                raise ValueError(
                    """Operator {} expects [{},
                    {}] inputs, but received {}"""
                    .format(type(self).__name__,
                            self._schema.MinNumInput(),
                            self._schema.MaxNumInput(),
                            len(inputs)))

            for t in inputs:
                if type(t) is not TensorReference:
                    raise TypeError(
                        """Expected inputs of type
                        TensorReference. Received 
                        input type {}"""
                        .format(type(t).__name__))
                self._spec.AddInput(t.name, t.device)

            num_output = self._schema.CalculateOutputs(self._spec)
            self._outputs = []
            for i in range(num_output):
                t_name = type(self).__name__ + "_output_" + str(i)
                t = TensorReference(t_name, self.device, self)
                self._spec.AddOutput(t.name, t.device)
                self._outputs.append(t)

            if num_output == 1:
                return self._outputs[0]
            return self._outputs

    Operator.__name__ = str(name)
    return Operator

# TODO(tgale): Do this for all cpu/gpu ops. Figure
# out how we want to expose what devices are supported
_all_ops = set(b.RegisteredCPUOps()).union(set(b.RegisteredGPUOps()))
for op_name in _all_ops:
    if (b.GetSchema(op_name).HasOutputFn()):
        # Note: We only expose operators for which
        # we can infer the number of outputs from
        # the op spec.
        setattr(sys.modules[__name__], op_name,
                python_op_factory(op_name))
