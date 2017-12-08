import ndll.backend as b
from ndll.tensor import TensorReference
import sys
import re

# TODO(tgale): Do we want device as a kwarg or actually
# in the name of the operator class?
        
# TODO(tgale): Support kwargs in the constructor so that
# users can define all the args to an op on construction
# and not in the call to execute it
def python_op_factory(name):
    class Operator(object):
        def __init__(self, **kwargs):
            # TODO(tgale): Can these be moved into class
            # wide member variables or will they be constant
            # for all the versions we create?
            self._spec = b.OpSpec(type(self).__name__)
            self._schema = b.GetSchema(type(self).__name__)
            self._device = "find me in kwargs!"

            # Unfortunately class docstrings are immutable, so
            # we append the operator docs to the operator constructor
            type(self).__init__.__func__.__doc__ = self._schema.Dox()
            
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

            for tensor in inputs:
                if type(tensor) is not TensorReference:
                    raise TypeError(
                        """Expected inputs of type
                        TensorReference. Received 
                        input type {}"""
                        .format(type(tensor).__name__))
                self._spec.AddInput(tensor.name, tensor.device)

            num_output = self._schema.CalculateOutputs(self._spec)
            outputs = []
            for i in range(num_output):
                outputs.append(TensorReference(
                    type(self).__name__ + "_output_" + str(i),
                    self.device, self))

            if num_output == 1:
                return outputs[0]
            return outputs

    Operator.__name__ = str(name)
    return Operator

# TODO(tgale): Do this for all cpu/gpu ops. Figure
# out how we want to expose what devices are supported
for op_name in b.RegisteredCPUOps():
    if (b.GetSchema(op_name).HasOutputFn()):
        # Note: We only expose operators for which
        # we can infer the number of outputs from
        # the op spec.
        setattr(sys.modules[__name__], op_name,
                python_op_factory(op_name))
