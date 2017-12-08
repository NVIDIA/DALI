import ndll.backend as b
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
            self._spec = b.OpSpec(type(self).__name__)

            # Store the specified arguments
            for key, value in kwargs.items():
                self._spec.AddArg(key, value)
                
        @property
        def spec(self):
            return self._spec

        def __call__(self, *inputs):
            for tensor in inputs:
                if type(tensor) is not TensorReference:
                    raise TypeError(
                        """Expected inputs of type
                        TensorReference. Received 
                        input type {}"""
                        .format(type(tensor).__name__))
                self._spec.AddInput(tensor.name, tensor.device)
            
                
            # TODO(tgale): Inputs come in as a list of
            # TensorReferences. Can we also support
            # kwargs based on the docstring?

            # TODO(tgale): How do we handle the case
            # where an op can have a variable number
            # of outputs? How do we know how many
            # tensors to return? Should we support
            # this? If we do, can we do it as part
            # of the opdoc?
            #
            # We can have users define a lambda that
            # returns the number of outputs and the
            # device of those outputs as a function
            # of the input opspec
            #
            # (device is implied by the device argument
            # in the opspec, if we were to ever allow
            # other types of outputs we could not rely
            # on this though)
            #
            # [] (const OpSpec &spec) {
            #  return 1;
            # }
            pass

    # TODO(tgale): Add docstring from backend registry
    # to the class definition.
    Operator.__name__ = str(name)
    return Operator

for op_name in b.RegisteredCPUOps():
    setattr(sys.modules[__name__], op_name,
            python_op_factory(op_name))
