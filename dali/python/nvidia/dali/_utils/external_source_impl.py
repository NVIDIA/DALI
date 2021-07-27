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

from enum import Enum
from nvidia.dali import types
from nvidia.dali import tensors

import inspect
import functools

np = None

def import_numpy():
    """Import numpy lazily, need to define global `np = None` variable"""
    global np
    if np is None:
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError('Could not import numpy. Please make sure you have numpy '
                               'installed before you use parallel mode.')


class SourceKind(Enum):
    CALLABLE       = 0
    ITERABLE       = 1
    GENERATOR_FUNC = 2

class SourceDescription:
    """Keep the metadata about the source parameter that was originally passed
    """
    def __init__(self, source, kind: SourceKind, has_inputs: bool, cycle: str):
        self.source = source
        self.kind = kind
        self.has_inputs = has_inputs
        self.cycle = cycle

    def __str__(self) -> str:
        if self.kind == SourceKind.CALLABLE:
            return "Callable source " + ("with" if self.has_inputs else "without") + " inputs: `{}`".format(self.source)
        elif self.kind == SourceKind.ITERABLE:
            return "Iterable (or iterator) source: `{}` with cycle: `{}`.".format(self.source, self.cycle)
        else:
            return "Generator function source: `{}` with cycle: `{}`.".format(self.source, self.cycle)


def assert_cpu_sample_data_type(sample, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    if isinstance(sample, np.ndarray):
        return True
    if types._is_mxnet_array(sample):
        if sample.ctx.device_type != 'cpu':
            raise TypeError("Unsupported callback return type. "
                            "GPU tensors are not supported. Got an MXNet GPU tensor.")
        return True
    if types._is_torch_tensor(sample):
        if sample.device.type != 'cpu':
            raise TypeError("Unsupported callback return type. "
                            "GPU tensors are not supported. Got a PyTorch GPU tensor.")
        return True
    elif isinstance(sample, tensors.TensorCPU):
        return True
    raise TypeError(error_str.format(type(sample)))


def assert_cpu_batch_data_type(batch, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    if isinstance(batch, tensors.TensorListCPU):
        return True
    elif isinstance(batch, list):
        for sample in batch:
            assert_cpu_sample_data_type(sample, error_str)
        return True
    elif assert_cpu_sample_data_type(batch, error_str):
        # Bach can be repsented as dense tensor
        return True
    else:
        raise TypeError(error_str.format(type(batch)))


def sample_to_numpy(sample, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    assert_cpu_sample_data_type(sample, error_str)
    if isinstance(sample, np.ndarray):
        return sample
    if types._is_mxnet_array(sample):
        if sample.ctx.device_type != 'cpu':
            raise TypeError("Unsupported callback return type. "
                            "GPU tensors are not supported. Got an MXNet GPU tensor.")
        return sample.asnumpy()
    if types._is_torch_tensor(sample):
        if sample.device.type != 'cpu':
            raise TypeError("Unsupported callback return type. "
                            "GPU tensors are not supported. Got a PyTorch GPU tensor.")
        return sample.numpy()
    elif isinstance(sample, tensors.TensorCPU):
        return np.array(sample)
    raise TypeError(error_str.format(type(sample)))


def batch_to_numpy(batch, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    assert_cpu_batch_data_type(batch, error_str)
    if isinstance(batch, tensors.TensorListCPU):
        # TODO(klecki): samples that are not uniform
        return batch.as_array()
    elif isinstance(batch, list):
        return [sample_to_numpy(sample, error_str) for sample in batch]
    else:
        return sample_to_numpy(batch, error_str)

class _CycleIter:
    def __init__(self, iterable, mode):
        self.source = iterable
        self.signaling = (mode == "raise")

    def __iter__(self):
        self.it = iter(self.source)
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.source)
            if self.signaling:
                raise
            else:
                return next(self.it)

class _CycleGenFunc():
    def __init__(self, gen_func, mode):
        self.source = gen_func
        self.signaling = (mode == "raise")

    def __iter__(self):
        self.it = iter(self.source())
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.source())
            if self.signaling:
                raise
            else:
                return next(self.it)


def _is_generator_function(x):
    """Checks whether x is a generator function or a callable object
    where __call__ is a generator function"""
    if inspect.isgeneratorfunction(x):
        return True
    if isinstance(x, functools.partial):
        return _is_generator_function(x.func)

    if x is None or inspect.isfunction(x) or inspect.ismethod(x):
        return False
    call = getattr(x, "__call__", None)
    if call == x:
        return False
    return _is_generator_function(call)

def _cycle_enabled(cycle):
    if cycle is None:
        return False
    if cycle == False or cycle == "no":
        return False
    if cycle == True or cycle == "quiet" or cycle == "raise":
        return True
    raise ValueError("""Invalid value {} for the argument `cycle`. Valid values are
  - "no", False or None - cycling disabled
  - "quiet", True - quietly rewind the data
  - "raise" - raise StopIteration on each rewind.""".format(repr(cycle)))

def accepted_arg_count(callable):
    if not inspect.isfunction(callable) and not inspect.ismethod(callable) and hasattr(callable, '__call__'):
        callable = callable.__call__
    if not inspect.ismethod(callable):
        implicit_args = 0
    else:
        implicit_args = 1
        callable = callable.__func__
    return callable.__code__.co_argcount - implicit_args

def get_callback_from_source(source, cycle):
    """Repack the source into a unified callback function. Additionally prepare the SourceDescription.

    Returns
    -------
    callback, SourceDescription
    """
    iterable = False
    desc = None
    if source is not None:
        try:
            if _cycle_enabled(cycle):
                if inspect.isgenerator(source):
                    raise TypeError("Cannot cycle through a generator - if the generator is a result "
                        "of calling a generator function, pass that function instead as `source`.")
                if _is_generator_function(source):
                    # We got a generator function, each call returns new "generator iterator"
                    desc = SourceDescription(source, SourceKind.GENERATOR_FUNC, False, cycle)
                    iterator = iter(_CycleGenFunc(source, cycle))
                else:
                    # We hopefully got an iterable, iter(source) should return new iterator.
                    # TODO(klecki): Iterators are self-iterable (they return self from `iter()`),
                    #               add a check if we have iterable and not iterator here,
                    #               so we can better support cycle.
                    desc = SourceDescription(source, SourceKind.ITERABLE, False, cycle)
                    iterator = iter(_CycleIter(source, cycle))
            else:
                # In non-cycling case, we go over the data once.
                if _is_generator_function(source):
                    # If we got a generator, we extract the "generator iterator"
                    desc = SourceDescription(source, SourceKind.GENERATOR_FUNC, False, cycle)
                    source = source()
                else:
                    desc = SourceDescription(source, SourceKind.ITERABLE, False, cycle)

                # We try to use the iterable/iterator.
                # If this is callable instead, we will throw an error containing 'not iterable'
                # in the error message.
                iterator = iter(source)
            iterable = True
            callback = lambda: next(iterator)
        except TypeError as err:
            if "not iterable" not in str(err):
                raise(err)
            if cycle is not None:
                raise ValueError("The argument `cycle` can only be specified if `source` is iterable")
            if not callable(source):
                raise TypeError("Source must be callable, iterable or a parameterless generator function")
            # We got a callable
            desc = SourceDescription(source, SourceKind.CALLABLE,
                                      accepted_arg_count(source) > 0, cycle)
            callback = source
    else:
        desc = None
        callback = None

    if not iterable and cycle:
        raise ValueError("`cycle` argument is only valid for iterable `source`")
    return callback, desc
