# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            raise RuntimeError(
                "Could not import numpy. Please make sure you have numpy "
                "installed before you use parallel mode."
            )


class SourceKind(Enum):
    CALLABLE = 0
    ITERABLE = 1
    GENERATOR_FUNC = 2


class SourceDescription:
    """Keep the metadata about the source parameter that was originally passed"""

    def __init__(self, source, kind: SourceKind, has_inputs: bool, cycle: str, batch_info=False):
        self.source = source
        self.kind = kind
        self.has_inputs = has_inputs
        self.cycle = cycle
        self.batch_info = batch_info

    def __str__(self) -> str:
        if self.kind == SourceKind.CALLABLE:
            inputs = "with" if self.has_inputs else "without"
            return f"Callable source {inputs} inputs: `{self.source}`"
        elif self.kind == SourceKind.ITERABLE:
            return f"Iterable (or iterator) source: `{self.source}` with cycle: `{self.cycle}`."
        else:
            return f"Generator function source: `{self.source}` with cycle: `{self.cycle}`."


_tf_sample_error_msg = (
    "Unsupported callback return type. Expected NumPy array, PyTorch or MXNet cpu tensors, "
    "DALI TensorCPU representing sample. Got `{}` instead."
)


_tf_batch_error_msg = (
    "Unsupported callback return type. Expected NumPy array, PyTorch or MXNet cpu tensors, "
    "DALI TensorCPU, list of those types or DALI TensorListCPU representing batch. "
    "Got `{}` instead."
)

_tf_uniform_error_msg = (
    "Unsupported callback return value. TensorFlow requires that the batches produced by input "
    "datasets or External Source `source` callback in batch mode (that is when batch=True) "
    " are dense and uniform - this means that every sample has the same shape. Got `{}` instead."
)


def assert_cpu_sample_data_type(sample, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    if isinstance(sample, np.ndarray):
        return True
    if types._is_mxnet_array(sample):
        if sample.context.device_type != "cpu":
            raise TypeError(
                "Unsupported callback return type. "
                "GPU tensors are not supported. Got an MXNet GPU tensor."
            )
        return True
    if types._is_torch_tensor(sample):
        if sample.device.type != "cpu":
            raise TypeError(
                "Unsupported callback return type. "
                "GPU tensors are not supported. Got a PyTorch GPU tensor."
            )
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
        # Bach can be represented as dense tensor
        return True
    else:
        raise TypeError(error_str.format(type(batch)))


def sample_to_numpy(sample, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    assert_cpu_sample_data_type(sample, error_str)
    if isinstance(sample, np.ndarray):
        return sample
    if types._is_mxnet_array(sample):
        if sample.context.device_type != "cpu":
            raise TypeError(
                "Unsupported callback return type. "
                "GPU tensors are not supported. Got an MXNet GPU tensor."
            )
        return sample.asnumpy()
    if types._is_torch_tensor(sample):
        if sample.device.type != "cpu":
            raise TypeError(
                "Unsupported callback return type. "
                "GPU tensors are not supported. Got a PyTorch GPU tensor."
            )
        return sample.numpy()
    elif isinstance(sample, tensors.TensorCPU):
        return np.array(sample)
    raise TypeError(error_str.format(type(sample)))


def batch_to_numpy(
    batch,
    error_str="Unsupported callback return type. Got: `{}`.",
    non_uniform_str="Uniform input is required (batch of tensors of equal shapes), got {}.",
):
    import_numpy()
    assert_cpu_batch_data_type(batch, error_str)
    if isinstance(batch, tensors.TensorListCPU):
        if not batch.is_dense_tensor():
            raise ValueError(non_uniform_str.format(batch))
        return batch.as_array()
    elif isinstance(batch, list):
        result = [sample_to_numpy(sample, error_str) for sample in batch]
        first_shape = result[0].shape
        for sample in result:
            if first_shape != sample.shape:
                raise ValueError(non_uniform_str.format(batch))
        return np.stack(result)
    else:
        return sample_to_numpy(batch, error_str)


class _CycleIter:
    def __init__(self, iterable, mode):
        self.source = iterable
        self.signaling = mode == "raise"

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


class _CycleGenFunc:
    def __init__(self, gen_func, mode):
        self.source = gen_func
        self.signaling = mode == "raise"

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
    if call is x:
        return False
    return _is_generator_function(call)


def _cycle_enabled(cycle):
    if cycle is None:
        return False
    if cycle is False or cycle == "no":
        return False
    if cycle is True or cycle == "quiet" or cycle == "raise":
        return True
    raise ValueError(
        """Invalid value {} for the argument `cycle`. Valid values are
  - "no", False or None - cycling disabled
  - "quiet", True - quietly rewind the data
  - "raise" - raise StopIteration on each rewind.""".format(
            repr(cycle)
        )
    )


def accepted_arg_count(callable):
    """Checks the number of accepted arguments by the callable, and validates if it is either
    0 or 1 positional argument allowed by the external source.

    Raises
    ------
    TypeError
        Indicates that the `source` callable accepts wrong number of type of arguments.
    """

    if not (inspect.isfunction(callable) or inspect.ismethod(callable)) and hasattr(
        callable, "__call__"
    ):
        callable = callable.__call__
    # Extracting the `__call__` for a method causes the signature to report `self` as a parameter,
    # so we have to subtract it.
    if not inspect.ismethod(callable):
        implicit_args = 0
    else:
        implicit_args = 1
        callable = callable.__func__
    signature = inspect.signature(callable)
    # TODO(klecki): Do we mention that callable is one of the alternatives?
    error_msg = (
        "The `source` callable must accept either 0 or 1 positional arguments to indicate "
        "whether it accepts the batch or sample indexing information."
    )
    for p in signature.parameters.values():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                error_msg + f" Found var-positional argument `*{p.name}` which is not allowed."
            )
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            raise TypeError(
                error_msg + f" Found var-keyword argument `**{p.name}` which is not allowed."
            )
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                error_msg + f" Found keyword-only argument `{p.name}` which is not allowed."
            )
    result = len(signature.parameters) - implicit_args
    if result not in [0, 1]:
        raise TypeError(
            error_msg + " Found more than one positional argument, which is not allowed."
        )
    return result


def get_callback_from_source(source, cycle, batch_info=False):
    """Repack the source into a unified callback function. Additionally prepare
    the SourceDescription.

    `batch_info` is usable only with callables.

    Returns
    -------
    callback, SourceDescription
    """
    is_iterable = False
    is_callable = False
    desc = None
    if source is not None:
        try:
            if _cycle_enabled(cycle):
                if inspect.isgenerator(source):
                    raise TypeError(
                        "Cannot cycle through a generator - if the generator is "
                        "a result of calling a generator function, "
                        "pass that function instead as `source`."
                    )
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
            is_iterable = True
            callback = lambda: next(iterator)  # noqa E731
        except TypeError as err:
            if "not iterable" not in str(err):
                raise err
            if cycle is not None:
                raise ValueError(
                    "The argument `cycle` can only be specified " "if `source` is iterable."
                )
            if not callable(source):
                raise TypeError(
                    "The `source` must be callable, iterable or a parameterless generator function."
                )
            # We got a callable
            is_callable = True
        # We want to exit the scope of except, to raise a separate exception when doing a validation
        # via the accepted_arg_count.
        if is_callable:
            desc = SourceDescription(
                source, SourceKind.CALLABLE, accepted_arg_count(source) > 0, cycle, batch_info
            )
            callback = source
    else:
        desc = None
        callback = None

    if not is_iterable and cycle:
        raise ValueError("`cycle` argument is only valid for iterable `source`")
    return callback, desc


# TODO(klecki): Maybe keep this data here instead of doing the copy twice
def _inspect_data(data, is_batched):
    # TODO(klecki): Add asserts for uniform input batches (as well as output batches)
    if is_batched:
        as_numpy = batch_to_numpy(data, _tf_batch_error_msg, non_uniform_str=_tf_uniform_error_msg)
        if isinstance(as_numpy, list):
            return as_numpy[0].dtype, (None,) * (as_numpy[0].ndim + 1)
        else:
            return as_numpy.dtype, (None,) * as_numpy.ndim
    else:
        as_numpy = sample_to_numpy(data, _tf_sample_error_msg)
        return as_numpy.dtype, (None,) * as_numpy.ndim


def get_batch_iterable_from_callback(source_desc: SourceDescription):
    """Transform batch callback accepting one argument into an Iterable"""
    first = source_desc.source(types.BatchInfo(0, 0) if source_desc.batch_info else 0)
    dtype, shape = _inspect_data(first, True)

    class CallableBatchIterator:
        first_value = first

        def __init__(self):
            self.iteration = 0
            self.source = source_desc.source

        def __iter__(self):
            self.iteration = 0
            return self

        def __next__(self):
            if self.iteration == 0 and CallableBatchIterator.first_value is not None:
                result = CallableBatchIterator.first_value
                CallableBatchIterator.first_value = None
            else:
                if source_desc.batch_info:
                    # There is no notion of epochs when iterating over DALI Dataset
                    # as the "raise" policy is not supported, so we use epoch 0 only.
                    argument = types.BatchInfo(self.iteration, 0)
                else:
                    argument = self.iteration
                result = self.source(argument)
            self.iteration += 1
            return batch_to_numpy(
                result, _tf_batch_error_msg, non_uniform_str=_tf_uniform_error_msg
            )

    return CallableBatchIterator, dtype, shape


def get_sample_iterable_from_callback(source_desc: SourceDescription, batch_size):
    """Transform sample callback accepting one argument into an Iterable"""
    first = source_desc.source(types.SampleInfo(0, 0, 0, 0))
    dtype, shape = _inspect_data(first, False)

    class CallableSampleIterator:
        first_value = first

        def __init__(self):
            self.idx_in_epoch = 0
            self.idx_in_batch = 0
            self.iteration = 0
            self.source = source_desc.source

        def __iter__(self):
            self.idx_in_epoch = 0
            self.idx_in_batch = 0
            self.iteration = 0
            return self

        def __next__(self):
            if self.idx_in_epoch == 0 and CallableSampleIterator.first_value is not None:
                result = CallableSampleIterator.first_value
                CallableSampleIterator.first_value = None
            else:
                # There is no notion of epochs when iterating over DALI Dataset
                # as the "raise" policy is not supported, so we use epoch 0 only.
                idx = types.SampleInfo(self.idx_in_epoch, self.idx_in_batch, self.iteration, 0)
                result = self.source(idx)
            self.idx_in_epoch += 1
            self.idx_in_batch += 1
            if self.idx_in_batch == batch_size:
                self.idx_in_batch = 0
                self.iteration += 1
            return sample_to_numpy(result, _tf_sample_error_msg)

    return CallableSampleIterator, dtype, shape


def get_iterable_from_callback(source_desc: SourceDescription, is_batched):
    """Transform callback that doesn't accept arguments into iterable"""
    print("get_iterable_from_callback")
    first = source_desc.source()
    dtype, shape = _inspect_data(first, is_batched)

    class CallableIterator:
        first_value = first

        def __init__(self):
            self.source = source_desc.source

        def __iter__(self):
            return self

        def __next__(self):
            if CallableIterator.first_value is not None:
                result = CallableIterator.first_value
                CallableIterator.first_value = None
            else:
                result = self.source()
            if is_batched:
                return batch_to_numpy(
                    result, _tf_batch_error_msg, non_uniform_str=_tf_uniform_error_msg
                )
            else:
                return sample_to_numpy(result, _tf_sample_error_msg)

    return CallableIterator, dtype, shape


def get_iterable_from_iterable_or_generator(source_desc: SourceDescription, is_batched):
    """Wrap iterable or generator function into another iterable while peeking the first element

    If the source is generator function it must be called first.
    """
    if source_desc.kind == SourceKind.GENERATOR_FUNC:
        first_iter = iter(source_desc.source())
    else:
        first_iter = iter(source_desc.source)
    first = next(first_iter)
    dtype, shape = _inspect_data(first, is_batched)

    class PeekFirstGenerator:
        first_iterator = first_iter
        first_value = first

        def __init__(self):
            self.source = source_desc.source

        def __iter__(self):
            if PeekFirstGenerator.first_iterator is not None:
                self.it = PeekFirstGenerator.first_iterator
                PeekFirstGenerator.first_iterator = None
            else:
                if source_desc.kind == SourceKind.GENERATOR_FUNC:
                    self.it = iter(source_desc.source())
                else:
                    self.it = iter(source_desc.source)
            return self

        def __next__(self):
            if PeekFirstGenerator.first_value is not None:
                result = PeekFirstGenerator.first_value
                PeekFirstGenerator.first_value = None
            else:
                result = next(self.it)
            if is_batched:
                return batch_to_numpy(
                    result, _tf_batch_error_msg, non_uniform_str=_tf_uniform_error_msg
                )
            else:
                return sample_to_numpy(result, _tf_sample_error_msg)

    return PeekFirstGenerator, dtype, shape


def _get_generator_from_source_desc(source_desc: SourceDescription, batch_size, is_batched):
    """Based on DALI source description create a generator function, type and shape specification
    compatible with TF Generator Dataset.

    Cycling is delegated to the dataset as some control of some cycling behavior cannot be
    realized in TF.
    """
    if source_desc.kind == SourceKind.CALLABLE:
        if source_desc.has_inputs:
            if is_batched:
                return get_batch_iterable_from_callback(source_desc)
            else:
                return get_sample_iterable_from_callback(source_desc, batch_size)
        else:
            # No inputs, plain iteration
            return get_iterable_from_callback(source_desc, is_batched)
    else:
        # Generator Func or iterable
        return get_iterable_from_iterable_or_generator(source_desc, is_batched)
