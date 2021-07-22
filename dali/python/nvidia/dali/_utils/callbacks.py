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


class _SourceKind(Enum):
    CALLABLE       = 0
    ITERABLE       = 1
    GENERATOR_FUNC = 2

class _SourceDescription:
    """Keep the metadata about the source parameter that was originally passed
    """
    def __init__(self, source, kind: _SourceKind, has_inputs: bool, cycle: str):
        self.source = source
        self.kind = kind
        self.has_inputs = has_inputs
        self.cycle = cycle

    def __str__(self) -> str:
        if self.kind == _SourceKind.CALLABLE:
            return "Callable source " + ("with" if self.has_inputs else "without") + " inputs: `{}`".format(self.source)
        elif self.kind == _SourceKind.ITERABLE:
            return "Iterable (or iterator) source: `{}` with cycle: `{}`.".format(self.source, self.cycle)
        else:
            return "Generator function source: `{}` with cycle: `{}`.".format(self.source, self.cycle)


_tf_sample_error_msg = (
    "Unsupported callback return type. Expected NumPy array, PyTorch or MXNet cpu tensors, "
    "DALI TensorCPU representing sample. Got `{}` instead.")


_tf_batch_error_msg = (
    "Unsupported callback return type. Expected NumPy array, PyTorch or MXNet cpu tensors, "
    "DALI TensorCPU, list of those types or DALI TensorListCPU representing batch. Got `{}` instead.")


def _assert_cpu_sample_data_type(sample, error_str="Unsupported callback return type. Got: `{}`."):
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


def _assert_cpu_batch_data_type(batch, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    if isinstance(batch, tensors.TensorListCPU):
        return True
    elif isinstance(batch, list):
        for sample in batch:
            _assert_cpu_sample_data_type(sample, error_str)
        return True
    elif _assert_cpu_sample_data_type(batch, error_str):
        # Bach can be repsented as dense tensor
        return True
    else:
        raise TypeError(error_str.format(type(batch)))


def _sample_to_numpy(sample, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    _assert_cpu_sample_data_type(sample, error_str)
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


def _batch_to_numpy(batch, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    _assert_cpu_batch_data_type(batch, error_str)
    if isinstance(batch, tensors.TensorListCPU):
        # TODO(klecki): samples that are not uniform
        return batch.as_array()
    elif isinstance(batch, list):
        return [_sample_to_numpy(sample, error_str) for sample in batch]
    else:
        return _sample_to_numpy(batch, error_str)


# TODO(klecki): Maybe keep this data here instead of doing the copy twice
def _inspect_data(data, is_batched):
    if is_batched:
        as_numpy = _batch_to_numpy(data, _tf_batch_error_msg)
        if isinstance(as_numpy, list):
            return as_numpy[0].dtype, (None,) * (as_numpy[0].ndim + 1) # TODO(klecki): HANDLE THE LISTS
        else:
            return as_numpy.dtype, (None,) * as_numpy.ndim
    else:
        as_numpy = _sample_to_numpy(data, _tf_sample_error_msg)
        return as_numpy.dtype, (None,) * as_numpy.ndim


def get_batch_iterable_from_callback(source_desc):
    """Transform batch callback accepting one argument into an Iterable
    """

    first = source_desc.source(0)
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
                result = self.source(self.iteration)
            self.iteration += 1
            return _batch_to_numpy(result, _tf_batch_error_msg)

    return CallableBatchIterator, dtype, shape

def get_sample_iterable_from_callback(source_desc, batch_size):
    """Transform sample callback accepting one argument into an Iterable
    """
    first = source_desc.source(types.SampleInfo(0, 0, 0))
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
                idx = types.SampleInfo(self.idx_in_epoch, self.idx_in_batch, self.iteration)
                result = self.source(idx)
            self.idx_in_epoch += 1
            self.idx_in_batch += 1
            if self.idx_in_batch == batch_size:
                self.idx_in_batch = 0
                self.iteration += 1
            return _sample_to_numpy(result, _tf_sample_error_msg)

    return CallableSampleIterator, dtype, shape

def get_iterable_from_callback(source_desc, is_batched):
    """Transform callback that doesn't accept arguments into iterable
    """
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
                return _batch_to_numpy(result, _tf_batch_error_msg)
            else:
                return _sample_to_numpy(result, _tf_sample_error_msg)

    return CallableIterator, dtype, shape


def get_iterable_from_iterable_or_generator(source_desc, is_batched):
    """Wrap iterable or generator function into another iterable while peeking the first element

    If the source is generator function it must be called first.
    """
    if source_desc.kind == _SourceKind.GENERATOR_FUNC:
        first_iter = iter(source_desc.source())
    else:
        first_iter = iter(source_desc.source)
    first =  next(first_iter)
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
                if source_desc.kind == _SourceKind.GENERATOR_FUNC:
                    self.it = iter(source_desc.source())
                else:
                    first_iter = iter(source_desc.source)
            return self

        def __next__(self):
            if PeekFirstGenerator.first_value is not None:
                result = PeekFirstGenerator.first_value
                PeekFirstGenerator.first_value = None
            else:
                result = next(self.it)
            if is_batched:
                return _batch_to_numpy(result, _tf_batch_error_msg)
            else:
                return _sample_to_numpy(result, _tf_sample_error_msg)

    return PeekFirstGenerator, dtype, shape

def _get_generator_from_source_desc(source_desc, batch_size, is_batched):
    """Based on DALI source description create a generator function, type and shape specification
    compatible with TF Generator Dataset.

    Cycling is delegated to the dataset as some control of some cycling behaviour cannot be
    realized in TF.
    """
    if source_desc.kind == _SourceKind.CALLABLE:
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
