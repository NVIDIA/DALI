# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

__all__ = ["Reader", "DictMapper", "ToTorch"]

import functools
from collections.abc import Iterable, Iterator
from typing import Any, Callable, TypeVar

import nvidia.dali.backend as _backend
import torch
import torchdata.nodes as tn

from .. import Batch, Tensor, _ops

T = TypeVar("T", bound=Tensor | Batch)


class _CUDANodeMeta(type(tn.BaseNode)):
    """
    Ensure that the device seen in next is the same as when the node was constructed,
    even if it's called from a different thread.
    """

    def __new__(cls, name, bases, classdict):
        if next := classdict.get("next"):
            classdict["next"] = cls._next_wrapper(next)
        if reset := classdict.get("reset"):
            classdict["reset"] = cls._reset_wrapper(reset)

        return super().__new__(cls, name, bases, classdict)

    @classmethod
    def _next_wrapper(cls, orig):
        @functools.wraps(orig)
        def next(self):
            expected_device = getattr(self, "_cuda_device_id", None)
            if expected_device is None:
                return orig(self)

            # Now we can safely call cudaGetDevice
            initial_device = _backend.GetCUDACurrentDevice()
            if initial_device == expected_device:
                return orig(self)

            _backend.SetCUDACurrentDevice(expected_device)
            try:
                return orig(self)
            finally:
                _backend.SetCUDACurrentDevice(initial_device)

        return next

    @classmethod
    def _reset_wrapper(cls, orig):
        @functools.wraps(orig)
        def reset(self, initial_state=None):
            if _backend.GetCUDADeviceCount() > 0:
                self._cuda_device_id = _backend.GetCUDACurrentDevice()
            else:
                self._cuda_device_id = None
            return orig(self, initial_state)

        return reset


class Reader(tn.BaseNode[dict[str, T]], metaclass=_CUDANodeMeta):
    """Wraps a reader as a node, yielding dictionaries.

    Parameters
    ----------
    reader_type : reader subclass
        The type of the reader to construct.
    batch_size : int, optional
        The batch size to pass to next_epoch(). If None, the iterator returns tensors.
    output_names : iterable of str, optional
        Names of the outputs, used as keys in the output dict.
        If the reader has two outputs, it can be omited and defaults to ``("data", "label")``.
    **kwargs
        Additional keyword arguments to pass to the reader constructor.
    """

    def __init__(
        self,
        reader_type: type[_ops.Reader],
        *,
        batch_size: int | None,
        output_names: Iterable[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        if batch_size is not None and "max_batch_size" not in kwargs:
            kwargs["max_batch_size"] = batch_size

        self._reader = reader_type(**kwargs)
        self._batch_size = batch_size
        self._epoch: Iterator[tuple[T, ...]] | None = None

        num_outputs = self._reader._infer_num_outputs()
        if output_names is not None:
            self._output_names = tuple(output_names)
            if num_outputs != len(self._output_names):
                raise ValueError("Number of output names does not match number of outputs")
        elif num_outputs == 2:
            self._output_names = ("data", "label")
        else:
            raise ValueError(f"Cannot infer output names for reader {reader_type.__qualname__}")

    def next(self):
        if self._epoch is None:
            raise RuntimeError("Reader.next() called before reset(). Call reset() first.")
        outputs = next(self._epoch)
        return dict(zip(self._output_names, outputs))

    def reset(self, initial_state: dict | None = None):
        if initial_state is not None:
            raise NotImplementedError("DALI Dynamic readers don't support checkpointing yet")

        super().reset(initial_state)
        self._epoch = iter(self._reader.next_epoch(batch_size=self._batch_size))

    def get_state(self):
        return {}

    def get_metadata(self) -> _ops.ReaderMeta:
        """Returns the metadata of the underlying reader operator"""
        return self._reader.get_metadata()


class DictMapper(tn.BaseNode[dict[str, T]], metaclass=_CUDANodeMeta):
    """Applies a transform to a single key in the dict yielded by a source node.

    Parameters
    ----------
    source : :class:`torchdata.nodes.BaseNode`
        The source node to pull from. Yields dictionaries of tensors or batches.
    map_fn : callable
        The function to apply to the specified key. Must return a tensor or batch.
    key : str, optional
        The key to apply the function to. Defaults to ``"data"``.
    """

    def __init__(
        self,
        source: tn.BaseNode[dict[str, T]],
        map_fn: Callable[[T], Tensor | Batch],
        key: str = "data",
    ):
        super().__init__()
        self._source = source
        self._map_fn = map_fn
        self._key = key

    def next(self) -> dict[str, Tensor | Batch]:
        data_dict = self._source.next().copy()
        data_dict[self._key] = self._map_fn(data_dict[self._key])
        return data_dict

    def reset(self, initial_state=None):
        super().reset(initial_state)
        self._source.reset(initial_state)

    def get_state(self):
        return {}


class ToTorch(tn.BaseNode[tuple[torch.Tensor, ...]], metaclass=_CUDANodeMeta):
    """Converts dictionaries of tensors or batches to tuples of :class:`torch.Tensor`.

    Parameters
    ----------
    source : :class:`torchdata.nodes.BaseNode`
        The source node to pull data from. Yields dictionaries of tensors or batches.
    """

    def __init__(self, source: tn.BaseNode[dict[str, Tensor | Batch]]):
        super().__init__()
        self._source = source

    def next(self) -> tuple[torch.Tensor, ...]:
        data_dict = self._source.next().copy()

        # Make sure that all elements are on the same device
        # Automatically move from CPU to GPU, but error out if GPU device indices differ
        cpu_keys: set[str] = set()
        device_index = None
        for key, data in data_dict.items():
            if data.device.device_type == "cpu":
                cpu_keys.add(key)
                continue

            if device_index is None:
                device_index = data.device.device_id
            elif device_index != data.device.device_id:
                raise ValueError("All GPU tensors/batches must be on the same device")

        if device_index is not None:
            for key in cpu_keys:
                data_dict[key] = data_dict[key].gpu(device_index)

        return tuple(data.torch() for data in data_dict.values())

    def reset(self, initial_state=None):
        super().reset(initial_state)
        self._source.reset(initial_state)

    def get_state(self):
        return {}
