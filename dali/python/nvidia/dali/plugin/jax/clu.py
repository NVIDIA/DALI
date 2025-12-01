# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import threading

import numpy as np
import jax.numpy as jnp

from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.jax.iterator import DALIGenericIterator, _data_iterator_impl
from nvidia.dali.pipeline import Pipeline, DataNode

from clu.data.dataset_iterator import ArraySpec, ElementSpec
import concurrent.futures

from typing import Union, Optional, Callable, Dict, List, Tuple

import jax
from jax.sharding import Sharding


def get_spec_for_array(jax_array):
    """Utility to get ArraySpec for given JAX array."""

    return ArraySpec(shape=jax_array.shape, dtype=jax_array.dtype)


class DALIGenericPeekableIterator(DALIGenericIterator):
    """DALI iterator for JAX with peek functionality. Compatible with Google CLU PeekableIterator.
    It supports peeking the next element in the iterator without advancing the iterator.

    Note:
        It is compatible with pipelines that return outputs with constant shape and type. It will
        throw an exception if the shape or type of the output changes between iterations.

    It provides ``element_spec`` property that returns a dictionary of ``ArraySpec`` objects
    for each output category.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of str
                List of strings which maps consecutive outputs
                of DALI pipelines to user specified name.
                Outputs will be returned from iterator as dictionary
                of those names.
                Each name should be distinct
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
                Providing -1 means that the iterator will work until StopIteration is raised
                from the inside of iter_setup(). The options `last_batch_policy` and
                `last_batch_padded` don't work in such case. It works with only one pipeline inside
                the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried for the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_padded`
                accordingly to match the reader's configuration.
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                is called internally automatically.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`.
                JAX iterator does not support LastBatchPolicy.PARTIAL
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with `last_batch_policy` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data
    sharding : `jax.sharding.Sharding`
                `jax.sharding.Sharding` compatible object that, if present, will be used to
                build an output jax.Array for each category. If ``None``, the iterator returns
                values compatible with pmapped JAX functions, if multiple pipelines are provided.

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``

    Note:
        JAX iterator does not support LastBatchPolicy.PARTIAL.
    """

    def __init__(
        self,
        pipelines: Union[List[Pipeline], Pipeline],
        output_map: List[str],
        size: int = -1,
        reader_name: Optional[str] = None,
        auto_reset: Union[str, bool, None] = False,
        last_batch_padded: bool = False,
        last_batch_policy: LastBatchPolicy = LastBatchPolicy.FILL,
        prepare_first_batch: bool = True,
        sharding: Optional[Sharding] = None,
    ):
        super().__init__(
            pipelines,
            output_map,
            size,
            reader_name,
            auto_reset,
            last_batch_padded,
            last_batch_policy,
            prepare_first_batch,
            sharding,
        )
        self._mutex = threading.Lock()
        self._pool = None
        self._peek = None

        self._element_spec = None

    def _set_element_spec(self, output):
        self._element_spec = {
            output_name: get_spec_for_array(output[output_name])
            for output_name in self._output_categories
        }

    def _assert_output_shape_and_type(self, output):
        if self._element_spec is None:
            # Set element spec based on the first seen element
            self._set_element_spec(output)

        for key in output:
            if get_spec_for_array(output[key]) != self._element_spec[key]:
                raise ValueError(
                    "The shape or type of the output changed between iterations. "
                    "This is not supported by JAX  peekable iterator. "
                    "Please make sure that the shape and type of the output is constant. "
                    f"Expected: {self._element_spec[key]}, got: {get_spec_for_array(output[key])} "
                    f"for output: {key}"
                )

        return output

    def _next_with_peek_impl(self):
        """Returns the next element from the iterator and advances the iterator.
        Is extracted as a separate method to be used by ``peek`` and ``next`` methods
        under the same lock.
        """
        if self._peek is None:
            return self._assert_output_shape_and_type(self._next_impl())
        peek = self._peek
        self._peek = None
        return self._assert_output_shape_and_type(peek)

    def __next__(self) -> Dict[str, jax.Array]:
        with self._mutex:
            return self._next_with_peek_impl()

    def __iter__(self):
        if self._counter != 0 and self._ever_consumed and self._peek is None:
            self.reset()
        return self

    def peek(self):
        """Returns the next element from the iterator without advancing the iterator.

        Returns:
           dict : dictionary of jax.Array objects with the next element from the iterator.
        """
        with self._mutex:
            if self._peek is None:
                self._peek = self._next_with_peek_impl()
            return self._peek

    def peek_async(self):
        """Returns future that will return the next element from
        the iterator without advancing the iterator.

        Note:
            Calling ``peek_async`` without waiting for the future to complete is not
            guaranteed to be executed before the next call to ``peek`` or ``next``.
            If you want to make sure that the next call to ``peek`` or ``next`` will
            return the same element as the future, you need to wait for the future to
            complete.

        Returns:
           concurrent.futures.Future: future that will return dictionary of jax.Array
                                     objects with the next element from the iterator.
        """
        if self._pool is None:
            # Create pool only if needed (peek_async is ever called)
            # to avoid thread creation overhead
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        future = self._pool.submit(self.peek)
        return future

    def checkpoints(self):
        if self._peek is not None:
            raise RuntimeError(
                "Checkpointing is not supported for peekable iterators with peeked data. "
                "Consume the peeked data completely using next() before saving a checkpoint."
            )
        return super().checkpoints()

    @property
    def element_spec(self) -> ElementSpec:
        """Returns the element spec for the elements returned by the iterator.
        ElementSpec contains ``ArraySpec`` for each output category which describes
        shape and type of the output.

        Returns:
            ElementSpec: Element spec for the elements returned by the iterator.
        """
        if self._element_spec is None:
            self._set_element_spec(self.peek())
        return self._element_spec

    @property
    def is_nonpadding(self):
        """Returns array of booleans that indicate if the returned sample is a padding sample.

        Returns:
            jax.Array of booleans that indicates if the returned sample is a padding sample.
            Shape is (batch_size,). Sharding is the same as the output.
        """
        left = self.batch_size - (
            self._counter - self._shard_sizes_per_gpu_initial[self._shards_id]
        )
        if_drop = np.less(left, self.batch_size)

        is_nonpadding_shards = []

        for i in range(self._num_gpus):
            if not if_drop[i]:
                is_nonpadding_shards.append(np.ones((self.batch_size,), dtype=bool))
            else:
                is_nonpadding_shards.append(
                    np.concatenate(
                        (
                            np.ones((left[i],), dtype=bool),
                            np.zeros((self.batch_size - left[i],), dtype=bool),
                        )
                    )
                )

        if self._sharding is None:
            return np.concatenate(is_nonpadding_shards)

        return jax.device_put(jnp.concatenate(is_nonpadding_shards), self._sharding)


def peekable_data_iterator(
    pipeline_fn: Optional[Callable[..., Union[DataNode, Tuple[DataNode, ...]]]] = None,
    output_map: List[str] = [],
    size: int = -1,
    reader_name: Optional[str] = None,
    auto_reset: Union[str, bool, None] = False,
    last_batch_padded: bool = False,
    last_batch_policy: LastBatchPolicy = LastBatchPolicy.FILL,
    prepare_first_batch: bool = True,
    sharding: Optional[Sharding] = None,
    devices: Optional[List[jax.Device]] = None,
):
    """Decorator for DALI pipelines that returns a peekable iterator. Compatible with Google CLU
    PeekableIterator. It supports peeking the next element in the iterator without advancing the
    iterator.

    Parameters
    ----------
    pipeline_fn function:
                Function to be decorated. It should be compatible with
                :meth:`nvidia.dali.pipeline.pipeline_def` decorator.
                For multigpu support it should accept `device_id`, `shard_id` and `num_shards` args.
    output_map : list of str
                List of strings which maps consecutive outputs
                of DALI pipelines to user specified name.
                Outputs will be returned from iterator as dictionary
                of those names.
                Each name should be distinct
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
                Providing -1 means that the iterator will work until StopIteration is raised
                from the inside of iter_setup(). The options `last_batch_policy` and
                `last_batch_padded` don't work in such case. It works with only one pipeline inside
                the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried for the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_padded`
                accordingly to match the reader's configuration.
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                is called internally automatically.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`.
                JAX iterator does not support LastBatchPolicy.PARTIAL
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with `last_batch_policy` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data
    sharding : `jax.sharding.Sharding`
                `jax.sharding.Sharding` compatible object that, if present, will be used to
                build an output jax.Array for each category. Iterator will return outputs
                compatible with automatic parallelization in JAX.
                This argument is mutually exclusive with `devices` argument. If `devices` is
                provided, `sharding` should be set to None.
    devices : list of `jax.Device`
                List of JAX devices to be used to run the pipeline in parallel. Iterator will
                return outputs compatible with pmapped JAX functions.
                This argument is  mutually exclusive with `sharding` argument. If `sharding`
                is provided, `devices` should be set to None.
    checkpoints : list of str, optional, default = None
                Checkpoints obtained with `.checkpoints()` method of the iterator.
                If provided, they will be used to restore the state of the pipelines.

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``

    Note:
        JAX iterator does not support LastBatchPolicy.PARTIAL.
    """
    return _data_iterator_impl(
        DALIGenericPeekableIterator,
        pipeline_fn,
        output_map,
        size,
        reader_name,
        auto_reset,
        last_batch_padded,
        last_batch_policy,
        prepare_first_batch,
        sharding,
        devices,
    )
