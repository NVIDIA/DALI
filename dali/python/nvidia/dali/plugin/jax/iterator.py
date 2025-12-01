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
import jax
import jax.dlpack
import jax.numpy as jnp
from jax.sharding import NamedSharding, PositionalSharding, Sharding

from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.pipeline import Pipeline, DataNode

from nvidia.dali.plugin.jax.integration import _to_jax_array

from typing import Union, Optional, Callable, Type, Dict, List, Tuple


class DALIGenericIterator(_DaliBaseIterator):
    """
    General DALI iterator for JAX. It can return any number of
    outputs from the DALI pipeline in the form of JAX Arrays.

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
        # check the assert first as _DaliBaseIterator would run the prefetch
        if len(set(output_map)) != len(output_map):
            raise AssertionError("output_map names should be distinct")
        self._output_categories = set(output_map)
        self.output_map = output_map

        if sharding is not None:
            assert isinstance(
                sharding, (NamedSharding, PositionalSharding)
            ), "`sharding` should be an instance of `NamedSharding` or `PositionalSharding`"
        self._sharding = sharding

        assert (
            last_batch_policy != LastBatchPolicy.PARTIAL
        ), "JAX iterator does not support partial last batch policy."

        _DaliBaseIterator.__init__(
            self,
            pipelines,
            size,
            reader_name,
            auto_reset,
            None,  # Default value for deprecated fill_last_batch argument
            last_batch_padded,
            last_batch_policy,
            prepare_first_batch=prepare_first_batch,
        )

        self._first_batch = None
        if self._prepare_first_batch:
            try:
                self._first_batch = DALIGenericIterator.__next__(self)
                # call to `next` sets _ever_consumed to True but if we are just calling it from
                # here we should set if to False again
                self._ever_consumed = False
            except StopIteration:
                self._report_no_data_in_pipeline()

    def _next_impl(self):
        self._ever_consumed = True
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        pipelines_outputs = self._get_outputs()  # Can be accessed by outputs[device_id][output_id]

        next_output = dict()
        for category_id, category_name in enumerate(self.output_map):
            category_outputs = self._gather_outputs_for_category(pipelines_outputs, category_id)

            if self._num_gpus == 1 and self._sharding is None:
                next_output[category_name] = category_outputs[0]
            else:
                self._assert_shards_shapes(category_outputs)
                if self._sharding is not None:
                    next_output[category_name] = self._build_output_with_sharding(category_outputs)
                else:
                    next_output[category_name] = self._build_output_with_device_put(
                        next_output, category_name, category_outputs
                    )

        self._schedule_runs()
        self._advance_and_check_drop_last()

        return next_output

    def __next__(self) -> Dict[str, jax.Array]:
        return self._next_impl()

    def _gather_outputs_for_category(self, pipelines_outputs, category_id):
        category_outputs = []

        for pipeline_id in range(self._num_gpus):
            category_outputs.append(
                _to_jax_array(
                    pipelines_outputs[pipeline_id][category_id].as_tensor(),
                    not self._pipes[pipeline_id].exec_dynamic,
                )
            )

        return category_outputs

    def _build_output_with_device_put(self, next_output, category_name, category_outputs):
        """Builds sharded jax.Array with `jax.device_put_sharded`. This output is compatible
        with pmapped JAX functions.
        """
        category_outputs_devices = tuple(
            map(lambda jax_shard: jax_shard.device(), category_outputs)
        )

        distinct_category_outputs_devices = set(category_outputs_devices)

        if len(category_outputs_devices) != len(distinct_category_outputs_devices):
            if len(distinct_category_outputs_devices) != 1:
                raise AssertionError(
                    "JAX iterator requires shards to be placed on \
                                                different devices or all on the same device."
                )
            else:
                # All shards are on one device (CPU or one GPU)
                return jnp.stack(category_outputs)
        else:
            # Build sharded JAX array as output for current category (compatible with pmap)
            return jax.device_put_sharded(category_outputs, category_outputs_devices)

    def _build_output_with_sharding(self, category_outputs):
        """Builds sharded jax.Array with `jax.make_array_from_single_device_arrays`.
        This output is compatible with automatic parallelization with JAX.
        """
        shard_shape = category_outputs[0].shape

        if isinstance(self._sharding, NamedSharding):
            global_shape = (self._sharding.mesh.size * shard_shape[0], *shard_shape[1:])
        else:
            global_shape = (self._sharding.shape[0] * shard_shape[0], *shard_shape[1:])

        return jax.make_array_from_single_device_arrays(
            global_shape, self._sharding, category_outputs
        )

    def _assert_shards_shapes(self, category_outputs):
        for shard in category_outputs:
            assert shard.shape == category_outputs[0].shape, "Shards shapes have to be the same."


def default_num_threads_value() -> int:
    """Returns default value for num_threads argument of DALI iterator decorator.

    .. note::
        This value should not be considered as optimized for any particular workload. For best
            performance, it is recommended to set this value manually.

    .. note::
        This value is subject to change in the future."""

    return 4


def _data_iterator_impl(
    iterator_type: Type[DALIGenericIterator],
    pipeline_fn: Optional[Callable[..., Union[DataNode, Tuple[DataNode, ...]]]],
    output_map: List[str],
    size: int = -1,
    reader_name: Optional[str] = None,
    auto_reset: Union[str, bool, None] = False,
    last_batch_padded: bool = False,
    last_batch_policy: LastBatchPolicy = LastBatchPolicy.FILL,
    prepare_first_batch: bool = True,
    sharding: Optional[Sharding] = None,
    devices: Optional[List[jax.Device]] = None,
):
    """Implementation of the data_iterator decorator. It is extracted to a separate function
    to be reused by the peekable iterator decorator.
    """

    if sharding is not None and devices is not None:
        raise ValueError("Only one of `sharding` and `devices` arguments can be provided.")

    def data_iterator_decorator(func):
        def create_iterator(*args, checkpoints=None, **wrapper_kwargs):
            pipeline_def_fn = pipeline_def(func)

            if "num_threads" not in wrapper_kwargs:
                wrapper_kwargs["num_threads"] = default_num_threads_value()

            if sharding is None and devices is None:
                if "device_id" not in wrapper_kwargs:
                    # Due to https://github.com/google/jax/issues/16024 the best we can do is to
                    # assume that the first device is the one we want to use.
                    wrapper_kwargs["device_id"] = 0

                checkpoint = checkpoints[0] if checkpoints else None
                pipelines = [pipeline_def_fn(*args, checkpoint=checkpoint, **wrapper_kwargs)]

                return iterator_type(
                    pipelines=pipelines,
                    output_map=output_map,
                    size=size,
                    reader_name=reader_name,
                    auto_reset=auto_reset,
                    last_batch_padded=last_batch_padded,
                    last_batch_policy=last_batch_policy,
                    prepare_first_batch=prepare_first_batch,
                )
            else:
                pipelines = []

                # Handle batch_size_per_gpu
                global_batch_size = wrapper_kwargs["batch_size"]
                batch_size_per_gpu = global_batch_size // len(jax.devices())

                wrapper_kwargs["batch_size"] = batch_size_per_gpu

                devices_to_use = devices if devices is not None else jax.local_devices()
                num_shards = len(devices) if devices is not None else jax.device_count()

                if devices is not None:
                    if jax.local_device_count() != jax.device_count():
                        raise RuntimeError(
                            "Iterator compatible with pmapped JAX functions does not support "
                            "running in multiprocess mode. Use `sharding` argument instead."
                        )

                if checkpoints and len(checkpoints) != len(devices_to_use):
                    raise RuntimeError(
                        f"The number of checkpoints provided ({len(checkpoints)}) should match "
                        f"the number of devices to use ({len(devices_to_use)})."
                    )

                for id, device in enumerate(devices_to_use):
                    # How device_id, shard_id and num_shards are used in the pipeline
                    # is affected by: https://github.com/google/jax/issues/16024
                    # TODO(awolant): Should this match device with index in jax.devices() as id?
                    # This is connected with pmap experimental `devices` argument.
                    checkpoint = checkpoints[id] if checkpoints else None
                    pipeline = pipeline_def_fn(
                        *args,
                        **wrapper_kwargs,
                        device_id=id,
                        shard_id=device.id,
                        num_shards=num_shards,
                        checkpoint=checkpoint,
                    )

                    pipelines.append(pipeline)

                if sharding is not None:
                    return iterator_type(
                        pipelines=pipelines,
                        output_map=output_map,
                        size=size,
                        reader_name=reader_name,
                        auto_reset=auto_reset,
                        last_batch_padded=last_batch_padded,
                        last_batch_policy=last_batch_policy,
                        prepare_first_batch=prepare_first_batch,
                        sharding=sharding,
                    )
                elif devices is not None:
                    return iterator_type(
                        pipelines=pipelines,
                        output_map=output_map,
                        size=size,
                        reader_name=reader_name,
                        auto_reset=auto_reset,
                        last_batch_padded=last_batch_padded,
                        last_batch_policy=last_batch_policy,
                        prepare_first_batch=prepare_first_batch,
                    )

                raise AssertionError(
                    "This should not happen. Check `sharding` and `devices` arguments."
                )

        return create_iterator

    return data_iterator_decorator(pipeline_fn) if pipeline_fn else data_iterator_decorator


def data_iterator(
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
    """Decorator for DALI iterator for JAX. Decorated function when called returns DALI
    iterator for JAX.

    Decorated function should return DALI pipeline definition function. Decorator accepts
    all arguments of :meth:`nvidia.dali.plugin.base_iterator.DALIGenericIterator.__init__` and
    passes them to the iterator constructor.
    If no `device_id` argument is passed to the decorated function, it is assumed that
    the first device is the one we want to use and `device_id` is set to 0.
    If the same argument is passed to the decorator and the decorated function,
    exception is raised.

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
        DALIGenericIterator,
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
