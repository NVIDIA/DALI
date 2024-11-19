# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import types
import math
import logging
import numpy as np
import warnings
import pickle  # nosec B403
from enum import Enum, unique
from collections.abc import Iterable


def _iterator_deprecation_warning():
    warnings.warn(
        "Please set `reader_name` and don't set last_batch_padded and size manually "
        + "whenever possible. This may lead, in some situations, to missing some "
        + "samples or returning duplicated ones. Check the Sharding section of the "
        "documentation for more details.",
        Warning,
        stacklevel=2,
    )


@unique
class LastBatchPolicy(Enum):
    """
    Describes the last batch policy behavior when there are not enough samples in the epoch
    to fill a whole batch.

        * FILL - The last batch is filled by either repeating the last sample or by wrapping
          up the data set. The precise behavior depends on the reader's ``pad_last_batch`` argument
        * DROP - The last batch is dropped if it cannot be fully filled with data from the current
          epoch
        * PARTIAL - The last batch is partially filled with the remaining data from the current\
          epoch, keeping the rest of the samples empty
    """

    FILL = 0
    DROP = 1
    PARTIAL = 2


class _DaliBaseIterator(object):
    """
    DALI base iterator class. Shouldn't be used directly.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of (str, str)
                 List of pairs (output_name, tag) which maps consecutive
                 outputs of DALI pipelines to proper field in MXNet's
                 DataBatch.
                 tag is one of DALIGenericIterator.DATA_TAG
                 and DALIGenericIterator.LABEL_TAG mapping given output
                 for data or label correspondingly.
                 output_names should be distinct.
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than one
                it is a sum). Providing -1 means that the iterator will work until StopIteration
                is raised from the inside of iter_setup(). The options `last_batch_policy`,
                `last_batch_padded` and `auto_reset` don't work in such case. It works with only
                one pipeline inside the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried for the shard size, number of shards, and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. Sets `last_batch_padded`
                accordingly to the reader's configuration (`pad_last_batch` reader argument)
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised and
                  reset() needs to be called. Calling ``iter()`` on the iterator would reset
                  it as well.
                * ``"yes"`` or ``True``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use `last_batch_policy` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with `last_batch_policy` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to False next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  -> last batch = ``[7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False -> last batch = ``[7]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
     next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``
    """

    def __init__(
        self,
        pipelines,
        size=-1,
        reader_name=None,
        auto_reset=False,
        fill_last_batch=None,
        last_batch_padded=False,
        last_batch_policy=LastBatchPolicy.FILL,
        prepare_first_batch=True,
    ):
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._num_gpus = len(pipelines)
        # frameworks expect from its data iterators to have batch_size field,
        # so it is not possible to use _batch_size instead
        self.batch_size = pipelines[0].max_batch_size
        assert np.all(
            np.equal([pipe.max_batch_size for pipe in pipelines], self.batch_size)
        ), "All pipelines should have the same batch size set"

        self._size = int(size)
        if auto_reset is False or auto_reset is None or auto_reset == "no":
            self._auto_reset = "no"
        elif auto_reset is True or auto_reset == "yes":
            self._auto_reset = "yes"
        else:
            raise ValueError(f"Unsupported value for `auto_reset` {auto_reset}")
        self._prepare_first_batch = prepare_first_batch

        if fill_last_batch is not None:
            warnings.warn(
                "Please do not use `fill_last_batch` and use `last_batch_policy` \
                           instead.",
                Warning,
                stacklevel=2,
            )
            if fill_last_batch:
                self._last_batch_policy = LastBatchPolicy.FILL
            else:
                self._last_batch_policy = LastBatchPolicy.PARTIAL
        else:
            if type(last_batch_policy) is not LastBatchPolicy:
                raise ValueError(
                    "Wrong type for `last_batch_policy`. "
                    f"Expected {LastBatchPolicy}, got {type(last_batch_policy)}"
                )
            self._last_batch_policy = last_batch_policy

        self._last_batch_padded = last_batch_padded
        assert self._size != 0, "Size cannot be 0"
        assert self._size > 0 or (
            self._size < 0 and (len(pipelines) == 1 or reader_name)
        ), "Negative size is supported only for a single pipeline"
        assert not reader_name or (
            reader_name and self._size < 0
        ), "When reader_name is provided, size should not be set"
        assert not reader_name or (
            reader_name and not last_batch_padded
        ), "When reader_name is provided, last_batch_padded should not be set"
        if self._size < 0 and not reader_name:
            self._last_batch_policy = LastBatchPolicy.FILL
            self._last_batch_padded = False
        if self.size > 0 and not reader_name:
            _iterator_deprecation_warning()
        self._pipes = pipelines
        self._counter = 0

        # Build all pipelines
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.build()

        self._reader_name = reader_name
        self._extract_from_reader_and_validate()
        self._ever_scheduled = False
        self._ever_consumed = False

        self._enable_checkpointing = self._pipes[0]._enable_checkpointing
        for p in self._pipes:
            if p._enable_checkpointing != self._enable_checkpointing:
                raise ValueError(
                    "All wrapped pipelines must have the same value for `enable_checkpointing`."
                )

        if self._enable_checkpointing:
            if any(p.is_restored_from_checkpoint for p in self._pipes):
                all_iterator_data = [p._iterator_data for p in self._pipes]
                if not all(p.is_restored_from_checkpoint for p in self._pipes):
                    logging.warning(
                        "Some, but not all of the pipelines used were restored from checkpoint. "
                        + "This iterator might produce unexpected results."
                    )
                elif not all(data == all_iterator_data[0] for data in all_iterator_data):
                    logging.warning(
                        "The provided pipelines had different iterator data in the checkpoints "
                        + "they were restored from. "
                        + "This iterator might produce unexpected results."
                    )
                iterator_data = all_iterator_data[0]
                self._restore_state(iterator_data)

            # Precompute the initial checkpoints, to prevent any problems
            # related to the `prepare_first_batch` flag.
            self._initial_checkpoints = [
                p._get_checkpoint(iterator_data=self._save_state()) for p in self._pipes
            ]

    def _checkpointed_fields(self):
        return [
            "_counter",
            "_counter_per_gpu",
            "_shard_sizes_per_gpu",
            "_shards_id",
            "_size",
        ]

    def _restore_state(self, iterator_data):
        """
        Restores state of the iterator based on serialized `iterator_data`
        """
        if not iterator_data:
            logging.warning(
                "Iterator data was not saved in the checkpoint. "
                "This iterator might produce unexpected results."
            )
            return

        iterator_data = pickle.loads(iterator_data)  # nosec B301
        for field in self._checkpointed_fields():
            if hasattr(self, field):
                setattr(self, field, iterator_data[field])

    def _save_state(self):
        iterator_data = pickle.dumps(
            {
                field: getattr(self, field)
                for field in self._checkpointed_fields()
                if hasattr(self, field)
            }
        )
        return iterator_data

    def _calculate_shard_sizes(self, shard_nums):
        shards_beg = np.floor(shard_nums * self._size_no_pad / self._shards_num)
        shards_end = np.floor((shard_nums + 1) * self._size_no_pad / self._shards_num)
        shards_beg = shards_beg.astype(np.int64)
        shards_end = shards_end.astype(np.int64)
        return shards_end - shards_beg

    def _extract_from_reader_and_validate(self):
        if self._reader_name:
            readers_meta = [p.reader_meta(self._reader_name) for p in self._pipes]

            def err_msg_gen(err_msg):
                return "Reader Operator should have the same {} in all the pipelines.".format(
                    err_msg
                )

            def check_equality_and_get(input_meta, name, err_msg):
                assert np.all(
                    np.equal([meta[name] for meta in input_meta], input_meta[0][name])
                ), err_msg_gen(err_msg)
                return input_meta[0][name]

            def check_all_or_none_and_get(input_meta, name, err_msg):
                assert np.all([meta[name] for meta in readers_meta]) or not np.any(
                    [meta[name] for meta in readers_meta]
                ), err_msg_gen(err_msg)
                return input_meta[0][name]

            self._size_no_pad = check_equality_and_get(readers_meta, "epoch_size", "size value")
            self._shards_num = check_equality_and_get(
                readers_meta, "number_of_shards", "`num_shards` argument set"
            )
            self._last_batch_padded = check_all_or_none_and_get(
                readers_meta, "pad_last_batch", "`pad_last_batch` argument set"
            )
            self._is_stick_to_shard = check_all_or_none_and_get(
                readers_meta, "stick_to_shard", "`stick_to_shard` argument set"
            )

            self._shards_id = np.array([meta["shard_id"] for meta in readers_meta], dtype=np.int64)

            if self._last_batch_policy == LastBatchPolicy.DROP:
                # when DROP policy is used round down the shard size
                self._size = self._size_no_pad // self._shards_num
            elif self._last_batch_padded:
                # if padding is enabled all shards are equal
                self._size = readers_meta[0]["epoch_size_padded"] // self._shards_num
            else:
                # get the size as a multiply of the batch size that is bigger or equal
                # than the biggest shard
                self._size = (
                    math.ceil(math.ceil(self._size_no_pad / self._shards_num) / self.batch_size)
                    * self.batch_size
                )

            # count where we starts inside each GPU shard in given epoch,
            # if shards are uneven this will differ epoch2epoch
            self._counter_per_gpu = np.zeros(self._shards_num, dtype=np.int64)
            self._shard_sizes_per_gpu = self._calculate_shard_sizes(np.arange(0, self._shards_num))

            # to avoid recalculation of shard sizes when iterator moves across the shards
            # memorize the initial shard sizes and then use changing self._shards_id to index it
            self._shard_sizes_per_gpu_initial = self._shard_sizes_per_gpu.copy()

    def _remove_padded(self):
        """
        Checks if remove any padded sample and how much.

        Calculates the number of padded samples in the batch for each pipeline
        wrapped up by the iterator. Returns if there is any padded data that
        needs to be dropped and if so how many samples in each GPU
        """
        if_drop = False
        left = -1
        if self._last_batch_policy == LastBatchPolicy.PARTIAL:
            # calculate each shard size for each id, and check how many samples are left
            # by subtracting from iterator counter the shard size, then go though all GPUs
            # and check how much data needs to be dropped
            left = self.batch_size - (
                self._counter - self._shard_sizes_per_gpu_initial[self._shards_id]
            )
            if_drop = np.less(left, self.batch_size)
        return if_drop, left

    def _get_outputs(self):
        """
        Checks iterator stop condition, gets DALI outputs and perform reset in case of StopIteration
        """
        # if pipeline was not scheduled ever do it here
        if not self._ever_scheduled:
            self._schedule_runs(False)
        if self._size > 0 and self._counter >= self._size:
            self._end_iteration()

        outputs = []
        try:
            for p in self._pipes:
                with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                    outputs.append(p.share_outputs())
        except StopIteration as e:
            # in case ExternalSource returns StopIteration
            if self._size < 0 and self._auto_reset == "yes":
                self.reset()
            raise e
        self._check_batch_size(outputs)
        return outputs

    def _check_batch_size(self, outs):
        if not isinstance(outs, Iterable):
            outs = [outs]
        if self._reader_name or self._size != -1:
            for out in outs:
                for o in out:
                    batch_len = len(o)
                    assert self.batch_size == batch_len, (
                        "Variable batch size is not supported by the iterator "
                        + "when reader_name is provided or iterator size is set explicitly"
                    )

    def _end_iteration(self):
        if self._auto_reset == "yes":
            self.reset()
        raise StopIteration

    def _schedule_runs(self, release_outputs=True):
        """
        Schedule DALI runs
        """
        self._ever_scheduled = True
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                if release_outputs:
                    p.release_outputs()
                p.schedule_run()

    def _advance_and_check_drop_last(self, dry_run=False, end_iteration=True):
        """
        Checks whether the current batch is not fully filled and whether it should be dropped.

        It could be dry run without changing the iterator state and not raising StopIteration
        """
        # check if for given initial count in any GPU with the current value of the samples read
        # if we read one more batch would we overflow
        counter = self._counter
        should_end = False
        if self._reader_name:
            counter += self.batch_size
            if self._last_batch_policy == LastBatchPolicy.DROP:
                should_end = np.any(self._counter_per_gpu + counter > self._shard_sizes_per_gpu)
        else:
            counter += self._num_gpus * self.batch_size
            if self._last_batch_policy == LastBatchPolicy.DROP:
                should_end = counter > self._size

        if not dry_run:
            self._counter = counter
            if should_end and end_iteration:
                self._end_iteration()

        return should_end

    def _report_no_data_in_pipeline(self):
        """
        Handles "no data in the pipeline" condition. If it's unexpected, raises an error.
        """

        # This might not be an error if we're iterating over pipeline that is
        # currently at the end of epoch, for example because it was restored from
        # checkpoint.
        if all(not p.is_restored_from_checkpoint or p._first_iter for p in self._pipes):
            raise RuntimeError(
                "It seems that there is no data in the pipeline. This may happen "
                "if `last_batch_policy` is set to PARTIAL and the requested batch size is "
                "greater than the shard size."
            )

    def checkpoints(self):
        """
        Returns the current checkpoints of the pipelines.
        """
        if not self._enable_checkpointing:
            raise ValueError("Cannot access checkpoints with checkpointing disabled")
        if not self._ever_consumed:
            return self._initial_checkpoints
        else:
            iterator_data = self._save_state()
            return [p._get_checkpoint(iterator_data=iterator_data) for p in self._pipes]

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        # in the case of the DROP policy the user who runs DALI, based on the iterator length,
        # can assume there is no more data in the pipeline where there still is the last,
        # incomplete batch, we need to extract from the pipeline and drop before rising
        # StopIteration indicating the pipeline is depleted. Here we first check if that
        # is the case, and if so we run the pipeline and drop the last batch
        if self._last_batch_policy == LastBatchPolicy.DROP:
            should_end = self._advance_and_check_drop_last(dry_run=True, end_iteration=False)
            already_ended = self._size > 0 and self._counter >= self._size
            if should_end and not already_ended:
                self._get_outputs()
                self._schedule_runs()
                self._advance_and_check_drop_last(end_iteration=False)

        if self._counter >= self._size or self._size < 0:
            if self._last_batch_policy == LastBatchPolicy.FILL and not self._last_batch_padded:
                if self._reader_name:
                    # accurate way
                    # get the number of samples read in this epoch by each GPU
                    # self._counter had initial value of min(self._counter_per_gpu) so subtract
                    # this to get the actual value
                    self._counter -= min(self._counter_per_gpu)
                    self._counter_per_gpu = self._counter_per_gpu + self._counter
                    # check how much each GPU read ahead from next shard, as shards have different
                    # size each epoch GPU may read ahead or not
                    self._counter_per_gpu = self._counter_per_gpu - self._shard_sizes_per_gpu
                    # to make sure that in the next epoch we read the whole shard we need
                    # to set start value to the smallest one
                    self._counter = min(self._counter_per_gpu)
                else:
                    # legacy way
                    self._counter = self._counter % self._size
            else:
                self._counter = 0
            # advance to the next shard
            if self._reader_name:
                if not self._is_stick_to_shard:
                    # move shards id for wrapped pipelines
                    self._shards_id = (self._shards_id + 1) % self._shards_num
                # revaluate _size
                if self._last_batch_policy == LastBatchPolicy.FILL and not self._last_batch_padded:
                    # move all shards ids GPU ahead
                    if not self._is_stick_to_shard:
                        self._shard_sizes_per_gpu = np.roll(self._shard_sizes_per_gpu, 1)
                    # check how many samples we need to reach from each shard in next epoch
                    # per each GPU taking into account already read
                    read_in_next_epoch = self._shard_sizes_per_gpu - self._counter_per_gpu
                    # get the maximum number of samples and round it up to full batch sizes
                    self._size = (
                        math.ceil(max(read_in_next_epoch) / self.batch_size) * self.batch_size
                    )
                    # in case some epoch is skipped because we have read ahead in this epoch so
                    # much that in the next one we done already
                    if self._size == 0:
                        # it means that self._shard_sizes_per_gpu == self._counter_per_gpu,
                        # so we can jump to the next epoch and zero self._counter_per_gpu
                        self._counter_per_gpu = np.zeros(self._shards_num, dtype=np.int64)
                        # self._counter = min(self._counter_per_gpu), but just set 0
                        # to make it simpler
                        self._counter = 0
                        # roll once again
                        self._shard_sizes_per_gpu = np.roll(self._shard_sizes_per_gpu, 1)
                        # as self._counter_per_gpu is 0 we can just use
                        # read_in_next_epoch = self._shard_sizes_per_gpu
                        self._size = (
                            math.ceil(max(self._shard_sizes_per_gpu) / self.batch_size)
                            * self.batch_size
                        )

            for p in self._pipes:
                p.reset()
                if p.empty():
                    with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                        p.schedule_run()
        else:
            logging.warning(
                "DALI iterator does not support resetting while epoch is not finished. \
                             Ignoring..."
            )

    def next(self):
        """
        Returns the next batch of data.
        """
        self._ever_consumed = True
        return self.__next__()

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        # avoid redundant reset when someone would call `iter()` on a new iterator
        # do not reset if no data was consumed from the iterator - to avoid unintended
        # buffering in the pipeline and the FW iterator
        if self._counter != 0 and self._ever_consumed:
            self.reset()
        return self

    @property
    def size(self):
        return self._size

    def __len__(self):
        if self._reader_name:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(self.size / self.batch_size)
            else:
                return self.size // self.batch_size
        else:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(self.size / (self._num_gpus * self.batch_size))
            else:
                return self.size // (self._num_gpus * self.batch_size)
