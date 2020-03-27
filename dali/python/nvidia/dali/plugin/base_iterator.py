# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

class _DaliBaseIterator(object):
    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=True,
                 last_batch_padded=False):

        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._num_gpus = len(pipelines)
        self.batch_size = pipelines[0].batch_size
        self._size = int(size)
        self._auto_reset = auto_reset

        self._fill_last_batch = fill_last_batch
        self._last_batch_padded = last_batch_padded
        assert self._size != 0, "Size cannot be 0"
        assert self._size > 0 or (self._size < 0 and (len(pipelines) == 1 or reader_name)), "Negative size is supported only for a single pipeline"
        assert not reader_name or (reader_name and self._size < 0), "When reader_name is provided, size should not be set"
        if self._size < 0 and not reader_name:
            self._auto_reset = False
            self._fill_last_batch = False
            self._last_batch_padded = False
        self._pipes = pipelines
        self._counter = 0
        # cont where we starts inside each GPU shard in given epoch,
        # if shards are uneven this will differ epoch2epoch
        self._counter_per_gpu = [0] * self._num_gpus

        # Build all pipelines
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.build()

        self._reader_name = reader_name
        if self._reader_name:
            self._size_no_pad = self._pipes[0].epoch_size(self._reader_name, False)
            assert np.all(np.equal([p.epoch_size(self._reader_name, False) for p in self._pipes], self._size_no_pad)), \
                "All pipelines readers should have the same size, check if they are reading the same data"

            self._shards_num = self._pipes[0].shards_number(self._reader_name)
            assert np.all(np.equal([p.shards_number(self._reader_name) for p in self._pipes], self._shards_num)), \
                "All pipelines readers should have the same shard number set"

            self._shards_id = [p.shard_id(self._reader_name) for p in self._pipes]

            assert np.all([p.if_reader_pads(self._reader_name) for p in self._pipes]) or \
                   not np.any([p.if_reader_pads(self._reader_name) for p in self._pipes]), \
                "All pipelines readers should have set padding in the same way"
            self._last_batch_padded = self._pipes[0].if_reader_pads(self._reader_name)

            assert np.all([p.if_sticks_to_shard(self._reader_name) for p in self._pipes]) or \
                   not np.any([p.if_sticks_to_shard(self._reader_name) for p in self._pipes]), \
                "All pipelines readers should have set stick to the shard in the same way"
            self._if_sticks_to_shard = self._pipes[0].if_sticks_to_shard(self._reader_name)

            if self._last_batch_padded:
                # if padding is enabled all shards are equal
                self._size = self._pipes[0].epoch_size(self._reader_name, True) // self._shards_num
            else:
                # get the size as a multiply of the batch size that is bigger or equal than the biggest shard
                self._size = math.ceil(math.ceil(self._size_no_pad / self._shards_num) / self.batch_size) * self.batch_size

    def _check_stop(self):
        """"
        Checks iterator stop condition and raise StopIteration if needed
        """
        if self._counter >= self._size and self._size > 0:
            if self._auto_reset:
                self.reset()
            raise StopIteration

    def _remove_padded(self):
        """
        Checks if remove any padded sample and how much
        """
        if_drop = False
        left = -1
        if not self._fill_last_batch:
            # calculate each shard size for each id, and check how many samples are left by substracting from iterator counter
            # the shard size, then go though all GPUs and return only relevant data by stripping padded one
            shards_beg = [int(id * self._size_no_pad / self._shards_num) for id in self._shards_id]
            shards_end = [int((id + 1) * self._size_no_pad / self._shards_num) for id in self._shards_id]
            left = [self._counter - (e - s) for e, s in zip(shards_end, shards_beg)]
            left = [self.batch_size - l for l in left]
            if_drop = np.less(left, self.batch_size)
        return if_drop, left

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        if self._counter >= self._size or self._size < 0:
            if self._fill_last_batch and not self._last_batch_padded:
                if self._reader_name:
                    # accurate way
                    # get the number of samples read in this epoch, self._counter started with min(self._counter_per_gpu) value
                    self._counter -= min(self._counter_per_gpu)
                    self._counter_per_gpu = [v + self._counter for v in self._counter_per_gpu]
                    shards_beg = [int(id * self._size_no_pad / self._shards_num) for id in self._shards_id]
                    shards_end = [int((id + 1) * self._size_no_pad / self._shards_num) for id in self._shards_id]
                    # check how much each GPU read ahead from next shard, as shards have different size each epoch
                    # GPU may read ahead or not
                    self._counter_per_gpu = [v - (e - b) for v, b, e in zip(self._counter_per_gpu, shards_beg, shards_end)]
                    self._counter = min(self._counter_per_gpu)
                else:
                    # legacy way
                    self._counter = self._counter % self._size
            else:
                self._counter = 0
            # advance to the next shard
            if self._reader_name and not self._if_sticks_to_shard:
                self._shards_id = [(v + 1) % self._shards_num for v in self._shards_id]
                if self._fill_last_batch and not self._last_batch_padded:
                    # revaluate _size
                    shards_beg = [int(id * self._size_no_pad / self._shards_num) for id in self._shards_id]
                    shards_end = [int((id + 1) * self._size_no_pad / self._shards_num) for id in self._shards_id]
                    self._size = math.ceil(max([e - v for e, v in zip(shards_end, shards_beg)]) / self.batch_size) * self.batch_size
            for p in self._pipes:
                p.reset()
                if p.empty():
                    with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                        p.schedule_run()
        else:
            logging.warning("DALI iterator does not support resetting while epoch is not finished. Ignoring...")

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__()

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    @property
    def size(self):
        return self._size