# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function, division
import math
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import numpy as np
import os
from test_utils import get_dali_extra_path

class COCOReaderPipeline(Pipeline):
    def __init__(self, data_paths, batch_size, num_threads, shard_id, num_gpus, random_shuffle, stick_to_shard, shuffle_after_epoch, pad_last_batch, initial_fill=1024):
        # use only 1 GPU, as we care only about shard_id
        super(COCOReaderPipeline, self).__init__(batch_size, num_threads, 0, prefetch_queue_depth=1)
        self.input = ops.COCOReader(file_root = data_paths[0], annotations_file=data_paths[1],
                                    shard_id = shard_id, num_shards = num_gpus, random_shuffle=random_shuffle,
                                    save_img_ids=True, stick_to_shard=stick_to_shard,shuffle_after_epoch=shuffle_after_epoch,
                                    pad_last_batch=pad_last_batch, initial_fill=initial_fill)

    def define_graph(self):
        images, bb, labels, ids = self.input(name="Reader")
        return ids

test_data_root = get_dali_extra_path()
coco_folder = os.path.join(test_data_root, 'db', 'coco')
data_sets = [[os.path.join(coco_folder, 'images'), os.path.join(coco_folder, 'instances.json')]]

def test_shuffling_patterns():
    for data_set in data_sets:
        #get reference ids
        ref_img_ids = []
        pipe = COCOReaderPipeline(batch_size=1, num_threads=4, shard_id=0, num_gpus=1, data_paths=data_set,
                                  random_shuffle=False, stick_to_shard=False, shuffle_after_epoch=False, pad_last_batch=False)
        pipe.build()
        iters = pipe.epoch_size("Reader")
        for j in range(iters):
            pipe._run()
            ref_img_ids.append(np.concatenate(pipe.outputs()[0].as_array()))
        ref_img_ids = set(np.concatenate(ref_img_ids))
        for num_gpus in [1, 2]:
            for batch_size in [1, 10, 100]:
                for stick_to_shard in [True, False]:
                    for shuffle_after_epoch in [True, False]:
                        for dry_run_num in [0, 1, 2]:
                            random_shuffle = not shuffle_after_epoch
                            pad_last_batch = batch_size != 1
                            pipes = [COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                        data_paths=data_set, random_shuffle=random_shuffle, stick_to_shard=stick_to_shard,
                                                        shuffle_after_epoch=shuffle_after_epoch, pad_last_batch=pad_last_batch) for gpu in range(num_gpus)]
                            if stick_to_shard and shuffle_after_epoch:
                                continue
                            [pipe.build() for pipe in pipes]
                            iters = pipes[0].epoch_size("Reader")
                            iters_tmp = iters
                            iters = iters // batch_size
                            if iters_tmp != iters * batch_size:
                                iters += 1
                            iters_tmp = iters

                            iters = iters // num_gpus
                            if iters_tmp != iters * num_gpus:
                                iters += 1

                            new_img_ids = []
                            # dry run
                            for j in range(iters * dry_run_num):
                                for pipe in pipes:
                                    pipe._run()
                                for pipe in pipes:
                                    pipe.outputs()
                            # get stats here
                            for j in range(iters):
                                for pipe in pipes:
                                    pipe._run()
                                for pipe in pipes:
                                    val = np.concatenate(pipe.outputs()[0].as_array())
                                    new_img_ids.append(val)
                            new_img_ids = set(np.concatenate(new_img_ids))
                            assert len(new_img_ids) == pipes[0].epoch_size("Reader")

                            yield check, data_set, num_gpus, batch_size, stick_to_shard, shuffle_after_epoch, dry_run_num


def gather_ids(pipes, iters, num_gpus):
    img_ids_list = [[] for i in range(num_gpus)]
    for _ in range(iters):
        for pipe in pipes:
            pipe._run()
        for pipe, new_img_ids in zip(pipes, img_ids_list):
            val = np.concatenate(pipe.outputs()[0].as_array())
            new_img_ids.append(val)

    set_list = []
    for elm in img_ids_list:
        set_list.append(set(np.concatenate(elm)))
    if num_gpus == 1:
        return img_ids_list[0], set_list[0]
    else:
        return img_ids_list, set_list

def test_global_shuffle_random_shuffle():
    num_gpus = 2
    batch_size = 1
    pipes = [COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                    data_paths=data_sets[0], random_shuffle=False, stick_to_shard=False,
                                                    shuffle_after_epoch=True, pad_last_batch=False) for gpu in range(num_gpus)]
    [pipe.build() for pipe in pipes]
    iters = pipes[0].epoch_size("Reader")

    _, img_ids_list_set = gather_ids(pipes, iters, num_gpus)
    _, img_ids_list_set_new = gather_ids(pipes, iters, num_gpus)

    assert img_ids_list_set[0] != img_ids_list_set_new[0]
    assert img_ids_list_set[1] != img_ids_list_set_new[1]
    assert img_ids_list_set[0].union(img_ids_list_set[1]) == img_ids_list_set_new[0].union(img_ids_list_set_new[1])


def test_global_shuffle_random_shuffle_2():
    num_gpus = 1
    batch_size = 1
    pipes = [COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=2,
                                                    data_paths=data_sets[0], random_shuffle=False, stick_to_shard=False,
                                                    shuffle_after_epoch=True, pad_last_batch=False, initial_fill=1) for gpu in range(num_gpus)]
    [pipe.build() for pipe in pipes]
    iters = pipes[0].epoch_size("Reader")
    iters = iters // 2

    img_ids_list, img_ids_list_set = gather_ids(pipes, iters, num_gpus)
    assert len(img_ids_list) == len(img_ids_list_set)

    img_ids_list_new, img_ids_list_new_set = gather_ids(pipes, iters, num_gpus)

    assert len(img_ids_list_new) == len(img_ids_list_new_set)
    assert len(img_ids_list_set.intersection(img_ids_list_new_set)) != 0


# with `random_shuffle=False` `shuffle_after_epoch=True` should still make data random between epochs
def test_global_shuffle_dont_mix_epochs():
    num_gpus = 2
    batch_size = 1
    pipes = [COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                    data_paths=data_sets[0], random_shuffle=False, stick_to_shard=False,
                                                    shuffle_after_epoch=True, pad_last_batch=False) for gpu in range(num_gpus)]
    [pipe.build() for pipe in pipes]
    iters = pipes[0].epoch_size("Reader")
    iters = iters // num_gpus

    _, img_ids_list_set = gather_ids(pipes, iters, num_gpus)
    _, img_ids_list_set_new = gather_ids(pipes, iters, num_gpus)

    assert img_ids_list_set[0] != img_ids_list_set_new[0]
    assert img_ids_list_set[1] != img_ids_list_set_new[1]
    assert img_ids_list_set[0].union(img_ids_list_set[1]) == img_ids_list_set_new[0].union(img_ids_list_set_new[1])


# with `random_shuffle=False` `shuffle_after_epoch=False` GPU0 data from epoch 0 should equal to data from GPU1 from epoch 1
def test_dont_mix_epochs():
    num_gpus = 2
    batch_size = 1
    pipes = [COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                    data_paths=data_sets[0], random_shuffle=False, stick_to_shard=False,
                                                    shuffle_after_epoch=False, pad_last_batch=False) for gpu in range(num_gpus)]
    [pipe.build() for pipe in pipes]
    iters = pipes[0].epoch_size("Reader")
    iters = iters // num_gpus

    _, img_ids_list_set = gather_ids(pipes, iters, num_gpus)
    _, img_ids_list_set_new = gather_ids(pipes, iters, num_gpus)

    assert img_ids_list_set[0] == img_ids_list_set_new[1]
    assert img_ids_list_set[1] == img_ids_list_set_new[0]


def create_pipeline(creator, batch_size, num_gpus):
    iters = 0
    #make sure that data size and batch are not divisible
    while iters % batch_size == 0:
        while iters != 0 and iters % batch_size == 0:
            batch_size += 1

        pipes = [creator(gpu) for gpu in range(num_gpus)]
        [pipe.build() for pipe in pipes]
        iters = pipes[0].epoch_size("Reader")
        iters = iters // num_gpus
    return pipes, iters


def test_pad_last_batch():
    num_gpus = 1
    batch_size = 100
    iters = 0

    pipes, iters = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=True), batch_size, num_gpus)
    data_size = iters

    iters_tmp = iters
    iters = iters // batch_size
    if iters_tmp != iters * batch_size:
        iters += 1
    iters_tmp = iters

    img_ids_list, _ = gather_ids(pipes, iters, num_gpus)

    img_ids_list = np.concatenate(img_ids_list)
    img_ids_list_set = set(img_ids_list)

    # check number of repeated samples
    remainder = int(math.ceil(data_size / batch_size)) * batch_size - data_size
    # check if repeated samples are equal to the last one
    mirrored_data = img_ids_list[-remainder - 1:]
    assert len(set(mirrored_data)) == 1
    assert len(img_ids_list) != len(img_ids_list_set)

    next_img_ids_list, _ = gather_ids(pipes, iters, num_gpus)

    next_img_ids_list = np.concatenate(next_img_ids_list)
    next_img_ids_list_set = set(next_img_ids_list)

    mirrored_data = next_img_ids_list[-remainder - 1:]
    assert len(set(mirrored_data)) == 1
    assert len(next_img_ids_list) != len(next_img_ids_list_set)


def check(data_set, num_gpus, batch_size, stick_to_shard, shuffle_after_epoch, dry_run_num):
    pass
