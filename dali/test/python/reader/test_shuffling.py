# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nose_utils  # noqa:F401
import math
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import numpy as np
import os
from test_utils import get_dali_extra_path


class COCOReaderPipeline(Pipeline):
    def __init__(
        self,
        data_paths,
        batch_size,
        num_threads,
        shard_id,
        num_gpus,
        random_shuffle,
        stick_to_shard,
        shuffle_after_epoch,
        pad_last_batch,
        initial_fill=1024,
    ):
        # use only 1 GPU, as we care only about shard_id
        super().__init__(batch_size, num_threads, 0, prefetch_queue_depth=1)
        self.input = ops.readers.COCO(
            file_root=data_paths[0],
            annotations_file=data_paths[1],
            shard_id=shard_id,
            num_shards=num_gpus,
            random_shuffle=random_shuffle,
            image_ids=True,
            stick_to_shard=stick_to_shard,
            shuffle_after_epoch=shuffle_after_epoch,
            pad_last_batch=pad_last_batch,
            initial_fill=initial_fill,
        )

    def define_graph(self):
        _, __, ___, ids = self.input(name="Reader")
        return ids


test_data_root = get_dali_extra_path()
coco_folder = os.path.join(test_data_root, "db", "coco")
datasets = [[os.path.join(coco_folder, "images"), os.path.join(coco_folder, "instances.json")]]


def test_shuffling_patterns():
    for dataset in datasets:
        # get reference ids
        ref_img_ids = []
        pipe = COCOReaderPipeline(
            batch_size=1,
            num_threads=4,
            shard_id=0,
            num_gpus=1,
            data_paths=dataset,
            random_shuffle=False,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=False,
        )
        iters = pipe.epoch_size("Reader")
        for _ in range(iters):
            pipe.schedule_run()
            ref_img_ids.append(np.concatenate(pipe.outputs()[0].as_array()))
        ref_img_ids = set(np.concatenate(ref_img_ids))
        for num_gpus in [1, 2, 3, 4]:
            for batch_size in [1, 10, 100]:
                for stick_to_shard in [True, False]:
                    for shuffle_after_epoch in [True, False]:
                        for dry_run_num in [0, 1, 2]:
                            yield (
                                check_shuffling_patterns,
                                dataset,
                                num_gpus,
                                batch_size,
                                stick_to_shard,
                                shuffle_after_epoch,
                                dry_run_num,
                                len(ref_img_ids),
                            )


def check_shuffling_patterns(
    dataset, num_gpus, batch_size, stick_to_shard, shuffle_after_epoch, dry_run_num, len_ref_img_ids
):
    random_shuffle = not shuffle_after_epoch
    pad_last_batch = batch_size != 1
    pipes = [
        COCOReaderPipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=dataset,
            random_shuffle=random_shuffle,
            stick_to_shard=stick_to_shard,
            shuffle_after_epoch=shuffle_after_epoch,
            pad_last_batch=pad_last_batch,
            initial_fill=1,
        )
        for gpu in range(num_gpus)
    ]

    if stick_to_shard and shuffle_after_epoch:
        return

    [pipe.build() for pipe in pipes]
    dataset_size = pipes[0].epoch_size("Reader")

    # dry run
    for j in range(dry_run_num):
        for n in range(num_gpus):
            mod = j
            if stick_to_shard or shuffle_after_epoch:
                mod = 0
            if pad_last_batch:
                shard_size = dataset_size // num_gpus
            else:
                shard_size = (
                    dataset_size * (n + 1 + mod) // num_gpus - dataset_size * (n + mod) // num_gpus
                )
            iters = shard_size // batch_size
            if shard_size != iters * batch_size:
                iters += 1
            for _ in range(iters):
                pipes[n].run()

    new_img_ids = []
    for n in range(num_gpus):
        mod = dry_run_num
        if stick_to_shard or shuffle_after_epoch:
            mod = 0
        if pad_last_batch:
            shard_size = dataset_size // num_gpus
        else:
            shard_size = (
                dataset_size * (n + 1 + mod) // num_gpus - dataset_size * (n + mod) // num_gpus
            )
        iters = shard_size // batch_size
        if shard_size != iters * batch_size:
            iters += 1
        for _ in range(iters):
            val = np.concatenate(pipes[n].run()[0].as_array())
            new_img_ids.append(val)
    new_img_ids = set(np.concatenate(new_img_ids))

    assert len(new_img_ids) == len_ref_img_ids


def gather_ids(pipes, epochs_run=0, batch_size=1, num_gpus_arg=None, gpus_arg=None):
    dataset_size = pipes[0].epoch_size("Reader")
    num_gpus = len(pipes)
    if num_gpus_arg:
        num_gpus = num_gpus_arg
    iterate_over = range(num_gpus)
    if gpus_arg:
        iterate_over = gpus_arg
    img_ids_list = [[] for _ in pipes]

    # Each GPU needs to iterate from `shard_id * data_size / num_gpus` samples
    # to `(shard_id + 1)* data_size / num_gpus`.
    # After each epoch, each GPU moves to the next shard.
    # The `epochs_run` variable  takes into account that after epoch readers advance to the
    # next shard. If shuffle_after_epoch or stick_to_shard is set, it doesn't matter
    # and could/should be 0; it is relevant only if pad_last_batch is False, otherwise each
    # shard has the same size due to padding.
    for img_ids_l, pipe, n in zip(img_ids_list, pipes, iterate_over):
        shard_size = (
            dataset_size * (n + 1 + epochs_run) // num_gpus
            - dataset_size * (n + epochs_run) // num_gpus
        )
        iters = int(math.ceil(shard_size / batch_size))
        for _ in range(iters):
            val = np.concatenate(pipe.run()[0].as_array())
            img_ids_l.append(val)

    set_list = []
    for elm in img_ids_list:
        set_list.append(set(np.concatenate(elm)))
    if len(pipes) == 1:
        return img_ids_list[0], set_list[0], epochs_run + 1
    else:
        return img_ids_list, set_list, epochs_run + 1


def test_global_shuffle_random_shuffle():
    num_gpus = 2
    batch_size = 1
    pipes = [
        COCOReaderPipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=datasets[0],
            random_shuffle=False,
            stick_to_shard=False,
            shuffle_after_epoch=True,
            pad_last_batch=False,
        )
        for gpu in range(num_gpus)
    ]

    _, img_ids_list_set, _ = gather_ids(pipes)
    _, img_ids_list_set_new, _ = gather_ids(pipes)

    assert img_ids_list_set[0] != img_ids_list_set_new[0]

    assert img_ids_list_set[1] != img_ids_list_set_new[1]

    assert img_ids_list_set[0].union(img_ids_list_set[1]) == img_ids_list_set_new[0].union(
        img_ids_list_set_new[1]
    )


def test_global_shuffle_random_shuffle_2():
    num_gpus = 1
    batch_size = 1
    pipes = [
        COCOReaderPipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=2,
            data_paths=datasets[0],
            random_shuffle=False,
            stick_to_shard=False,
            shuffle_after_epoch=True,
            pad_last_batch=False,
            initial_fill=1,
        )
        for gpu in range(num_gpus)
    ]

    img_ids_list, img_ids_list_set, _ = gather_ids(pipes, num_gpus_arg=2, gpus_arg=[0])
    assert len(img_ids_list) == len(img_ids_list_set)

    img_ids_list_new, img_ids_list_new_set, _ = gather_ids(pipes, num_gpus_arg=2, gpus_arg=[0])

    assert len(img_ids_list_new) == len(img_ids_list_new_set)
    assert len(img_ids_list_set.intersection(img_ids_list_new_set)) != 0


def test_global_shuffle_dont_mix_epochs():
    # with `random_shuffle=False` `shuffle_after_epoch=True` should
    # still make data random between epochs
    num_gpus = 2
    batch_size = 1
    pipes = [
        COCOReaderPipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=datasets[0],
            random_shuffle=False,
            stick_to_shard=False,
            shuffle_after_epoch=True,
            pad_last_batch=False,
        )
        for gpu in range(num_gpus)
    ]

    _, img_ids_list_set, _ = gather_ids(pipes)
    _, img_ids_list_set_new, _ = gather_ids(pipes)

    assert img_ids_list_set[0] != img_ids_list_set_new[0]
    assert img_ids_list_set[1] != img_ids_list_set_new[1]
    assert img_ids_list_set[0].union(img_ids_list_set[1]) == img_ids_list_set_new[0].union(
        img_ids_list_set_new[1]
    )


def test_dont_mix_epochs():
    # with `random_shuffle=False` `shuffle_after_epoch=False` GPU0 data
    # from epoch 0 should equal to data from GPU1 from epoch 1
    num_gpus = 2
    batch_size = 1
    pipes = [
        COCOReaderPipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=datasets[0],
            random_shuffle=False,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=False,
        )
        for gpu in range(num_gpus)
    ]

    _, img_ids_list_set, epochs_run = gather_ids(pipes)
    _, img_ids_list_set_new, _ = gather_ids(pipes, epochs_run)

    assert img_ids_list_set[0] == img_ids_list_set_new[1]
    assert img_ids_list_set[1] == img_ids_list_set_new[0]


def create_pipeline(creator, batch_size, num_gpus):
    iters = 0
    # make sure that data size and batch are not divisible
    while iters % batch_size == 0:
        while iters != 0 and iters % batch_size == 0:
            batch_size += 1

        pipes = [creator(gpu) for gpu in range(num_gpus)]
        [pipe.build() for pipe in pipes]
        iters = pipes[0].epoch_size("Reader")
        iters = iters // num_gpus
    return pipes, iters


def test_pad_last_batch_epoch_size():
    pipe = COCOReaderPipeline(
        batch_size=10,
        num_threads=4,
        shard_id=0,
        num_gpus=1,
        data_paths=datasets[0],
        random_shuffle=True,
        stick_to_shard=False,
        shuffle_after_epoch=False,
        pad_last_batch=True,
    )
    reference_size = pipe.epoch_size("Reader")

    for num_gpus in range(1, 10):
        pipe = COCOReaderPipeline(
            batch_size=10,
            num_threads=4,
            shard_id=0,
            num_gpus=num_gpus,
            data_paths=datasets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
        )
        size = pipe.epoch_size("Reader")
        print(reference_size, size, num_gpus)
        assert size == int(math.ceil(reference_size * 1.0 / num_gpus)) * num_gpus


def test_pad_last_batch():
    num_gpus = 1
    batch_size = 100

    pipes, iters = create_pipeline(
        lambda gpu: COCOReaderPipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=datasets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
        ),
        batch_size,
        num_gpus,
    )

    img_ids_list, _, epochs_run = gather_ids(pipes, batch_size=batch_size)

    img_ids_list = np.concatenate(img_ids_list)
    img_ids_list_set = set(img_ids_list)

    # check number of repeated samples
    remainder = int(math.ceil(iters * 1.0 / batch_size)) * batch_size - iters
    # check if repeated samples are equal to the last one
    mirrored_data = img_ids_list[-remainder - 1 :]
    print(iters, remainder, set(mirrored_data), img_ids_list)
    assert len(set(mirrored_data)) == 1
    assert len(img_ids_list) != len(img_ids_list_set)

    next_img_ids_list, _, _ = gather_ids(pipes, epochs_run, batch_size=batch_size)

    next_img_ids_list = np.concatenate(next_img_ids_list)
    next_img_ids_list_set = set(next_img_ids_list)

    mirrored_data = next_img_ids_list[-remainder - 1 :]
    print(set(mirrored_data))
    assert len(set(mirrored_data)) == 1
    assert len(next_img_ids_list) != len(next_img_ids_list_set)
