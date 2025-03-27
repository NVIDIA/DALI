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

import math
from nvidia.dali.pipeline import Pipeline, pipeline_def
from nvidia.dali.backend_impl import TensorCPU
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import numpy as np
import os
from test_utils import get_dali_extra_path
from nose_utils import raises, assert_raises, nottest, attr
from nose2.tools import params
from nvidia.dali.plugin.base_iterator import LastBatchPolicy as LastBatchPolicy
import random


def create_coco_pipeline(
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
    return_labels=False,
    exec_dynamic=False,
):
    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=0,
        prefetch_queue_depth=1,
        exec_dynamic=exec_dynamic,
    )
    with pipe:
        _, _, labels, ids = fn.readers.coco(
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
            name="Reader",
        )
        if return_labels:
            pipe.set_outputs(labels, ids)
        else:
            pipe.set_outputs(ids)
        return pipe


test_data_root = get_dali_extra_path()
coco_folder = os.path.join(test_data_root, "db", "coco")
data_sets = [[os.path.join(coco_folder, "images"), os.path.join(coco_folder, "instances.json")]]
image_data_set = os.path.join(test_data_root, "db", "single", "jpeg")


def gather_ids(dali_train_iter, data_getter, pad_getter, data_size):
    img_ids_list = []
    batch_size = dali_train_iter.batch_size
    pad = 0
    for it in iter(dali_train_iter):
        if not isinstance(it, dict):
            it = it[0]
        tmp = data_getter(it).copy()
        pad += pad_getter(it)
        img_ids_list.append(tmp)
    img_ids_list = np.concatenate(img_ids_list)
    img_ids_list_set = set(img_ids_list)

    remainder = int(math.ceil(data_size / batch_size)) * batch_size - data_size
    mirrored_data = img_ids_list[-remainder - 1 :]

    return img_ids_list, img_ids_list_set, mirrored_data, pad, remainder


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


@attr("mxnet")
def test_mxnet_iterator_model_fit():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    import mxnet as mx

    num_gpus = 1
    batch_size = 1

    def create_test_pipeline(batch_size, num_threads, device_id, num_gpus, data_paths):
        pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
        with pipe:
            _, labels = fn.readers.file(
                file_root=data_paths, shard_id=device_id, num_shards=num_gpus, name="Reader"
            )
        pipe.set_outputs(labels)
        return pipe

    pipes, _ = create_pipeline(
        lambda gpu: create_test_pipeline(
            batch_size=batch_size,
            num_threads=4,
            device_id=gpu,
            num_gpus=num_gpus,
            data_paths=image_data_set,
        ),
        batch_size,
        num_gpus,
    )
    pipe = pipes[0]

    class MXNetIteratorWrapper(MXNetIterator):
        def __init__(self, iter):
            self.iter = iter

        def __getattr__(self, attr):
            return getattr(self.iter, attr)

        def __next__(self):
            ret = self.iter.__next__()[0]
            return ret

    dali_train_iter = MXNetIterator(
        pipe, [("labels", MXNetIterator.LABEL_TAG)], size=pipe.epoch_size("Reader")
    )
    data = mx.symbol.Variable("labels")

    # create a dummy model
    _ = mx.model.FeedForward.create(
        data, X=MXNetIteratorWrapper(dali_train_iter), num_epoch=1, learning_rate=0.01
    )


@attr("mxnet")
def test_mxnet_iterator_last_batch_no_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=False,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = MXNetIterator(
        pipes,
        [("ids", MXNetIterator.DATA_TAG)],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.FILL,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x.data[0].squeeze(-1).asnumpy(), lambda x: x.pad, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1


@attr("mxnet")
def test_mxnet_iterator_empty_array():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    import mxnet as mx

    batch_size = 4
    size = 5

    all_np_types = [
        np.bool_,
        np.int_,
        np.intc,
        np.intp,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float16,
        np.short,
        int,
        np.longlong,
        np.ushort,
        np.ulonglong,
    ]
    np_types = []
    # store in np_types only types supported by MXNet
    for t in all_np_types:
        try:
            mx.nd.zeros([2, 2, 2], ctx=None, dtype=t)
            np_types.append(t)
        except mx.base.MXNetError:
            pass

    test_data_shape = [1, 3, 0, 4]

    def get_data():
        # create batch of [type_a, type_a, type_b, type_b, ...]
        out = [[np.empty(test_data_shape, dtype=t)] * batch_size for t in np_types]
        out = [val for pair in zip(out, out) for val in pair]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=0)
    outs = fn.external_source(source=get_data, num_outputs=len(np_types) * 2)
    pipe.set_outputs(*outs)

    # create map of [(data, type_a), (label, type_a), ...]
    data_map = [("data_{}".format(i), MXNetIterator.DATA_TAG) for i, t in enumerate(np_types)]
    label_map = [("label_{}".format(i), MXNetIterator.LABEL_TAG) for i, t in enumerate(np_types)]
    out_map = [val for pair in zip(data_map, label_map) for val in pair]

    iterator = MXNetIterator(pipe, output_map=out_map, size=size, dynamic_shape=True)

    for batch in iterator:
        for d, t in zip(batch[0].data, np_types):
            shape = d.asnumpy().shape
            assert shape[0] == batch_size
            assert np.array_equal(shape[1:], test_data_shape)
            assert d.asnumpy().dtype == t


@attr("mxnet")
def test_mxnet_iterator_last_batch_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = MXNetIterator(
        pipes,
        [("ids", MXNetIterator.DATA_TAG)],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.FILL,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x.data[0].squeeze(-1).asnumpy(), lambda x: x.pad, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x.data[0].squeeze(-1).asnumpy(), lambda x: x.pad, data_size
    )

    assert len(next_img_ids_list) > data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1


@attr("mxnet")
def test_mxnet_iterator_not_fill_last_batch_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = MXNetIterator(
        pipes,
        [("ids", MXNetIterator.DATA_TAG)],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )

    img_ids_list, img_ids_list_set, mirrored_data, pad, remainder = gather_ids(
        dali_train_iter, lambda x: x.data[0].squeeze(-1).asnumpy(), lambda x: x.pad, data_size
    )

    assert pad == remainder
    assert len(img_ids_list) - pad == data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, pad, remainder = gather_ids(
        dali_train_iter, lambda x: x.data[0].squeeze(-1).asnumpy(), lambda x: x.pad, data_size
    )

    assert pad == remainder
    assert len(next_img_ids_list) - pad == data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1


def check_iterator_results(
    pad,
    pipes_number,
    shards_num,
    out_set,
    last_batch_policy,
    img_ids_list,
    ids,
    data_set_size,
    sample_counter,
    per_gpu_counter,
    stick_to_shard,
    epoch_counter,
    rounded_shard_size,
):
    if pad and pipes_number == shards_num:
        assert len(set.intersection(*out_set)) == 0, "Shards should not overlaps in the epoch"
    if last_batch_policy == LastBatchPolicy.DROP:
        if pad:
            assert len(set.union(*out_set)) <= sum(
                [len(v) for v in img_ids_list]
            ), "Data returned from shard should not duplicate values"
        for id_list, id_set, id in zip(img_ids_list, out_set, ids):
            shard_size = int((id + 1) * data_set_size / shards_num) - int(
                id * data_set_size / shards_num
            )
            assert len(id_list) <= shard_size
            assert len(id_set) <= shard_size
    elif last_batch_policy == LastBatchPolicy.PARTIAL:
        if pad:
            assert len(set.union(*out_set)) == sum(
                [len(v) for v in img_ids_list]
            ), "Data returned from shard should not duplicate values"
        for id_list, id_set, id in zip(img_ids_list, out_set, ids):
            shard_size = int((id + 1) * data_set_size / shards_num) - int(
                id * data_set_size / shards_num
            )
            assert len(id_list) == shard_size
            assert len(id_set) == shard_size
    else:
        sample_counter -= min(per_gpu_counter)
        per_gpu_counter = [v + sample_counter for v in per_gpu_counter]

        if not stick_to_shard:
            shard_id_mod = epoch_counter
        else:
            shard_id_mod = 0
        shard_beg = [
            int(((id + shard_id_mod) % shards_num) * data_set_size / shards_num)
            for id in range(shards_num)
        ]
        shard_end = [
            int((((id + shard_id_mod) % shards_num) + 1) * data_set_size / shards_num)
            for id in range(shards_num)
        ]
        shard_sizes = [
            int((id + 1) * data_set_size / shards_num) - int(id * data_set_size / shards_num)
            for id in ids
        ]
        per_gpu_counter = [c - (e - b) for c, b, e in zip(per_gpu_counter, shard_beg, shard_end)]
        if pad:
            assert len(set.union(*out_set)) == sum(shard_sizes)
        for id_list, id_set, size in zip(img_ids_list, out_set, shard_sizes):
            if not pad:
                assert len(id_list) == sample_counter
            else:
                assert len(id_list) == rounded_shard_size
            if not stick_to_shard:
                if not pad:
                    assert len(id_list) == len(id_set)
                else:
                    assert len(id_list) == rounded_shard_size
                    assert len(id_set) == size
            else:
                assert len(id_set) == min(size, sample_counter)
        if not pad:
            sample_counter = min(per_gpu_counter)
        else:
            sample_counter = 0

    if not stick_to_shard:
        ids = [(id + 1) % shards_num for id in ids]
    epoch_counter += 1

    # these values are modified so return them
    return (ids, sample_counter, per_gpu_counter, epoch_counter, rounded_shard_size)


@attr("mxnet")
def check_mxnet_iterator_pass_reader_name(
    shards_num,
    pipes_number,
    batch_size,
    stick_to_shard,
    pad,
    iters,
    last_batch_policy,
    auto_reset=False,
):
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    pipes = [
        create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=id,
            num_gpus=shards_num,
            data_paths=data_sets[0],
            random_shuffle=False,
            stick_to_shard=stick_to_shard,
            shuffle_after_epoch=False,
            pad_last_batch=pad,
        )
        for id in range(pipes_number)
    ]

    data_set_size = pipes[0].reader_meta("Reader")["epoch_size"]
    rounded_shard_size = math.ceil(math.ceil(data_set_size / shards_num) / batch_size) * batch_size
    ids = [pipe.reader_meta("Reader")["shard_id"] for pipe in pipes]
    per_gpu_counter = [0] * shards_num
    epoch_counter = 0
    sample_counter = 0

    if batch_size > data_set_size // shards_num and last_batch_policy == LastBatchPolicy.DROP:
        assert_raises(
            RuntimeError,
            MXNetIterator,
            pipes,
            [("ids", MXNetIterator.DATA_TAG)],
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            glob="It seems that there is no data in the pipeline*last_batch_policy*",
        )
        return
    else:
        dali_train_iter = MXNetIterator(
            pipes,
            [("ids", MXNetIterator.DATA_TAG)],
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            auto_reset=auto_reset,
        )

    for _ in range(iters):
        out_set = []
        img_ids_list = [[] for _ in range(pipes_number)]
        orig_length = length = len(dali_train_iter)
        for it in iter(dali_train_iter):
            for id in range(pipes_number):
                tmp = it[id].data[0].squeeze(-1).asnumpy().copy()
                if it[id].pad:
                    tmp = tmp[0 : -it[id].pad]
                img_ids_list[id].append(tmp)
            sample_counter += batch_size
            length -= 1

        assert length == 0, (
            f"The iterator has reported the length of {orig_length} "
            f"but provided {orig_length - length} iterations."
        )
        if not auto_reset:
            dali_train_iter.reset()
        for id in range(pipes_number):
            img_ids_list[id] = np.concatenate(img_ids_list[id])
            out_set.append(set(img_ids_list[id]))

        ret = check_iterator_results(
            pad,
            pipes_number,
            shards_num,
            out_set,
            last_batch_policy,
            img_ids_list,
            ids,
            data_set_size,
            sample_counter,
            per_gpu_counter,
            stick_to_shard,
            epoch_counter,
            rounded_shard_size,
        )
        (ids, sample_counter, per_gpu_counter, epoch_counter, rounded_shard_size) = ret


@attr("mxnet")
def test_mxnet_iterator_pass_reader_name():
    for shards_num in [3, 5, 17]:
        for batch_size in [3, 5, 7]:
            for stick_to_shard in [False, True]:
                for pad in [True, False]:
                    for last_batch_policy in [
                        LastBatchPolicy.PARTIAL,
                        LastBatchPolicy.FILL,
                        LastBatchPolicy.DROP,
                    ]:
                        for iters in [1, 2, 3, 2 * shards_num]:
                            for pipes_number in [1, shards_num]:
                                yield (
                                    check_mxnet_iterator_pass_reader_name,
                                    shards_num,
                                    pipes_number,
                                    batch_size,
                                    stick_to_shard,
                                    pad,
                                    iters,
                                    last_batch_policy,
                                    False,
                                )


@attr("mxnet")
def test_mxnet_iterator_pass_reader_name_autoreset():
    for auto_reset in [True, False]:
        yield (
            check_mxnet_iterator_pass_reader_name,
            3,
            1,
            3,
            False,
            True,
            3,
            LastBatchPolicy.DROP,
            auto_reset,
        )


@attr("gluon")
def test_gluon_iterator_last_batch_no_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=False,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = GluonIterator(
        pipes, size=pipes[0].epoch_size("Reader"), last_batch_policy=LastBatchPolicy.FILL
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x[0].squeeze(-1).asnumpy(), lambda x: 0, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1


@attr("gluon")
def test_gluon_iterator_last_batch_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = GluonIterator(
        pipes, size=pipes[0].epoch_size("Reader"), last_batch_policy=LastBatchPolicy.FILL
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x[0].squeeze(-1).asnumpy(), lambda x: 0, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x[0].squeeze(-1).asnumpy(), lambda x: 0, data_size
    )

    assert len(next_img_ids_list) > data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1


@attr("gluon")
def test_gluon_iterator_not_fill_last_batch_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=False,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = GluonIterator(
        pipes, size=pipes[0].epoch_size("Reader"), last_batch_policy=LastBatchPolicy.PARTIAL
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x[0].squeeze(-1).asnumpy(), lambda x: 0, data_size
    )

    assert len(img_ids_list) == data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, pad, remainder = gather_ids(
        dali_train_iter, lambda x: x[0].squeeze(-1).asnumpy(), lambda x: 0, data_size
    )

    assert len(next_img_ids_list) == data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) != 1


@attr("gluon")
def test_gluon_iterator_sparse_batch():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator
    from mxnet.ndarray.ndarray import NDArray

    num_gpus = 1
    batch_size = 16

    pipes, _ = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
            return_labels=True,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = GluonIterator(
        pipes,
        pipes[0].epoch_size("Reader"),
        output_types=[GluonIterator.SPARSE_TAG, GluonIterator.DENSE_TAG],
        last_batch_policy=LastBatchPolicy.FILL,
    )

    for it in dali_train_iter:
        labels, ids = it[0]  # gpu 0
        # labels should be a sparse batch: a list of per-sample NDArray's
        # ids should be a dense batch: a single NDarray representing the batch
        assert isinstance(labels, (tuple, list))
        assert len(labels) == batch_size
        assert isinstance(labels[0], NDArray)
        assert isinstance(ids, NDArray)


@attr("gluon")
def check_gluon_iterator_pass_reader_name(
    shards_num,
    pipes_number,
    batch_size,
    stick_to_shard,
    pad,
    iters,
    last_batch_policy,
    auto_reset=False,
):
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    pipes = [
        create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=id,
            num_gpus=shards_num,
            data_paths=data_sets[0],
            random_shuffle=False,
            stick_to_shard=stick_to_shard,
            shuffle_after_epoch=False,
            pad_last_batch=pad,
        )
        for id in range(pipes_number)
    ]

    data_set_size = pipes[0].reader_meta("Reader")["epoch_size"]
    rounded_shard_size = math.ceil(math.ceil(data_set_size / shards_num) / batch_size) * batch_size
    ids = [pipe.reader_meta("Reader")["shard_id"] for pipe in pipes]
    per_gpu_counter = [0] * shards_num
    epoch_counter = 0
    sample_counter = 0

    if batch_size > data_set_size // shards_num and last_batch_policy == LastBatchPolicy.DROP:
        assert_raises(
            RuntimeError,
            GluonIterator,
            pipes,
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            glob="It seems that there is no data in the pipeline. This may happen "
            "if `last_batch_policy` is set to PARTIAL and the requested "
            "batch size is greater than the shard size.",
        )
        return
    else:
        dali_train_iter = GluonIterator(
            pipes, reader_name="Reader", last_batch_policy=last_batch_policy, auto_reset=auto_reset
        )

    for _ in range(iters):
        out_set = []
        img_ids_list = [[] for _ in range(pipes_number)]
        orig_length = length = len(dali_train_iter)
        for it in iter(dali_train_iter):
            for id in range(pipes_number):
                if len(it[id][0]):
                    tmp = it[id][0].squeeze(-1).asnumpy().copy()
                else:
                    tmp = np.empty([0])
                img_ids_list[id].append(tmp)
            sample_counter += batch_size
            length -= 1

        assert length == 0, (
            f"The iterator has reported the length of {orig_length} "
            f"but provided {orig_length - length} iterations."
        )
        if not auto_reset:
            dali_train_iter.reset()
        for id in range(pipes_number):
            assert (
                batch_size > data_set_size // shards_num
                and last_batch_policy == LastBatchPolicy.DROP
            ) or len(img_ids_list[id])
            if len(img_ids_list[id]):
                img_ids_list[id] = np.concatenate(img_ids_list[id])
                out_set.append(set(img_ids_list[id]))

        if len(out_set) == 0 and last_batch_policy == LastBatchPolicy.DROP:
            return

        ret = check_iterator_results(
            pad,
            pipes_number,
            shards_num,
            out_set,
            last_batch_policy,
            img_ids_list,
            ids,
            data_set_size,
            sample_counter,
            per_gpu_counter,
            stick_to_shard,
            epoch_counter,
            rounded_shard_size,
        )
        (ids, sample_counter, per_gpu_counter, epoch_counter, rounded_shard_size) = ret


@attr("gluon")
def test_gluon_iterator_pass_reader_name():
    for shards_num in [3, 5, 17]:
        for batch_size in [3, 5, 7]:
            for stick_to_shard in [False, True]:
                for pad in [True, False]:
                    for last_batch_policy in [
                        LastBatchPolicy.PARTIAL,
                        LastBatchPolicy.FILL,
                        LastBatchPolicy.DROP,
                    ]:
                        for iters in [1, 2, 3, 2 * shards_num]:
                            for pipes_number in [1, shards_num]:
                                yield (
                                    check_gluon_iterator_pass_reader_name,
                                    shards_num,
                                    pipes_number,
                                    batch_size,
                                    stick_to_shard,
                                    pad,
                                    iters,
                                    last_batch_policy,
                                    False,
                                )


@attr("gluon")
def test_gluon_iterator_pass_reader_name_autoreset():
    for auto_reset in [True, False]:
        yield (
            check_gluon_iterator_pass_reader_name,
            3,
            1,
            3,
            False,
            True,
            3,
            LastBatchPolicy.DROP,
            auto_reset,
        )


@attr("pytorch")
def test_pytorch_iterator_last_batch_no_pad_last_batch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=False,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = PyTorchIterator(
        pipes,
        output_map=["data"],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.FILL,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x["data"].squeeze(-1).numpy(), lambda x: 0, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1


@attr("pytorch")
def test_pytorch_iterator_last_batch_pad_last_batch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = PyTorchIterator(
        pipes,
        output_map=["data"],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.FILL,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x["data"].squeeze(-1).numpy(), lambda x: 0, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x["data"].squeeze(-1).numpy(), lambda x: 0, data_size
    )

    assert len(next_img_ids_list) > data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1


@attr("pytorch")
def test_pytorch_iterator_not_fill_last_batch_pad_last_batch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=False,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = PyTorchIterator(
        pipes,
        output_map=["data"],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.PARTIAL,
        last_batch_padded=True,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x["data"].squeeze(-1).numpy(), lambda x: 0, data_size
    )

    assert len(img_ids_list) == data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x["data"].squeeze(-1).numpy(), lambda x: 0, data_size
    )

    # there is no mirroring as data in the output is just cut off,
    # in the mirrored_data there is real data
    assert len(next_img_ids_list) == data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) != 1


@attr("jax")
def test_jax_iterator_last_batch_no_pad_last_batch():
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=False,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = JaxIterator(
        pipes,
        output_map=["data"],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.FILL,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x["data"].squeeze(-1), lambda x: 0, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1


@attr("jax")
@params((False,), (True,))
def test_jax_iterator_last_batch_pad_last_batch(exec_dynamic):
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
            exec_dynamic=exec_dynamic,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = JaxIterator(
        pipes,
        output_map=["data"],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.FILL,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x["data"].squeeze(-1), lambda x: 0, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: x["data"].squeeze(-1), lambda x: 0, data_size
    )

    assert len(next_img_ids_list) > data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1


def create_custom_pipeline(batch_size, num_threads, device_id, num_gpus, data_paths):
    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=0, prefetch_queue_depth=1
    )
    with pipe:
        jpegs, _ = fn.readers.file(
            file_root=data_paths, shard_id=device_id, num_shards=num_gpus, name="Reader"
        )
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.random_resized_crop(images, size=(224, 224))
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        pipe.set_outputs(images)
    return pipe


@attr("pytorch")
def test_pytorch_iterator_feed_ndarray():
    from nvidia.dali.plugin.pytorch import feed_ndarray as feed_ndarray
    import torch

    num_gpus = 1
    batch_size = 100
    pipes, _ = create_pipeline(
        lambda gpu: create_custom_pipeline(
            batch_size=batch_size,
            num_threads=4,
            device_id=gpu,
            num_gpus=num_gpus,
            data_paths=image_data_set,
        ),
        batch_size,
        num_gpus,
    )
    for gpu_id in range(num_gpus):
        pipe = pipes[gpu_id]
        outs = pipe.run()
        out_data = outs[0].as_tensor()
        device = torch.device("cuda", gpu_id)
        arr = torch.zeros(out_data.shape(), dtype=torch.float32, device=device)
        feed_ndarray(out_data, arr, cuda_stream=torch.cuda.current_stream(device=device))
        np.testing.assert_equal(arr.cpu().numpy(), outs[0].as_cpu().as_array())


@attr("pytorch")
def check_pytorch_iterator_feed_ndarray_types(data_type):
    from nvidia.dali.plugin.pytorch import feed_ndarray as feed_ndarray
    import torch

    to_torch_type = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.float16: torch.float16,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.bool_: torch.bool,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }

    shape = [3, 9]
    if np.issubdtype(data_type, np.integer):
        arr = np.random.randint(
            np.iinfo(data_type).min, high=np.iinfo(data_type).max, size=shape, dtype=data_type
        )
    elif data_type == np.bool_:
        arr = np.random.randint(0, high=2, size=shape, dtype=data_type)
    else:
        arr = np.random.randn(*shape).astype(data_type)
    tensor = TensorCPU(arr, "NHWC")
    pyt = torch.empty(shape, dtype=to_torch_type[data_type])
    feed_ndarray(tensor, pyt)
    assert np.all(pyt.numpy() == arr)


@attr("pytorch")
def test_pytorch_iterator_feed_ndarray_types():
    types = [
        np.float32,
        np.float64,
        np.float16,
        np.uint8,
        np.int8,
        np.bool_,
        np.int16,
        np.int32,
        np.int64,
    ]
    for data_type in types:
        yield check_pytorch_iterator_feed_ndarray_types, data_type


@params((False,), (True,))
def test_ragged_iterator_sparse_coo_batch(exec_dynamic):
    from nvidia.dali.plugin.pytorch import DALIRaggedIterator as RaggedIterator

    num_gpus = 1
    batch_size = 16

    pipes, _ = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
            return_labels=True,
            exec_dynamic=exec_dynamic,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = RaggedIterator(
        pipes,
        output_map=["labels", "ids"],
        size=pipes[0].epoch_size("Reader"),
        output_types=[RaggedIterator.SPARSE_COO_TAG, RaggedIterator.DENSE_TAG],
        last_batch_policy=LastBatchPolicy.FILL,
    )

    for it in dali_train_iter:
        labels, ids = it[0]["labels"], it[0]["ids"]  # gpu 0
        # labels should be a sparse coo batch: a sparse coo tensor
        # ids should be a dense batch: a single dense tensor
        assert len(labels) == batch_size
        assert labels.is_sparse is True
        assert ids.is_sparse is False


@params((False,), (True,))
def test_ragged_iterator_sparse_list_batch(exec_dynamic):
    from nvidia.dali.plugin.pytorch import DALIRaggedIterator as RaggedIterator

    num_gpus = 1
    batch_size = 16

    pipes, _ = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
            return_labels=True,
            exec_dynamic=exec_dynamic,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = RaggedIterator(
        pipes,
        output_map=["labels", "ids"],
        size=pipes[0].epoch_size("Reader"),
        output_types=[RaggedIterator.SPARSE_LIST_TAG, RaggedIterator.DENSE_TAG],
        last_batch_policy=LastBatchPolicy.FILL,
    )

    for it in dali_train_iter:
        labels, ids = it[0]["labels"], it[0]["ids"]  # gpu 0
        # labels should be a sparse list batch: a list of dense tensor
        # ids should be a dense batch: a single dense tensor
        assert len(labels) == batch_size
        assert isinstance(labels, list) is True
        assert ids.is_sparse is False


@attr("mxnet")
def test_mxnet_iterator_feed_ndarray():
    from nvidia.dali.plugin.mxnet import feed_ndarray as feed_ndarray
    import mxnet as mx

    num_gpus = 1
    batch_size = 100
    pipes, _ = create_pipeline(
        lambda gpu: create_custom_pipeline(
            batch_size=batch_size,
            num_threads=4,
            device_id=gpu,
            num_gpus=num_gpus,
            data_paths=image_data_set,
        ),
        batch_size,
        num_gpus,
    )
    for gpu_id in range(num_gpus):
        pipe = pipes[gpu_id]
        outs = pipe.run()
        out_data = outs[0].as_tensor()
        with mx.Context(mx.gpu(gpu_id)):
            arr = mx.nd.zeros(out_data.shape(), dtype=np.float32)
            mx.base._LIB.MXNDArrayWaitToWrite(arr.handle)
            # Using DALI's internal stream
            feed_ndarray(out_data, arr, cuda_stream=None)
            np.testing.assert_equal(arr.asnumpy(), outs[0].as_cpu().as_array())

            arr2 = mx.nd.zeros(out_data.shape(), dtype=np.float32)
            mx.base._LIB.MXNDArrayWaitToWrite(arr2.handle)
            feed_ndarray(out_data, arr2, cuda_stream=0)  # Using default stream
            np.testing.assert_equal(arr2.asnumpy(), outs[0].as_cpu().as_array())


@attr("mxnet")
def check_mxnet_iterator_feed_ndarray_types(data_type):
    from nvidia.dali.plugin.mxnet import feed_ndarray as feed_ndarray
    import mxnet as mx

    shape = [3, 9]
    if np.issubdtype(data_type, np.integer):
        arr = np.random.randint(
            np.iinfo(data_type).min, high=np.iinfo(data_type).max, size=shape, dtype=data_type
        )
    elif data_type == np.bool_:
        arr = np.random.randint(0, high=2, size=shape, dtype=data_type)
    else:
        arr = np.random.randn(*shape).astype(data_type)
    tensor = TensorCPU(arr, "NHWC")
    mnt = mx.nd.empty(shape, dtype=data_type)
    feed_ndarray(tensor, mnt)
    assert np.all(mnt.asnumpy() == arr)


@attr("mxnet")
def test_mxnet_iterator_feed_ndarray_types():
    # MXNet doesn't support int16
    types = [np.float32, np.float64, np.float16, np.uint8, np.int8, np.bool_, np.int32, np.int64]
    for data_type in types:
        yield check_mxnet_iterator_feed_ndarray_types, data_type


@attr("paddle")
def test_paddle_iterator_feed_ndarray():
    from nvidia.dali.plugin.paddle import feed_ndarray as feed_ndarray
    import paddle

    num_gpus = 1
    batch_size = 100
    pipes, _ = create_pipeline(
        lambda gpu: create_custom_pipeline(
            batch_size=batch_size,
            num_threads=4,
            device_id=gpu,
            num_gpus=num_gpus,
            data_paths=image_data_set,
        ),
        batch_size,
        num_gpus,
    )
    for gpu_id in range(num_gpus):
        pipe = pipes[gpu_id]
        outs = pipe.run()
        out_data = outs[0].as_tensor()

        lod_tensor = paddle.framework.core.LoDTensor()
        lod_tensor._set_dims(out_data.shape())
        gpu_place = paddle.CUDAPlace(gpu_id)

        ptr = lod_tensor._mutable_data(gpu_place, paddle.framework.core.VarDesc.VarType.FP32)
        np.array(lod_tensor)
        # Using DALI's internal stream
        feed_ndarray(out_data, ptr, cuda_stream=None)
        np.testing.assert_equal(np.array(lod_tensor), outs[0].as_cpu().as_array())

        lod_tensor2 = paddle.framework.core.LoDTensor()
        lod_tensor2._set_dims(out_data.shape())

        ptr2 = lod_tensor2._mutable_data(gpu_place, paddle.framework.core.VarDesc.VarType.FP32)
        np.array(lod_tensor2)
        feed_ndarray(out_data, ptr2, cuda_stream=0)  # Using default stream
        np.testing.assert_equal(np.array(lod_tensor2), outs[0].as_cpu().as_array())


@attr("paddle")
def check_paddle_iterator_feed_ndarray_types(data_type):
    from nvidia.dali.plugin.paddle import feed_ndarray as feed_ndarray
    import paddle

    dtype_map = {
        np.bool_: paddle.framework.core.VarDesc.VarType.BOOL,
        np.float32: paddle.framework.core.VarDesc.VarType.FP32,
        np.float64: paddle.framework.core.VarDesc.VarType.FP64,
        np.float16: paddle.framework.core.VarDesc.VarType.FP16,
        np.uint8: paddle.framework.core.VarDesc.VarType.UINT8,
        np.int8: paddle.framework.core.VarDesc.VarType.INT8,
        np.int16: paddle.framework.core.VarDesc.VarType.INT16,
        np.int32: paddle.framework.core.VarDesc.VarType.INT32,
        np.int64: paddle.framework.core.VarDesc.VarType.INT64,
    }

    shape = [3, 9]
    if np.issubdtype(data_type, np.integer):
        arr = np.random.randint(
            np.iinfo(data_type).min, high=np.iinfo(data_type).max, size=shape, dtype=data_type
        )
    elif data_type == np.bool_:
        arr = np.random.randint(0, high=2, size=shape, dtype=data_type)
    else:
        arr = np.random.randn(*shape).astype(data_type)
    tensor = TensorCPU(arr, "NHWC")
    pddt = paddle.framework.core.LoDTensor()
    pddt._set_dims(shape)
    ptr = pddt._mutable_data(paddle.CPUPlace(), dtype_map[data_type])
    feed_ndarray(tensor, ptr)
    assert np.all(np.array(pddt) == arr)


@attr("paddle")
def test_paddle_iterator_feed_ndarray_types():
    types = [
        np.float32,
        np.float64,
        np.float16,
        np.uint8,
        np.int8,
        np.bool_,
        np.int16,
        np.int32,
        np.int64,
    ]
    for data_type in types:
        yield check_paddle_iterator_feed_ndarray_types, data_type


@attr("pytorch")
def check_pytorch_iterator_pass_reader_name(
    shards_num,
    pipes_number,
    batch_size,
    stick_to_shard,
    pad,
    exec_dynamic,
    iters,
    last_batch_policy,
    auto_reset=False,
):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    pipes = [
        create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=id,
            num_gpus=shards_num,
            data_paths=data_sets[0],
            random_shuffle=False,
            stick_to_shard=stick_to_shard,
            shuffle_after_epoch=False,
            pad_last_batch=pad,
            exec_dynamic=exec_dynamic,
        )
        for id in range(pipes_number)
    ]

    data_set_size = pipes[0].reader_meta("Reader")["epoch_size"]
    rounded_shard_size = math.ceil(math.ceil(data_set_size / shards_num) / batch_size) * batch_size
    ids = [pipe.reader_meta("Reader")["shard_id"] for pipe in pipes]
    per_gpu_counter = [0] * shards_num
    epoch_counter = 0
    sample_counter = 0

    if batch_size > data_set_size // shards_num and last_batch_policy == LastBatchPolicy.DROP:
        assert_raises(
            RuntimeError,
            PyTorchIterator,
            pipes,
            output_map=["data"],
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            glob="It seems that there is no data in the pipeline. This may happen "
            "if `last_batch_policy` is set to PARTIAL and the requested batch size "
            "is greater than the shard size.",
        )
        return
    else:
        dali_train_iter = PyTorchIterator(
            pipes,
            output_map=["data"],
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            auto_reset=auto_reset,
        )

    for _ in range(iters):
        out_set = []
        img_ids_list = [[] for _ in range(pipes_number)]
        orig_length = length = len(dali_train_iter)
        for it in iter(dali_train_iter):
            for id in range(pipes_number):
                tmp = it[id]["data"].squeeze(dim=1).numpy().copy()
                img_ids_list[id].append(tmp)
            sample_counter += batch_size
            length -= 1

        assert length == 0, (
            f"The iterator has reported the length of {orig_length} "
            f"but provided {orig_length - length} iterations."
        )
        if not auto_reset:
            dali_train_iter.reset()
        for id in range(pipes_number):
            img_ids_list[id] = np.concatenate(img_ids_list[id])
            out_set.append(set(img_ids_list[id]))

        ret = check_iterator_results(
            pad,
            pipes_number,
            shards_num,
            out_set,
            last_batch_policy,
            img_ids_list,
            ids,
            data_set_size,
            sample_counter,
            per_gpu_counter,
            stick_to_shard,
            epoch_counter,
            rounded_shard_size,
        )
        (ids, sample_counter, per_gpu_counter, epoch_counter, rounded_shard_size) = ret


@attr("pytorch")
def test_pytorch_iterator_pass_reader_name():
    for shards_num in [3, 5, 17]:
        for batch_size in [3, 5, 7]:
            for stick_to_shard in [False, True]:
                for pad in [True, False]:
                    for last_batch_policy in [
                        LastBatchPolicy.PARTIAL,
                        LastBatchPolicy.FILL,
                        LastBatchPolicy.DROP,
                    ]:
                        for exec_dynamic in [False, True]:
                            for pipes_number in [1, shards_num]:
                                yield (
                                    check_pytorch_iterator_pass_reader_name,
                                    shards_num,
                                    pipes_number,
                                    batch_size,
                                    stick_to_shard,
                                    pad,
                                    exec_dynamic,
                                    3 * shards_num,
                                    last_batch_policy,
                                    False,
                                )


@attr("pytorch")
def test_pytorch_iterator_pass_reader_name_autoreset():
    for auto_reset in [True, False]:
        yield (
            check_pytorch_iterator_pass_reader_name,
            3,
            1,
            3,
            False,
            True,
            False,
            3,
            LastBatchPolicy.DROP,
            auto_reset,
        )


@attr("paddle")
def test_paddle_iterator_last_batch_no_pad_last_batch():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=False,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = PaddleIterator(
        pipes,
        output_map=["data"],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.FILL,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1


@attr("paddle")
@params((False,), (True,))
def test_paddle_iterator_last_batch_pad_last_batch(exec_dynamic):
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
            exec_dynamic=exec_dynamic,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = PaddleIterator(
        pipes,
        output_map=["data"],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.FILL,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size
    )

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size
    )

    assert len(next_img_ids_list) > data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1


@attr("paddle")
def test_paddle_iterator_not_fill_last_batch_pad_last_batch():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=False,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=True,
        ),
        batch_size,
        num_gpus,
    )

    dali_train_iter = PaddleIterator(
        pipes,
        output_map=["data"],
        size=pipes[0].epoch_size("Reader"),
        last_batch_policy=LastBatchPolicy.PARTIAL,
        last_batch_padded=True,
    )

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size
    )

    assert len(img_ids_list) == data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = gather_ids(
        dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size
    )

    # there is no mirroring as data in the output is just cut off,
    # in the mirrored_data there is real data
    assert len(next_img_ids_list) == data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) != 1


@attr("paddle")
def check_paddle_iterator_pass_reader_name(
    shards_num,
    pipes_number,
    batch_size,
    stick_to_shard,
    pad,
    iters,
    last_batch_policy,
    auto_reset=False,
    exec_dynamic=False,
):
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    pipes = [
        create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=id,
            num_gpus=shards_num,
            data_paths=data_sets[0],
            random_shuffle=False,
            stick_to_shard=stick_to_shard,
            shuffle_after_epoch=False,
            pad_last_batch=pad,
            exec_dynamic=exec_dynamic,
        )
        for id in range(pipes_number)
    ]

    data_set_size = pipes[0].reader_meta("Reader")["epoch_size"]
    rounded_shard_size = math.ceil(math.ceil(data_set_size / shards_num) / batch_size) * batch_size
    ids = [pipe.reader_meta("Reader")["shard_id"] for pipe in pipes]
    per_gpu_counter = [0] * shards_num
    epoch_counter = 0
    sample_counter = 0

    if batch_size > data_set_size // shards_num and last_batch_policy == LastBatchPolicy.DROP:
        assert_raises(
            RuntimeError,
            PaddleIterator,
            pipes,
            output_map=["data"],
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            glob="It seems that there is no data in the pipeline. This may happen "
            "if `last_batch_policy` is set to PARTIAL and the requested batch size "
            "is greater than the shard size.",
        )
        return
    else:
        dali_train_iter = PaddleIterator(
            pipes,
            output_map=["data"],
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            auto_reset=auto_reset,
        )

    for _ in range(iters):
        out_set = []
        img_ids_list = [[] for _ in range(pipes_number)]
        orig_length = length = len(dali_train_iter)
        for it in iter(dali_train_iter):
            for id in range(pipes_number):
                tmp = np.array(it[id]["data"]).squeeze(axis=1).copy()
                img_ids_list[id].append(tmp)
            sample_counter += batch_size
            length -= 1

        assert length == 0, (
            f"The iterator has reported the length of {orig_length} "
            f"but provided {orig_length - length} iterations."
        )
        if not auto_reset:
            dali_train_iter.reset()
        for id in range(pipes_number):
            img_ids_list[id] = np.concatenate(img_ids_list[id])
            out_set.append(set(img_ids_list[id]))

        ret = check_iterator_results(
            pad,
            pipes_number,
            shards_num,
            out_set,
            last_batch_policy,
            img_ids_list,
            ids,
            data_set_size,
            sample_counter,
            per_gpu_counter,
            stick_to_shard,
            epoch_counter,
            rounded_shard_size,
        )
        (ids, sample_counter, per_gpu_counter, epoch_counter, rounded_shard_size) = ret


@attr("paddle")
def test_paddle_iterator_pass_reader_name():
    for shards_num in [3, 5, 17]:
        for batch_size in [3, 5, 7]:
            for stick_to_shard in [False, True]:
                for pad in [True, False]:
                    for last_batch_policy in [
                        LastBatchPolicy.PARTIAL,
                        LastBatchPolicy.FILL,
                        LastBatchPolicy.DROP,
                    ]:
                        for iters in [1, max(3, 2 * shards_num)]:
                            for exec_dynamic in [False, True]:
                                for pipes_number in [1, shards_num]:
                                    yield (
                                        check_paddle_iterator_pass_reader_name,
                                        shards_num,
                                        pipes_number,
                                        batch_size,
                                        stick_to_shard,
                                        pad,
                                        iters,
                                        last_batch_policy,
                                        False,
                                        exec_dynamic,
                                    )


@attr("paddle")
def test_paddle_iterator_pass_reader_name_autoreset():
    for auto_reset in [True, False]:
        yield (
            check_paddle_iterator_pass_reader_name,
            3,
            1,
            3,
            False,
            True,
            3,
            LastBatchPolicy.DROP,
            auto_reset,
        )


class TestIterator:
    def __init__(self, iters_per_epoch, batch_size, total_iter_num=-1):
        self.n = iters_per_epoch
        self.total_n = total_iter_num
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        batch = []
        # setting -1 means that no total iteration limit is set
        if self.i < self.n and self.total_n != 0:
            batch = [np.arange(0, 10, dtype=np.uint8) for _ in range(self.batch_size)]
            self.i += 1
            self.total_n -= 1
            return batch
        else:
            self.i = 0
            raise StopIteration

    next = __next__

    @property
    def size(
        self,
    ):
        return self.n * self.batch_size


@nottest
def create_test_iter_pipeline(batch_size, device_id, data_source, num_threads=4):
    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=0, prefetch_queue_depth=1
    )
    with pipe:
        outs = fn.external_source(source=data_source)
    pipe.set_outputs(outs)
    return pipe


def check_stop_iter(
    fw_iter, iterator_name, batch_size, epochs, iter_num, total_iter_num, auto_reset, infinite
):
    it = TestIterator(iter_num, batch_size, total_iter_num)
    pipe = create_test_iter_pipeline(batch_size, 0, it)
    if infinite:
        iter_size = -1
    else:
        iter_size = it.size
    loader = fw_iter(pipe, iter_size, auto_reset)
    count = 0
    for _ in range(epochs):
        for _ in enumerate(loader):
            count += 1
        if not auto_reset:
            loader.reset()
    if total_iter_num < 0:
        # infinite source of data
        assert count == iter_num * epochs
    else:
        # at most total_iter_num should be returned by the iterator
        assert count == min(
            total_iter_num, iter_num * epochs
        ), f"{count}, {total_iter_num}, {iter_num} * {epochs} == {iter_num * epochs}"


@raises(Exception, glob="Negative size is supported only for a single pipeline")
def check_stop_iter_fail_multi(fw_iter):
    batch_size = 1
    iter_num = 1
    pipes = [
        create_test_iter_pipeline(batch_size, 0, TestIterator(iter_num, batch_size))
        for _ in range(2)
    ]
    fw_iter(pipes, -1, False)


@raises(Exception, glob="Size cannot be 0")
def check_stop_iter_fail_single(fw_iter):
    batch_size = 1
    iter_num = 1
    pipes = [
        create_test_iter_pipeline(batch_size, 0, TestIterator(iter_num, batch_size))
        for _ in range(1)
    ]
    fw_iter(pipes, 0, False)


def stop_iteration_case_generator():
    for epochs in [1, 3, 6]:
        for iter_num in [1, 2, 9]:
            for total_iters in [-1, iter_num - 1, 2 * iter_num - 1]:
                if total_iters == 0 or total_iters > epochs * iter_num:
                    continue
                for batch_size in [1, 10, 100]:
                    for auto_reset in [True, False]:
                        for infinite in [False, True]:
                            args = (batch_size, epochs, iter_num, total_iters, auto_reset, infinite)
                            yield args


def check_iterator_wrapper_first_iteration(BaseIterator, *args, **kwargs):
    # This wrapper is used to test that the base class iterator doesn't invoke
    # the wrapper self.__next__ function accidentally
    class IteratorWrapper(BaseIterator):
        def __init__(self, *args, **kwargs):
            self._allow_next = False
            super(IteratorWrapper, self).__init__(*args, **kwargs)

        # Asserting if __next__ is called, unless self._allow_next has been set to True explicitly
        def __next__(self):
            assert self._allow_next
            _ = super(IteratorWrapper, self).__next__()

    pipe = Pipeline(batch_size=16, num_threads=1, device_id=0)
    with pipe:
        data = fn.random.uniform(range=(-1, 1), shape=(2, 2, 2), seed=1234)
    pipe.set_outputs(data)

    iterator_wrapper = IteratorWrapper([pipe], *args, **kwargs)
    # Only now, we allow the wrapper __next__ to run
    iterator_wrapper._allow_next = True
    for i, _ in enumerate(iterator_wrapper):
        if i == 2:
            break


def check_external_source_autoreset(Iterator, *args, to_np=None, **kwargs):
    max_batch_size = 4
    iter_limit = 4
    runs = 3
    test_data_shape = [2, 3, 4]
    i = 0
    dataset = [
        [
            [
                np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
                for _ in range(max_batch_size)
            ]
        ]
        for _ in range(iter_limit)
    ]

    def get_data():
        nonlocal i
        if i == iter_limit:
            i = 0
            raise StopIteration
        out = dataset[i]
        i += 1
        return out

    pipe = Pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    with pipe:
        outs = fn.external_source(source=get_data, num_outputs=1)
    pipe.set_outputs(*outs)

    it = Iterator([pipe], *args, auto_reset=True, **kwargs)
    counter = 0
    for _ in range(runs):
        for j, data in enumerate(it):
            assert (to_np(data) == np.concatenate(dataset[j])).all()
            counter += 1
    assert counter == iter_limit * runs


def check_external_source_variable_size(
    Iterator, *args, iter_size=-1, to_np=None, exec_dynamic=False, **kwargs
):
    max_batch_size = 1
    iter_limit = 4
    runs = 3
    test_data_shape = [2, 3, 4]
    i = 0
    dataset = [
        [
            [
                np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
                for _ in range(random.randint(1, max_batch_size))
            ]
        ]
        for _ in range(iter_limit)
    ]

    def get_data():
        nonlocal i
        if i == iter_limit:
            i = 0
            raise StopIteration
        out = dataset[i]
        i += 1
        return out

    pipe = Pipeline(
        batch_size=max_batch_size, num_threads=1, device_id=0, exec_dynamic=exec_dynamic
    )
    with pipe:
        outs = fn.external_source(source=get_data, num_outputs=1)
    pipe.set_outputs(*outs)

    it = Iterator([pipe], *args, auto_reset=True, size=iter_size, **kwargs)
    counter = 0
    for _ in range(runs):
        for j, data in enumerate(it):
            assert (to_np(data[0]) == np.concatenate(dataset[j])).all()
            counter += 1
    assert counter == iter_limit * runs


# MXNet


@attr("mxnet")
def test_stop_iteration_mxnet():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    def fw_iter(pipe, size, auto_reset):
        return MXNetIterator(
            pipe, [("data", MXNetIterator.DATA_TAG)], size=size, auto_reset=auto_reset
        )

    iter_name = "MXNetIterator"
    for (
        batch_size,
        epochs,
        iter_num,
        total_iter_num,
        auto_reset,
        infinite,
    ) in stop_iteration_case_generator():
        check_stop_iter(
            fw_iter, iter_name, batch_size, epochs, iter_num, total_iter_num, auto_reset, infinite
        )


@attr("mxnet")
def test_stop_iteration_mxnet_fail_multi():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    def fw_iter(pipe, size, auto_reset):
        return MXNetIterator(
            pipe, [("data", MXNetIterator.DATA_TAG)], size=size, auto_reset=auto_reset
        )

    check_stop_iter_fail_multi(fw_iter)


@attr("mxnet")
def test_stop_iteration_mxnet_fail_single():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    def fw_iter(pipe, size, auto_reset):
        return MXNetIterator(
            pipe, [("data", MXNetIterator.DATA_TAG)], size=size, auto_reset=auto_reset
        )

    check_stop_iter_fail_single(fw_iter)


@attr("mxnet")
def test_mxnet_iterator_wrapper_first_iteration():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    check_iterator_wrapper_first_iteration(
        MXNetIterator, [("data", MXNetIterator.DATA_TAG)], size=100
    )


@attr("mxnet")
def test_mxnet_external_source_autoreset():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    check_external_source_autoreset(
        MXNetIterator, [("data", MXNetIterator.DATA_TAG)], to_np=lambda x: x[0].data[0].asnumpy()
    )


@attr("mxnet")
def test_mxnet_external_source_do_not_prepare():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    check_external_source_autoreset(
        MXNetIterator,
        [("data", MXNetIterator.DATA_TAG)],
        to_np=lambda x: x[0].data[0].asnumpy(),
        prepare_first_batch=False,
    )


@attr("mxnet")
def check_mxnet_iterator_properties(prepare_ahead):
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    def data_to_np(x):
        return x.data[0].asnumpy()

    def label_to_np(x):
        return x.label[0].asnumpy()

    max_batch_size = 4
    iter_limit = 4
    runs = 3
    test_data_shape = [2, 3, 4]
    test_label_shape = [2, 7, 5]
    i = 0
    dataset = [
        [
            [
                np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
                for _ in range(max_batch_size)
            ],
            [
                np.random.randint(0, 255, size=test_label_shape, dtype=np.uint8)
                for _ in range(max_batch_size)
            ],
        ]
        for _ in range(iter_limit)
    ]

    def get_data():
        nonlocal i
        if i == iter_limit:
            i = 0
            raise StopIteration
        out = dataset[i]
        i += 1
        return out

    pipe = Pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    with pipe:
        outs = fn.external_source(source=get_data, num_outputs=2)
    pipe.set_outputs(*outs)

    it = MXNetIterator(
        [pipe],
        [("data", MXNetIterator.DATA_TAG), ("label", MXNetIterator.LABEL_TAG)],
        auto_reset=True,
        prepare_first_batch=prepare_ahead,
    )
    counter = 0
    assert getattr(it, "provide_data")[0].shape == tuple([max_batch_size] + test_data_shape)
    assert getattr(it, "provide_label")[0].shape == tuple([max_batch_size] + test_label_shape)
    for _ in range(runs):
        for j, data in enumerate(it):
            assert (data_to_np(data[0]) == np.stack(dataset[j][0])).all()
            assert (label_to_np(data[0]) == np.stack(dataset[j][1])).all()
            assert getattr(it, "provide_data")[0].shape == tuple([max_batch_size] + test_data_shape)
            assert getattr(it, "provide_label")[0].shape == tuple(
                [max_batch_size] + test_label_shape
            )
            counter += 1
    assert counter == iter_limit * runs


@attr("mxnet")
def test_mxnet_iterator_properties():
    for prep in [True, False]:
        yield check_mxnet_iterator_properties, prep


@attr("mxnet")
def test_mxnet_external_source_variable_size_pass():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    check_external_source_variable_size(
        MXNetIterator,
        [("data", MXNetIterator.DATA_TAG)],
        to_np=lambda x: x.data[0].asnumpy(),
        dynamic_shape=True,
    )


@attr("mxnet")
def test_mxnet_external_source_variable_size_fail():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    assert_raises(
        AssertionError,
        check_external_source_variable_size,
        MXNetIterator,
        [("data", MXNetIterator.DATA_TAG)],
        to_np=lambda x: x.data[0].asnumpy(),
        iter_size=5,
        dynamic_shape=True,
    )


# Gluon


@attr("gluon")
@params(*stop_iteration_case_generator())
def test_stop_iteration_gluon(batch_size, epochs, iter_num, total_iter_num, auto_reset, infinite):
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    def fw_iter(pipe, size, auto_reset):
        return GluonIterator(
            pipe, size, output_types=[GluonIterator.DENSE_TAG], auto_reset=auto_reset
        )

    iter_name = "GluonIterator"
    check_stop_iter(
        fw_iter, iter_name, batch_size, epochs, iter_num, total_iter_num, auto_reset, infinite
    )


@attr("gluon")
def test_stop_iteration_gluon_fail_multi():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    def fw_iter(pipe, size, auto_reset):
        return GluonIterator(pipe, size, auto_reset=auto_reset)

    check_stop_iter_fail_multi(fw_iter)


@attr("gluon")
def test_stop_iteration_gluon_fail_single():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    def fw_iter(pipe, size, auto_reset):
        return GluonIterator(pipe, size=size, auto_reset=auto_reset)

    check_stop_iter_fail_single(fw_iter)


@attr("gluon")
def test_gluon_iterator_wrapper_first_iteration():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    check_iterator_wrapper_first_iteration(
        GluonIterator, output_types=[GluonIterator.DENSE_TAG], size=100
    )


@attr("gluon")
def test_gluon_external_source_autoreset():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    check_external_source_autoreset(
        GluonIterator, output_types=[GluonIterator.DENSE_TAG], to_np=lambda x: x[0][0].asnumpy()
    )


@attr("gluon")
def test_gluon_external_source_do_not_prepare():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    check_external_source_autoreset(
        GluonIterator,
        output_types=[GluonIterator.DENSE_TAG],
        to_np=lambda x: x[0][0].asnumpy(),
        prepare_first_batch=False,
    )


@attr("gluon")
def test_gluon_external_source_variable_size_pass():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    check_external_source_variable_size(
        GluonIterator, output_types=[GluonIterator.DENSE_TAG], to_np=lambda x: x[0].asnumpy()
    )


@attr("gluon")
def test_gluon_external_source_variable_size_fail():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    assert_raises(
        AssertionError,
        check_external_source_variable_size,
        GluonIterator,
        output_types=[GluonIterator.DENSE_TAG],
        to_np=lambda x: x[0].asnumpy(),
        iter_size=5,
    )


# PyTorch


@attr("pytorch")
@params(*stop_iteration_case_generator())
def test_stop_iteration_pytorch(batch_size, epochs, iter_num, total_iter_num, auto_reset, infinite):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    def fw_iter(pipe, size, auto_reset):
        return PyTorchIterator(pipe, output_map=["data"], size=size, auto_reset=auto_reset)

    iter_name = "PyTorchIterator"

    check_stop_iter(
        fw_iter,
        iter_name,
        batch_size,
        epochs,
        iter_num,
        total_iter_num,
        auto_reset,
        infinite,
    )


@attr("pytorch")
def test_stop_iteration_pytorch_fail_multi():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    def fw_iter(pipe, size, auto_reset):
        return PyTorchIterator(pipe, output_map=["data"], size=size, auto_reset=auto_reset)

    check_stop_iter_fail_multi(fw_iter)


@attr("pytorch")
def test_stop_iteration_pytorch_fail_single():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    def fw_iter(pipe, size, auto_reset):
        return PyTorchIterator(pipe, output_map=["data"], size=size, auto_reset=auto_reset)

    check_stop_iter_fail_single(fw_iter)


@attr("pytorch")
def test_pytorch_iterator_wrapper_first_iteration():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    check_iterator_wrapper_first_iteration(PyTorchIterator, output_map=["data"], size=100)


@attr("pytorch")
def test_pytorch_external_source_autoreset():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    check_external_source_autoreset(
        PyTorchIterator, output_map=["data"], to_np=lambda x: x[0]["data"].numpy()
    )


@attr("pytorch")
def test_pytorch_external_source_do_not_prepare():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    check_external_source_autoreset(
        PyTorchIterator,
        output_map=["data"],
        to_np=lambda x: x[0]["data"].numpy(),
        prepare_first_batch=False,
    )


@attr("pytorch")
@params((False,), (True,))
def test_pytorch_external_source_variable_size_pass(exec_dynamic):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    check_external_source_variable_size(
        PyTorchIterator,
        output_map=["data"],
        to_np=lambda x: x["data"].numpy(),
        dynamic_shape=True,
        exec_dynamic=exec_dynamic,
    )


@attr("pytorch")
def test_pytorch_external_source_variable_size_fail():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    assert_raises(
        AssertionError,
        check_external_source_variable_size,
        PyTorchIterator,
        output_map=["data"],
        to_np=lambda x: x["data"].numpy(),
        iter_size=5,
        dynamic_shape=True,
    )


# PaddlePaddle


@attr("paddle")
@params(*stop_iteration_case_generator())
def test_stop_iteration_paddle(batch_size, epochs, iter_num, total_iter_num, auto_reset, infinite):
    def fw_iter(pipe, size, auto_reset):
        from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

        return PaddleIterator(pipe, output_map=["data"], size=size, auto_reset=auto_reset)

    iter_name = "PaddleIterator"

    check_stop_iter(
        fw_iter,
        iter_name,
        batch_size,
        epochs,
        iter_num,
        total_iter_num,
        auto_reset,
        infinite,
    )


@attr("paddle")
def test_stop_iteration_paddle_fail_multi():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    def fw_iter(pipe, size, auto_reset):
        return PaddleIterator(pipe, output_map=["data"], size=size, auto_reset=auto_reset)

    check_stop_iter_fail_multi(fw_iter)


@attr("paddle")
def test_stop_iteration_paddle_fail_single():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    def fw_iter(pipe, size, auto_reset):
        return PaddleIterator(pipe, output_map=["data"], size=size, auto_reset=auto_reset)

    check_stop_iter_fail_single(fw_iter)


@attr("paddle")
def test_paddle_iterator_wrapper_first_iteration():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    check_iterator_wrapper_first_iteration(PaddleIterator, output_map=["data"], size=100)


@attr("paddle")
def test_paddle_external_source_autoreset():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    check_external_source_autoreset(
        PaddleIterator, output_map=["data"], to_np=lambda x: np.array(x[0]["data"])
    )


@attr("paddle")
def test_paddle_external_source_do_not_prepare():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    check_external_source_autoreset(
        PaddleIterator,
        output_map=["data"],
        to_np=lambda x: np.array(x[0]["data"]),
        prepare_first_batch=False,
    )


@attr("paddle")
@params((False,), (True,))
def test_paddle_external_source_variable_size_pass(exec_dynamic):
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    check_external_source_variable_size(
        PaddleIterator,
        output_map=["data"],
        to_np=lambda x: np.array(x["data"]),
        dynamic_shape=True,
        exec_dynamic=exec_dynamic,
    )


@attr("paddle")
def test_paddle_external_source_variable_size_fail():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    assert_raises(
        AssertionError,
        check_external_source_variable_size,
        PaddleIterator,
        output_map=["data"],
        to_np=lambda x: np.array(x["data"]),
        iter_size=5,
        dynamic_shape=True,
    )


# JAX


@attr("jax")
@params(*stop_iteration_case_generator())
def test_stop_iteration_jax(batch_size, epochs, iter_num, total_iter_num, auto_reset, infinite):
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    def fw_iter(pipe, size, auto_reset):
        return JaxIterator(pipe, output_map=["data"], size=size, auto_reset=auto_reset)

    iter_name = "JaxIterator"

    check_stop_iter(
        fw_iter,
        iter_name,
        batch_size,
        epochs,
        iter_num,
        total_iter_num,
        auto_reset,
        infinite,
    )


@attr("jax")
def test_stop_iteration_jax_fail_multi():
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    def fw_iter(pipe, size, auto_reset):
        return JaxIterator(pipe, output_map=["data"], size=size, auto_reset=auto_reset)

    check_stop_iter_fail_multi(fw_iter)


@attr("jax")
def test_stop_iteration_jax_fail_single():
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    def fw_iter(pipe, size, auto_reset):
        return JaxIterator(pipe, output_map=["data"], size=size, auto_reset=auto_reset)

    check_stop_iter_fail_single(fw_iter)


@attr("jax")
def test_jax_iterator_wrapper_first_iteration():
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    check_iterator_wrapper_first_iteration(JaxIterator, output_map=["data"], size=100)


@attr("jax")
def test_jax_external_source_autoreset():
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    check_external_source_autoreset(JaxIterator, output_map=["data"], to_np=lambda x: x["data"])


@attr("jax")
def test_jax_external_source_do_not_prepare():
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    check_external_source_autoreset(
        JaxIterator, output_map=["data"], to_np=lambda x: x["data"], prepare_first_batch=False
    )


def check_prepare_first_batch(Iterator, *args, to_np=None, **kwargs):
    max_batch_size = 4
    iter_limit = 4
    runs = 3
    test_data_shape = [2, 3, 4]
    i = 0
    dataset = [
        [
            [
                np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
                for _ in range(max_batch_size)
            ]
        ]
        for _ in range(iter_limit)
    ]

    def get_data():
        nonlocal i
        if i == iter_limit:
            i = 0
            raise StopIteration
        out = dataset[i]
        i += 1
        return out

    pipe = Pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    with pipe:
        outs = fn.external_source(source=get_data, num_outputs=1)
    pipe.set_outputs(*outs)

    it = Iterator([pipe], *args, auto_reset=True, prepare_first_batch=False, **kwargs)
    counter = 0
    for r in range(runs):
        if r == 0:
            # when prepare_first_batch=False pipeline should not be run until first call to next(it)
            assert i == 0, "external_source should not be run yet"
        for j, data in enumerate(it):
            if not isinstance(data, dict):
                data = data[0]
            assert (to_np(data) == np.concatenate(dataset[j])).all()
            counter += 1
    assert counter == iter_limit * runs


@attr("mxnet")
def test_mxnet_prepare_first_batch():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    check_prepare_first_batch(
        MXNetIterator,
        [("data", MXNetIterator.DATA_TAG)],
        to_np=lambda x: x.data[0].asnumpy(),
        dynamic_shape=True,
    )


@attr("gluon")
def test_gluon_prepare_first_batch():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    check_prepare_first_batch(
        GluonIterator, output_types=[GluonIterator.DENSE_TAG], to_np=lambda x: x[0].asnumpy()
    )


@attr("pytorch")
def test_pytorch_prepare_first_batch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    check_prepare_first_batch(
        PyTorchIterator, output_map=["data"], to_np=lambda x: x["data"].numpy()
    )


@attr("paddle")
def test_paddle_prepare_first_batch():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    check_prepare_first_batch(
        PaddleIterator, output_map=["data"], to_np=lambda x: np.array(x["data"])
    )


@attr("jax")
def test_jax_prepare_first_batch():
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    check_prepare_first_batch(JaxIterator, output_map=["data"], to_np=lambda x: np.array(x["data"]))


@pipeline_def
def feed_ndarray_test_pipeline():
    return np.array([1], dtype=float)


@attr("mxnet")
def test_mxnet_feed_ndarray():
    from nvidia.dali.plugin.mxnet import feed_ndarray
    import mxnet

    pipe = feed_ndarray_test_pipeline(batch_size=1, num_threads=1, device_id=0)
    out = pipe.run()[0]
    mxnet_tensor = mxnet.nd.empty([1], None, np.int8)
    assert_raises(
        AssertionError,
        feed_ndarray,
        out,
        mxnet_tensor,
        glob="The element type of DALI Tensor/TensorList doesn't match "
        "the element type of the target MXNet NDArray",
    )


@attr("pytorch")
def test_pytorch_feed_ndarray():
    from nvidia.dali.plugin.pytorch import feed_ndarray
    import torch

    pipe = feed_ndarray_test_pipeline(batch_size=1, num_threads=1, device_id=0)
    out = pipe.run()[0]
    torch_tensor = torch.empty((1), dtype=torch.int8, device="cpu")
    assert_raises(
        AssertionError,
        feed_ndarray,
        out,
        torch_tensor,
        glob="The element type of DALI Tensor/TensorList doesn't match "
        "the element type of the target PyTorch Tensor:",
    )


# last_batch_policy type check


def check_iterator_build_error(ErrorType, Iterator, glob, *args, **kwargs):
    batch_size = 4
    num_gpus = 1
    pipes, _ = create_pipeline(
        lambda gpu: create_coco_pipeline(
            batch_size=batch_size,
            num_threads=4,
            shard_id=gpu,
            num_gpus=num_gpus,
            data_paths=data_sets[0],
            random_shuffle=True,
            stick_to_shard=False,
            shuffle_after_epoch=False,
            pad_last_batch=False,
        ),
        batch_size,
        num_gpus,
    )
    with assert_raises(ErrorType, glob=glob):
        Iterator(pipes, size=pipes[0].epoch_size("Reader"), *args, **kwargs)


@attr("pytorch")
def test_pytorch_wrong_last_batch_policy_type():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    check_iterator_build_error(
        ValueError,
        PyTorchIterator,
        glob="Wrong type for `last_batch_policy`.",
        output_map=["data"],
        last_batch_policy="FILL",
    )


@attr("paddle")
def test_paddle_wrong_last_batch_policy_type():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    check_iterator_build_error(
        ValueError,
        PaddleIterator,
        glob="Wrong type for `last_batch_policy`.",
        output_map=["data"],
        last_batch_policy="FILL",
    )


@attr("mxnet")
def test_mxnet_wrong_last_batch_policy_type():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    check_iterator_build_error(
        ValueError,
        MXNetIterator,
        glob="Wrong type for `last_batch_policy`.",
        output_map=[("data", MXNetIterator.DATA_TAG)],
        last_batch_policy="FILL",
    )


@attr("gluon")
def test_gluon_wrong_last_batch_policy_type():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    check_iterator_build_error(
        ValueError,
        GluonIterator,
        glob="Wrong type for `last_batch_policy`.",
        output_types=[GluonIterator.DENSE_TAG],
        last_batch_policy="FILL",
    )


@attr("jax")
def test_jax_wrong_last_batch_policy_type():
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    check_iterator_build_error(
        ValueError,
        JaxIterator,
        glob="Wrong type for `last_batch_policy`.",
        output_map=["data"],
        last_batch_policy="FILL",
    )


@attr("jax")
def test_jax_unsupported_last_batch_policy_type():
    from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

    check_iterator_build_error(
        AssertionError,
        JaxIterator,
        glob="JAX iterator does not support partial last batch policy.",
        output_map=["data"],
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )


def check_autoreset_iter(fw_iterator, extract_data, auto_reset_op, policy):
    batch_size = 2
    number_of_samples = 11
    images_files = [__file__] * number_of_samples
    labels = list(range(number_of_samples))

    @pipeline_def
    def BoringPipeline():
        _, ls = fn.readers.file(
            files=images_files,
            labels=labels,
            stick_to_shard=True,
            name="reader",
            pad_last_batch=True,
        )

        return ls

    pipeline = BoringPipeline(batch_size=batch_size, device_id=0, num_threads=1)

    loader = fw_iterator(
        pipeline, reader_name="reader", auto_reset=auto_reset_op, last_batch_policy=policy
    )
    for _ in range(2):
        loader_iter = iter(loader)
        for i in range(len(loader_iter)):
            data = next(loader_iter)
            if not isinstance(data, dict):
                data = data[0]
            for j, d in enumerate(extract_data(data)):
                if policy is LastBatchPolicy.FILL:
                    if i * batch_size + j >= number_of_samples:
                        assert d[0] == number_of_samples - 1, f"{d[0]} {number_of_samples - 1}"
                    else:
                        assert d[0] == i * batch_size + j, f"{d[0]} {i * batch_size + j}"
                else:
                    assert d[0] == i * batch_size + j, f"{d[0]} {i * batch_size + j}"


def autoreset_iter_params():
    for auto_reset_op in ["yes", "no"]:
        for policy in [LastBatchPolicy.FILL, LastBatchPolicy.DROP, LastBatchPolicy.PARTIAL]:
            yield auto_reset_op, policy


@attr("mxnet")
@params(*autoreset_iter_params())
def test_mxnet_autoreset_iter(auto_reset_op, policy):
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    def fw_iterator(pipeline, reader_name, auto_reset, last_batch_policy):
        return MXNetIterator(
            pipeline,
            [("data", MXNetIterator.DATA_TAG)],
            reader_name=reader_name,
            auto_reset=auto_reset,
            last_batch_policy=last_batch_policy,
        )

    def extract_data(x):
        data = x.data[0].asnumpy()
        data = data[0 : -x.pad]
        return data

    check_autoreset_iter(fw_iterator, extract_data, auto_reset_op, policy)


@attr("gluon")
@params(*autoreset_iter_params())
def test_gluon_autoreset_iter(auto_reset_op, policy):
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator

    def fw_iterator(pipeline, reader_name, auto_reset, last_batch_policy):
        return GluonIterator(
            pipeline,
            reader_name=reader_name,
            auto_reset=auto_reset,
            last_batch_policy=last_batch_policy,
        )

    def extract_data(x):
        return x[0].asnumpy()

    check_autoreset_iter(fw_iterator, extract_data, auto_reset_op, policy)


@attr("pytorch")
@params(*autoreset_iter_params())
def test_pytorch_autoreset_iter(auto_reset_op, policy):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    def fw_iterator(pipeline, reader_name, auto_reset, last_batch_policy):
        return PyTorchIterator(
            pipeline,
            output_map=["data"],
            reader_name=reader_name,
            auto_reset=auto_reset,
            last_batch_policy=last_batch_policy,
        )

    def extract_data(x):
        return x["data"].numpy()

    check_autoreset_iter(fw_iterator, extract_data, auto_reset_op, policy)


@attr("paddle")
@params(*autoreset_iter_params())
def test_paddle_autoreset_iter(auto_reset_op, policy):
    def fw_iterator(pipeline, reader_name, auto_reset, last_batch_policy):
        from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

        return PaddleIterator(
            pipeline,
            output_map=["data"],
            reader_name=reader_name,
            auto_reset=auto_reset,
            last_batch_policy=last_batch_policy,
        )

    def extract_data(x):
        return np.array(x["data"])

    check_autoreset_iter(fw_iterator, extract_data, auto_reset_op, policy)


def autoreset_iter_params_jax():
    for auto_reset_op in ["yes", "no"]:
        for policy in [LastBatchPolicy.FILL, LastBatchPolicy.DROP]:
            yield auto_reset_op, policy


@attr("jax")
@params(*autoreset_iter_params_jax())
def test_jax_autoreset_iter(auto_reset_op, policy):
    def fw_iterator(pipeline, reader_name, auto_reset, last_batch_policy):
        from nvidia.dali.plugin.jax import DALIGenericIterator as JaxIterator

        return JaxIterator(
            pipeline,
            output_map=["data"],
            reader_name=reader_name,
            auto_reset=auto_reset,
            last_batch_policy=last_batch_policy,
        )

    def extract_data(x):
        return np.array(x["data"])

    check_autoreset_iter(fw_iterator, extract_data, auto_reset_op, policy)
