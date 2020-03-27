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
import nvidia.dali.types as types
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import numpy as np
import os
from test_utils import get_dali_extra_path
from nose.tools import raises

class COCOReaderPipeline(Pipeline):
    def __init__(self, data_paths, batch_size, num_threads, shard_id, num_gpus, random_shuffle, stick_to_shard, shuffle_after_epoch, pad_last_batch, initial_fill=1024, return_labels=False):
        # use only 1 GPU, as we care only about shard_id
        super(COCOReaderPipeline, self).__init__(batch_size, num_threads, 0, prefetch_queue_depth=1)
        self.input = ops.COCOReader(file_root = data_paths[0], annotations_file=data_paths[1],
                                    shard_id = shard_id, num_shards = num_gpus, random_shuffle=random_shuffle,
                                    save_img_ids=True, stick_to_shard=stick_to_shard,shuffle_after_epoch=shuffle_after_epoch,
                                    pad_last_batch=pad_last_batch, initial_fill=initial_fill)
        self.return_labels=return_labels

    def define_graph(self):
        _, _, labels, ids = self.input(name="Reader")
        if self.return_labels:
            return labels, ids
        return ids

test_data_root = get_dali_extra_path()
coco_folder = os.path.join(test_data_root, 'db', 'coco')
data_sets = [[os.path.join(coco_folder, 'images'), os.path.join(coco_folder, 'instances.json')]]
image_data_set = os.path.join(test_data_root, 'db', 'single', 'jpeg')


def gather_ids(dali_train_iter, data_getter, pad_getter, data_size):
    img_ids_list = []
    batch_size = dali_train_iter.batch_size
    pad = 0
    for it in iter(dali_train_iter):
        tmp = data_getter(it[0]).copy()
        pad += pad_getter(it[0])
        img_ids_list.append(tmp)
    img_ids_list = np.concatenate(img_ids_list)
    img_ids_list_set = set(img_ids_list)

    remainder = int(math.ceil(data_size / batch_size)) * batch_size - data_size
    mirrored_data = img_ids_list[-remainder - 1:]

    return img_ids_list, img_ids_list_set, mirrored_data, pad, remainder


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


def test_mxnet_iterator_model_fit():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    import mxnet as mx
    num_gpus = 1
    batch_size = 1
    class RN50Pipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths):
            super(RN50Pipeline, self).__init__(batch_size, num_threads, device_id,)
            self.input = ops.FileReader(file_root = data_paths, shard_id = device_id, num_shards = num_gpus)
            self.decode_gpu = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
            self.res = ops.RandomResizedCrop(device="cpu", size =(224,224))

            self.cmnp = ops.CropMirrorNormalize(device="cpu",
                                                output_dtype=types.FLOAT,
                                                output_layout=types.NCHW,
                                                crop=(224, 224),
                                                image_type=types.RGB,
                                                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                                std=[0.229 * 255,0.224 * 255,0.225 * 255])
            self.coin = ops.CoinFlip(probability=0.5)

        def define_graph(self):
            rng = self.coin()
            jpegs, labels = self.input(name="Reader")
            images = self.decode_gpu(jpegs)
            images = self.res(images)
            output = self.cmnp(images, mirror=rng)
            return labels

    pipes, _ = create_pipeline(lambda gpu: RN50Pipeline(batch_size=batch_size, num_threads=4, device_id=gpu, num_gpus=num_gpus,
                                                        data_paths=image_data_set), batch_size, num_gpus)
    pipe = pipes[0]

    class MXNetIteratorWrapper(MXNetIterator):
        def __init__(self, iter):
            self.iter = iter

        def __getattr__(self, attr):
            return getattr(self.iter, attr)

        def __next__(self):
            ret = self.iter.__next__()[0]
            return ret

    dali_train_iter = MXNetIterator(pipe, [("labels", MXNetIterator.LABEL_TAG)],
                                    size=pipe.epoch_size("Reader"))
    data = mx.symbol.Variable('labels')

    # create a dummy model
    _ = mx.model.FeedForward.create(data,
                                    X=MXNetIteratorWrapper(dali_train_iter),
                                    num_epoch=1,
                                    learning_rate=0.01)


def test_mxnet_iterator_last_batch_no_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                  data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                  shuffle_after_epoch=False, pad_last_batch=False), batch_size, num_gpus)

    dali_train_iter = MXNetIterator(pipes, [("ids", MXNetIterator.DATA_TAG)],
                                    size=pipes[0].epoch_size("Reader"), fill_last_batch=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x.data[0].squeeze().asnumpy(), lambda x: x.pad, data_size)

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1

def test_mxnet_iterator_empty_array():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    import mxnet as mx

    batch_size = 4
    size = 5

    all_np_types = [np.bool_, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64, np.float_, np.float32, np.float16,
                    np.short, np.long, np.longlong, np.ushort, np.ulonglong]
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
        out = [[np.empty(test_data_shape, dtype = t)] * batch_size for t in np_types]
        out = [val for pair in zip(out, out) for val in pair]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=0)
    outs = fn.external_source(source = get_data, num_outputs = len(np_types) * 2)
    pipe.set_outputs(*outs)
    pipe.build()

    # create map of [(data, type_a), (label, type_a), ...]
    data_map = [('data_{}'.format(i), MXNetIterator.DATA_TAG) for i, t in enumerate(np_types)]
    label_map = [('label_{}'.format(i), MXNetIterator.LABEL_TAG) for i, t in enumerate(np_types)]
    out_map = [val for pair in zip(data_map, label_map) for val in pair]

    iterator = MXNetIterator(
    pipe,
    output_map=out_map,
    size=size,
    dynamic_shape=True)

    for batch in iterator:
        for d, t in zip(batch[0].data, np_types):
            shape = d.asnumpy().shape
            assert shape[0] == batch_size
            print(shape)
            assert np.array_equal(shape[1:], test_data_shape)
            assert d.asnumpy().dtype == t

def test_mxnet_iterator_last_batch_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=True), batch_size, num_gpus)

    dali_train_iter = MXNetIterator(pipes, [("ids", MXNetIterator.DATA_TAG)],
                                    size=pipes[0].epoch_size("Reader"), fill_last_batch=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x.data[0].squeeze().asnumpy(), lambda x: x.pad, data_size)

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x.data[0].squeeze().asnumpy(), lambda x: x.pad, data_size)

    assert len(next_img_ids_list) > data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1


def test_mxnet_iterator_not_fill_last_batch_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=True), batch_size, num_gpus)

    dali_train_iter = MXNetIterator(pipes, [("ids", MXNetIterator.DATA_TAG)], size=pipes[0].epoch_size("Reader"),
                                    fill_last_batch=False)

    img_ids_list, img_ids_list_set, mirrored_data, pad, remainder = \
        gather_ids(dali_train_iter, lambda x: x.data[0].squeeze().asnumpy(), lambda x: x.pad, data_size)

    assert pad == remainder
    assert len(img_ids_list) - pad == data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, pad, remainder = \
        gather_ids(dali_train_iter, lambda x: x.data[0].squeeze().asnumpy(), lambda x: x.pad, data_size)

    assert pad == remainder
    assert len(next_img_ids_list) - pad == data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1

def check_mxnet_iterator_pass_reader_name(shards_num, pipes_number, batch_size, stick_to_shard, pad, iters, fill_last_batch):
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    pipes = [COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=id, num_gpus=shards_num,
                                data_paths=data_sets[0], random_shuffle=False, stick_to_shard=stick_to_shard,
                                shuffle_after_epoch=False, pad_last_batch=pad) for id in range(shards_num)]
    dali_train_iter = MXNetIterator(pipes, [("ids", MXNetIterator.DATA_TAG)], reader_name="Reader", fill_last_batch=fill_last_batch)
    data_set_size = pipes[0].epoch_size("Reader", False)
    rounded_shard_size = math.ceil(data_set_size / batch_size) * batch_size // shards_num
    ids = [pipe.shard_id("Reader") for pipe in pipes]
    per_gpu_counter = [0] * shards_num
    for _ in range(iters):
        out_set = []
        img_ids_list = [[] for _ in range(pipes_number)]
        for it in iter(dali_train_iter):
            for id in range(pipes_number):
                tmp = it[id].data[0].squeeze().asnumpy().copy()
                if it[id].pad:
                    tmp = tmp[0:-it[id].pad]
                img_ids_list[id].append(tmp)
        dali_train_iter.reset()
        for id in range(pipes_number):
            img_ids_list[id] = np.concatenate(img_ids_list[id])
            out_set.append(set(img_ids_list[id]))

        if pad:
            assert pipes_number == 1 or len(set.intersection(*out_set)) == 0, "Shards should not overlaps in the epoch"
            assert len(set.union(*out_set)) == sum([len(v) for v in img_ids_list]), "Data returned from shard should not duplicate values"
        if not fill_last_batch:
           for id_list, id_set, id in zip(img_ids_list, out_set, ids):
                shard_size = int((id + 1) *data_set_size / shards_num) - int(id  *data_set_size / shards_num)
                assert len(id_list) == shard_size
                assert len(id_set) == shard_size
        else:
            samples_to_read = rounded_shard_size
            # if each GPU read ahead at least batch size, then the iteration would be shorter
            if min(per_gpu_counter) >= batch_size:
                samples_to_read -= min(per_gpu_counter)

            shard_size =[int((id + 1) *data_set_size / shards_num) - int(id  *data_set_size / shards_num) for id in ids]
            per_gpu_counter = [samples_to_read - s + c for s, c in zip(shard_size, per_gpu_counter)]
            for id_list, id_set, size in zip(img_ids_list, out_set, shard_size):
                assert len(id_list) == samples_to_read
                if not stick_to_shard:
                    assert len(id_list) == len(id_set)
                else:
                    assert len(id_set) == size
        if not stick_to_shard:
            ids = [(id + 1) % shards_num for id in ids]

def test_mxnet_iterator_pass_reader_name():
    for shards_num in [3, 5, 17]:
        for batch_size in [3, 5, 7]:
            for stick_to_shard in [False, True]:
                for pad in [True, False]:
                    for fill_last_batch in  [False, False]:
                        for iters in [1, 2, 3, 2*shards_num]:
                            for pipes_number in [1, shards_num]:
                                yield check_mxnet_iterator_pass_reader_name, shards_num, pipes_number, batch_size, stick_to_shard, pad, iters, fill_last_batch

def test_gluon_iterator_last_batch_no_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                  data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                  shuffle_after_epoch=False, pad_last_batch=False), batch_size, num_gpus)

    dali_train_iter = GluonIterator(pipes, size=pipes[0].epoch_size("Reader"), fill_last_batch=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x[0].squeeze().asnumpy(), lambda x: 0, data_size)

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1

def test_gluon_iterator_last_batch_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=True), batch_size, num_gpus)

    dali_train_iter = GluonIterator(pipes,
                                    size=pipes[0].epoch_size("Reader"), fill_last_batch=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x[0].squeeze().asnumpy(), lambda x: 0, data_size)

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x[0].squeeze().asnumpy(), lambda x: 0, data_size)

    assert len(next_img_ids_list) > data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1

def test_gluon_iterator_not_fill_last_batch_pad_last_batch():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=False, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=True), batch_size, num_gpus)

    dali_train_iter = GluonIterator(pipes, size=pipes[0].epoch_size("Reader"),
                                    fill_last_batch=False)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x[0].squeeze().asnumpy(), lambda x: 0, data_size)

    assert len(img_ids_list) == data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1


    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, pad, remainder = \
        gather_ids(dali_train_iter, lambda x: x[0].squeeze().asnumpy(), lambda x: 0, data_size)

    assert len(next_img_ids_list) == data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) != 1

def test_gluon_iterator_sparse_batch():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator
    from mxnet.ndarray.ndarray import NDArray
    num_gpus = 1
    batch_size = 16

    pipes, _ = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                  data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                  shuffle_after_epoch=False, pad_last_batch=True, return_labels=True), batch_size, num_gpus)

    dali_train_iter = GluonIterator(pipes, pipes[0].epoch_size("Reader"),
                                           output_types=[GluonIterator.SPARSE_TAG,
                                            GluonIterator.DENSE_TAG],
                                            fill_last_batch=True)

    for it in dali_train_iter:
        labels, ids = it[0] # gpu 0
        # labels should be a sparse batch: a list of per-sample NDArray's
        # ids should be a dense batch: a single NDarray reprenseting the batch
        assert isinstance(labels, (tuple,list))
        assert len(labels) == batch_size
        assert isinstance(labels[0], NDArray)
        assert isinstance(ids, NDArray)


def check_gluon_iterator_pass_reader_name(shards_num, pipes_number, batch_size, stick_to_shard, pad, iters, fill_last_batch):
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator
    pipes = [COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=id, num_gpus=shards_num,
                                data_paths=data_sets[0], random_shuffle=False, stick_to_shard=stick_to_shard,
                                shuffle_after_epoch=False, pad_last_batch=pad) for id in range(shards_num)]
    dali_train_iter = GluonIterator(pipes, reader_name="Reader", fill_last_batch=fill_last_batch)
    data_set_size = pipes[0].epoch_size("Reader", False)
    rounded_shard_size = math.ceil(data_set_size / batch_size) * batch_size // shards_num
    ids = [pipe.shard_id("Reader") for pipe in pipes]
    per_gpu_counter = [0] * shards_num

    for _ in range(iters):
        out_set = []
        img_ids_list = [[] for _ in range(pipes_number)]
        for it in iter(dali_train_iter):
            for id in range(pipes_number):
                if len(it[id][0]):
                    tmp = it[id][0].squeeze().asnumpy().copy()
                else:
                    tmp = np.empty([0])
                img_ids_list[id].append(tmp)
        dali_train_iter.reset()
        for id in range(pipes_number):
            img_ids_list[id] = np.concatenate(img_ids_list[id])
            out_set.append(set(img_ids_list[id]))

        if pad:
            assert pipes_number == 1 or len(set.intersection(*out_set)) == 0, "Shards should not overlaps in the epoch"
            assert len(set.union(*out_set)) == sum([len(v) for v in img_ids_list]), "Data returned from shard should not duplicate values"
        if not fill_last_batch:
           for id_list, id_set, id in zip(img_ids_list, out_set, ids):
                shard_size = int((id + 1) *data_set_size / shards_num) - int(id  *data_set_size / shards_num)
                assert len(id_list) == shard_size
                assert len(id_set) == shard_size
        else:
            samples_to_read = rounded_shard_size
            # if each GPU read ahead at least batch size, then the iteration would be shorter
            if min(per_gpu_counter) >= batch_size:
                samples_to_read -= min(per_gpu_counter)

            shard_size =[int((id + 1) *data_set_size / shards_num) - int(id  *data_set_size / shards_num) for id in ids]
            per_gpu_counter = [samples_to_read - s + c for s, c in zip(shard_size, per_gpu_counter)]
            for id_list, id_set, size in zip(img_ids_list, out_set, shard_size):
                assert len(id_list) == samples_to_read
                if not stick_to_shard:
                    assert len(id_list) == len(id_set)
                else:
                    assert len(id_set) == size
        if not stick_to_shard:
            ids = [(id + 1) % shards_num for id in ids]

def test_gluon_iterator_pass_reader_name():
   for shards_num in [3, 5, 17]:
        for batch_size in [3, 5, 7]:
            for stick_to_shard in [False, True]:
                for pad in [True, False]:
                    for fill_last_batch in  [False, False]:
                        for iters in [1, 2, 3, 2*shards_num]:
                            for pipes_number in [1, shards_num]:
                                yield check_gluon_iterator_pass_reader_name, shards_num, pipes_number, batch_size, stick_to_shard, pad, iters, fill_last_batch

def test_pytorch_iterator_last_batch_no_pad_last_batch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=False), batch_size, num_gpus)

    dali_train_iter = PyTorchIterator(pipes, output_map=["data"], size=pipes[0].epoch_size("Reader"), fill_last_batch=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x["data"].squeeze().numpy(), lambda x: 0, data_size)

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1

def test_pytorch_iterator_last_batch_pad_last_batch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=True), batch_size, num_gpus)

    dali_train_iter = PyTorchIterator(pipes, output_map=["data"], size=pipes[0].epoch_size("Reader"), fill_last_batch=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x["data"].squeeze().numpy(), lambda x: 0, data_size)

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x["data"].squeeze().numpy(), lambda x: 0, data_size)

    assert len(next_img_ids_list) > data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1


def test_pytorch_iterator_not_fill_last_batch_pad_last_batch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                     data_paths=data_sets[0], random_shuffle=False, stick_to_shard=False,
                                                                     shuffle_after_epoch=False, pad_last_batch=True), batch_size, num_gpus)

    dali_train_iter = PyTorchIterator(pipes, output_map=["data"], size=pipes[0].epoch_size("Reader"), fill_last_batch=False, last_batch_padded=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x["data"].squeeze().numpy(), lambda x: 0, data_size)

    assert len(img_ids_list) == data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: x["data"].squeeze().numpy(), lambda x: 0, data_size)

    # there is no mirroring as data in the output is just cut off,
    # in the mirrored_data there is real data
    assert len(next_img_ids_list) == data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) != 1

class CustomPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths):
        super(CustomPipe, self).__init__(batch_size, num_threads, device_id,)
        self.input = ops.FileReader(file_root = data_paths, shard_id = device_id, num_shards = num_gpus)
        self.decode_gpu = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size =(224, 224))

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        jpegs, _ = self.input(name="Reader")
        images = self.decode_gpu(jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return output

def test_pytorch_iterator_feed_ndarray():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
    from nvidia.dali.plugin.pytorch import feed_ndarray as feed_ndarray
    import torch
    num_gpus = 1
    batch_size = 100
    pipes, _ = create_pipeline(lambda gpu: CustomPipe(batch_size=batch_size, num_threads=4, device_id=gpu, num_gpus=num_gpus,
                                                      data_paths=image_data_set), batch_size, num_gpus)
    for gpu_id in range(num_gpus):
        pipe = pipes[gpu_id]
        pipe.build()
        outs = pipe.run()
        out_data = outs[0].as_tensor()
        device = torch.device('cuda', gpu_id)
        arr = torch.zeros(out_data.shape(), dtype=torch.float32, device=device)
        feed_ndarray(out_data, arr, cuda_stream = torch.cuda.current_stream(device=device))
        np.testing.assert_equal(arr.cpu().numpy(), outs[0].as_cpu().as_array())

def test_mxnet_iterator_feed_ndarray():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    from nvidia.dali.plugin.mxnet import feed_ndarray as feed_ndarray
    import mxnet as mx

    num_gpus = 1
    batch_size = 100
    pipes, _ = create_pipeline(lambda gpu: CustomPipe(batch_size=batch_size, num_threads=4, device_id=gpu, num_gpus=num_gpus,
                                                      data_paths=image_data_set), batch_size, num_gpus)
    for gpu_id in range(num_gpus):
        pipe = pipes[gpu_id]
        pipe.build()
        outs = pipe.run()
        out_data = outs[0].as_tensor()
        with mx.Context(mx.gpu(gpu_id)):
            arr = mx.nd.zeros(out_data.shape(), dtype=np.float32)
            mx.base._LIB.MXNDArrayWaitToWrite(arr.handle)
            feed_ndarray(out_data, arr, cuda_stream = None)  # Using DALI's internal stream
            np.testing.assert_equal(arr.asnumpy(), outs[0].as_cpu().as_array())

            arr2 = mx.nd.zeros(out_data.shape(), dtype=np.float32)
            mx.base._LIB.MXNDArrayWaitToWrite(arr2.handle)
            feed_ndarray(out_data, arr2, cuda_stream = 0)  # Using default stream
            np.testing.assert_equal(arr2.asnumpy(), outs[0].as_cpu().as_array())

def test_paddle_iterator_feed_ndarray():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator
    from nvidia.dali.plugin.paddle import feed_ndarray as feed_ndarray
    from paddle import fluid

    num_gpus = 1
    batch_size = 100
    pipes, _ = create_pipeline(lambda gpu: CustomPipe(batch_size=batch_size, num_threads=4, device_id=gpu, num_gpus=num_gpus,
                                                      data_paths=image_data_set), batch_size, num_gpus)
    for gpu_id in range(num_gpus):
        pipe = pipes[gpu_id]
        pipe.build()
        outs = pipe.run()
        out_data = outs[0].as_tensor()

        lod_tensor = fluid.core.LoDTensor()
        lod_tensor._set_dims(out_data.shape())
        gpu_place = fluid.CUDAPlace(gpu_id)

        ptr = lod_tensor._mutable_data(gpu_place, fluid.core.VarDesc.VarType.FP32)
        arr = np.array(lod_tensor)
        feed_ndarray(out_data, ptr, cuda_stream = None)  # Using DALI's internal stream
        np.testing.assert_equal(np.array(lod_tensor), outs[0].as_cpu().as_array())

        lod_tensor2 = fluid.core.LoDTensor()
        lod_tensor2._set_dims(out_data.shape())

        ptr2 = lod_tensor2._mutable_data(gpu_place, fluid.core.VarDesc.VarType.FP32)
        arr2 = np.array(lod_tensor2)
        feed_ndarray(out_data, ptr2, cuda_stream = 0)  # Using default stream
        np.testing.assert_equal(np.array(lod_tensor2), outs[0].as_cpu().as_array())

def check_pytorch_iterator_pass_reader_name(shards_num, batch_size, stick_to_shard, pad, iters):
def check_pytorch_iterator_pass_reader_name(shards_num, pipes_number, batch_size, stick_to_shard, pad, iters, fill_last_batch):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
    pipes = [COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=id, num_gpus=shards_num,
                                data_paths=data_sets[0], random_shuffle=False, stick_to_shard=stick_to_shard,
                                shuffle_after_epoch=False, pad_last_batch=pad) for id in range(pipes_number)]
    dali_train_iter = PyTorchIterator(pipes, output_map=["data"], reader_name="Reader", fill_last_batch=fill_last_batch)
    data_set_size = pipes[0].epoch_size("Reader", False)
    rounded_shard_size = math.ceil(data_set_size / batch_size) * batch_size // shards_num
    ids = [pipe.shard_id("Reader") for pipe in pipes]
    per_gpu_counter = [0] * shards_num
    for _ in range(iters):
        out_set = []
        img_ids_list = [[] for _ in range(pipes_number)]
        for it in iter(dali_train_iter):
            for id in range(pipes_number):
                tmp = it[id]["data"].squeeze(dim=1).numpy().copy()
                img_ids_list[id].append(tmp)
        dali_train_iter.reset()
        for id in range(pipes_number):
            img_ids_list[id] = np.concatenate(img_ids_list[id])
            out_set.append(set(img_ids_list[id]))

        if pad:
            assert pipes_number == 1 or len(set.intersection(*out_set)) == 0, "Shards should not overlaps in the epoch"
            assert len(set.union(*out_set)) == sum([len(v) for v in img_ids_list]), "Data returned from shard should not duplicate values"
        if not fill_last_batch:
           for id_list, id_set, id in zip(img_ids_list, out_set, ids):
                shard_size = int((id + 1) *data_set_size / shards_num) - int(id  *data_set_size / shards_num)
                assert len(id_list) == shard_size
                assert len(id_set) == shard_size
        else:
            samples_to_read = rounded_shard_size
            # if each GPU read ahead at least batch size, then the iteration would be shorter
            if min(per_gpu_counter) >= batch_size:
                samples_to_read -= min(per_gpu_counter)

            shard_size =[int((id + 1) *data_set_size / shards_num) - int(id  *data_set_size / shards_num) for id in ids]
            per_gpu_counter = [samples_to_read - s + c for s, c in zip(shard_size, per_gpu_counter)]
            for id_list, id_set, size in zip(img_ids_list, out_set, shard_size):
                assert len(id_list) == samples_to_read
                if not stick_to_shard:
                    assert len(id_list) == len(id_set)
                else:
                    assert len(id_set) == size
        if not stick_to_shard:
            ids = [(id + 1) % shards_num for id in ids]

def test_pytorch_iterator_pass_reader_name():
    for shards_num in [3, 5, 17]:
        for batch_size in [3, 5, 7]:
            for stick_to_shard in [False, True]:
                for pad in [True, False]:
                    for fill_last_batch in  [False, False]:
                        for iters in [1, 2, 3, 2*shards_num]:
                            for pipes_number in [1, shards_num]:
                                yield check_pytorch_iterator_pass_reader_name, shards_num, pipes_number, batch_size, stick_to_shard, pad, iters, fill_last_batch

def test_paddle_iterator_last_batch_no_pad_last_batch():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=False), batch_size, num_gpus)

    dali_train_iter = PaddleIterator(pipes, output_map=["data"], size=pipes[0].epoch_size("Reader"), fill_last_batch=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size)

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1

def test_paddle_iterator_last_batch_pad_last_batch():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator
    num_gpus = 1
    batch_size = 100

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=True, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=True), batch_size, num_gpus)

    dali_train_iter = PaddleIterator(pipes, output_map=["data"], size=pipes[0].epoch_size("Reader"), fill_last_batch=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size)

    assert len(img_ids_list) > data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) == 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size)

    assert len(next_img_ids_list) > data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) == 1

def test_paddle_iterator_not_fill_last_batch_pad_last_batch():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator
    num_gpus = 1
    batch_size = 100
    iters = 0

    pipes, data_size = create_pipeline(lambda gpu: COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=gpu, num_gpus=num_gpus,
                                                                      data_paths=data_sets[0], random_shuffle=False, stick_to_shard=False,
                                                                      shuffle_after_epoch=False, pad_last_batch=True), batch_size, num_gpus)

    dali_train_iter = PaddleIterator(pipes, output_map=["data"], size=pipes[0].epoch_size("Reader"), fill_last_batch=False, last_batch_padded=True)

    img_ids_list, img_ids_list_set, mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size)

    assert len(img_ids_list) == data_size
    assert len(img_ids_list_set) == data_size
    assert len(set(mirrored_data)) != 1

    dali_train_iter.reset()
    next_img_ids_list, next_img_ids_list_set, next_mirrored_data, _, _ = \
        gather_ids(dali_train_iter, lambda x: np.array(x["data"]).squeeze(), lambda x: 0, data_size)

    # there is no mirroring as data in the output is just cut off,
    # in the mirrored_data there is real data
    assert len(next_img_ids_list) == data_size
    assert len(next_img_ids_list_set) == data_size
    assert len(set(next_mirrored_data)) != 1


def check_paddle_iterator_pass_reader_name(shards_num, pipes_number, batch_size, stick_to_shard, pad, iters, fill_last_batch):
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator
    pipes = [COCOReaderPipeline(batch_size=batch_size, num_threads=4, shard_id=id, num_gpus=shards_num,
                                data_paths=data_sets[0], random_shuffle=False, stick_to_shard=stick_to_shard,
                                shuffle_after_epoch=False, pad_last_batch=pad) for id in range(shards_num)]
    dali_train_iter = PaddleIterator(pipes, output_map=["data"], reader_name="Reader", fill_last_batch=fill_last_batch)
    data_set_size = pipes[0].epoch_size("Reader", False)
    rounded_shard_size = math.ceil(data_set_size / batch_size) * batch_size // shards_num
    ids = [pipe.shard_id("Reader") for pipe in pipes]
    per_gpu_counter = [0] * shards_num
    for _ in range(iters):
        out_set = []
        img_ids_list = [[] for _ in range(pipes_number)]
        for it in iter(dali_train_iter):
            for id in range(pipes_number):
                tmp = np.array(it[id]["data"]).squeeze(axis=1).copy()
                img_ids_list[id].append(tmp)
        dali_train_iter.reset()
        for id in range(pipes_number):
            img_ids_list[id] = np.concatenate(img_ids_list[id])
            out_set.append(set(img_ids_list[id]))

        if pad:
            assert pipes_number == 1 or len(set.intersection(*out_set)) == 0, "Shards should not overlaps in the epoch"
            assert len(set.union(*out_set)) == sum([len(v) for v in img_ids_list]), "Data returned from shard should not duplicate values"
        if not fill_last_batch:
           for id_list, id_set, id in zip(img_ids_list, out_set, ids):
                shard_size = int((id + 1) *data_set_size / shards_num) - int(id  *data_set_size / shards_num)
                assert len(id_list) == shard_size
                assert len(id_set) == shard_size
        else:
            samples_to_read = rounded_shard_size
            # if each GPU read ahead at least batch size, then the iteration would be shorter
            if min(per_gpu_counter) >= batch_size:
                samples_to_read -= min(per_gpu_counter)

            shard_size =[int((id + 1) *data_set_size / shards_num) - int(id  *data_set_size / shards_num) for id in ids]
            per_gpu_counter = [samples_to_read - s + c for s, c in zip(shard_size, per_gpu_counter)]
            for id_list, id_set, size in zip(img_ids_list, out_set, shard_size):
                assert len(id_list) == samples_to_read
                if not stick_to_shard:
                    assert len(id_list) == len(id_set)
                else:
                    assert len(id_set) == size
        if not stick_to_shard:
            ids = [(id + 1) % shards_num for id in ids]

def test_paddle_iterator_pass_reader_name():
    for shards_num in [3, 5, 17]:
        for batch_size in [3, 5, 7]:
            for stick_to_shard in [False, True]:
                for pad in [True, False]:
                    for fill_last_batch in  [False, False]:
                        for iters in [1, 2, 3, 2*shards_num]:
                            for pipes_number in [1, shards_num]:
                                yield check_paddle_iterator_pass_reader_name, shards_num, pipes_number, batch_size, stick_to_shard, pad, iters, fill_last_batch

class TestIterator():
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        batch = []
        if self.i < self.n:
            batch = [np.arange(0, 10 , dtype=np.uint8) for _ in range(self.batch_size)]
            self.i += 1
            return batch
        else:
            self.i = 0
            raise StopIteration
    next = __next__

    @property
    def size(self,):
        return self.n * self.batch_size

class TestIterPipeline(Pipeline):
    def __init__(self, batch_size, device_id, data_source, num_threads=4):
        super(TestIterPipeline, self).__init__(batch_size, num_threads, device_id)
        self.data_source = data_source
        self.dataset = iter(self.data_source)
        self.test_feeder = ops.ExternalSource()

    def define_graph(self,):
        self.test_data = self.test_feeder()
        return self.test_data

    def iter_setup(self,):
        try:
            data = self.dataset.next()
            self.feed_input(self.test_data, data)
        except StopIteration:
            self.dataset = iter(self.data_source)
            raise StopIteration

    @property
    def size(self):
        return self.data_source.size

def check_stop_iter(fw_iter, iterator_name, batch_size, epochs, iter_num, auto_reset, infinite):
    pipe = TestIterPipeline(batch_size, 0, TestIterator(iter_num, batch_size))
    if infinite:
        iter_size = -1
    else:
        iter_size = pipe.size
    loader = fw_iter(pipe, iter_size, auto_reset)
    count = 0
    for e in range(epochs):
        for i, outputs in enumerate(loader):
            count += 1
        if not auto_reset or infinite:
            loader.reset()
    assert(count == iter_num * epochs)

@raises(Exception)
def check_stop_iter_fail_multi(fw_iter):
    batch_size = 1
    iter_num = 1
    pipes = [TestIterPipeline(batch_size, 0, TestIterator(iter_num, batch_size)) for _ in range(2)]
    fw_iter(pipes, -1, False)

@raises(Exception)
def check_stop_iter_fail_single(fw_iter):
    batch_size = 1
    iter_num = 1
    pipes = [TestIterPipeline(batch_size, 0, TestIterator(iter_num, batch_size)) for _ in range(1)]
    fw_iter(pipes, 0, False)

def stop_teration_case_generator():
    for epochs in [1, 3 ,6]:
        for iter_num in [1, 2, 5, 9]:
            for batch_size in [1, 10, 100]:
                for auto_reset in [True, False]:
                    for infinite in [False, True]:
                        yield batch_size, epochs, iter_num, auto_reset, infinite

# MXNet
def test_stop_iteration_mxnet():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    fw_iter = lambda pipe, size, auto_reset : MXNetIterator(pipe, [("data", MXNetIterator.DATA_TAG)], size=size, auto_reset=auto_reset)
    iter_name = "MXNetIterator"
    for batch_size, epochs, iter_num, auto_reset, infinite in stop_teration_case_generator():
        yield check_stop_iter, fw_iter, iter_name, batch_size, epochs, iter_num, auto_reset, infinite

def test_stop_iteration_mxnet_fail_multi():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    fw_iter = lambda pipe, size, auto_reset : MXNetIterator(pipe, [("data", MXNetIterator.DATA_TAG)], size=size, auto_reset=auto_reset)
    check_stop_iter_fail_multi(fw_iter)

def test_stop_iteration_mxnet_fail_single():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
    fw_iter = lambda pipe, size, auto_reset : MXNetIterator(pipe, [("data", MXNetIterator.DATA_TAG)], size=size, auto_reset=auto_reset)
    check_stop_iter_fail_single(fw_iter)

# Gluon
def test_stop_iteration_gluon():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator
    fw_iter = lambda pipe, size, auto_reset : GluonIterator(pipe, size, output_types=[GluonIterator.DENSE_TAG], auto_reset=auto_reset)
    iter_name = "GluonIterator"
    for batch_size, epochs, iter_num, auto_reset, infinite in stop_teration_case_generator():
        yield check_stop_iter, fw_iter, iter_name, batch_size, epochs, iter_num, auto_reset, infinite

def test_stop_iteration_gluon_fail_multi():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator
    fw_iter = lambda pipe, size, auto_reset : GluonIterator(pipe, size, auto_reset=auto_reset)
    check_stop_iter_fail_multi(fw_iter)

def test_stop_iteration_gluon_fail_single():
    from nvidia.dali.plugin.mxnet import DALIGluonIterator as GluonIterator
    fw_iter = lambda pipe, size, auto_reset : GluonIterator(pipe, size=size, auto_reset=auto_reset)
    check_stop_iter_fail_single(fw_iter)

# PyTorch
def test_stop_iteration_pytorch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
    fw_iter = lambda pipe, size, auto_reset : PyTorchIterator(pipe, output_map=["data"],  size=size, auto_reset=auto_reset)
    iter_name = "PyTorchIterator"
    for batch_size, epochs, iter_num, auto_reset, infinite in stop_teration_case_generator():
        yield check_stop_iter, fw_iter, iter_name, batch_size, epochs, iter_num, auto_reset, infinite

def test_stop_iteration_pytorch_fail_multi():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
    fw_iter = lambda pipe, size, auto_reset : PyTorchIterator(pipe, output_map=["data"],  size=size, auto_reset=auto_reset)
    check_stop_iter_fail_multi(fw_iter)

def test_stop_iteration_pytorch_fail_single():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
    fw_iter = lambda pipe, size, auto_reset : PyTorchIterator(pipe, output_map=["data"],  size=size, auto_reset=auto_reset)
    check_stop_iter_fail_single(fw_iter)

# PaddlePaddle
def test_stop_iteration_paddle():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator
    fw_iter = lambda pipe, size, auto_reset : PaddleIterator(pipe, output_map=["data"],  size=size, auto_reset=auto_reset)
    iter_name = "PaddleIterator"
    for batch_size, epochs, iter_num, auto_reset, infinite in stop_teration_case_generator():
        yield check_stop_iter, fw_iter, iter_name, batch_size, epochs, iter_num, auto_reset, infinite

def test_stop_iteration_paddle_fail_multi():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator
    fw_iter = lambda pipe, size, auto_reset : PaddleIterator(pipe, output_map=["data"],  size=size, auto_reset=auto_reset)
    check_stop_iter_fail_multi(fw_iter)

def test_stop_iteration_paddle_fail_single():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator
    fw_iter = lambda pipe, size, auto_reset : PaddleIterator(pipe, output_map=["data"],  size=size, auto_reset=auto_reset)
    check_stop_iter_fail_single(fw_iter)
