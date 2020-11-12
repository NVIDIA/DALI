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

import os
import argparse
import time
import cv2
import numpy as np

import nvidia.dali as dali
import nvidia.dali.types as types
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from test_utils import AverageMeter


class FilesDataSet:

    def __init__(self, data_path):
        files = self.get_files_list(data_path)
        self.data_path = data_path
        self.classes_no = len(files)
        self.files_map = {}

        counter = 0
        for class_no, (_, samples) in enumerate(files):
            for sample in samples:
                self.files_map[counter] = (sample, class_no)
                counter += 1
        self.size = counter

    @classmethod
    def get_files_list(cls, data_path):
        dirs = [
            (dir_name, [
                file_path for file_path
                in map(lambda name: os.path.join(dir_path, name), os.listdir(dir_path))
                if os.path.isfile(file_path)
            ])
            for (dir_name, dir_path)
            in map(lambda name: (name, os.path.join(data_path, name)), os.listdir(data_path))
            if os.path.isdir(dir_path)
        ]
        dirs.sort(key=lambda dir_files: dir_files[0])
        for _, files in dirs:
            files.sort()
        return dirs

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i >= self.size:
            raise StopIteration
        return self.files_map[i]


class SampleLoader(object):

    DATA_SET = FilesDataSet

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_set = self.create_data_set()

    def __getattr__(self, key):
        if key == 'data_set':
            self._data_set = self._data_set or self.create_data_set()
            self.__dict__['data_set'] = self._data_set
            return self._data_set
        raise AttributeError

    def create_data_set(self):
        return self.DATA_SET(data_path=self.data_path)

    def read_file(self, file_path):
        with open(file_path, 'rb') as f:
            return f.read()

    def __call__(self, sample_info):
        file_path, class_no = self.data_set[sample_info.idx_in_epoch]
        return self.read_file(file_path), np.array([class_no])


class ShuffledFilesDataSet(FilesDataSet):

        def __init__(self, data_path):
            super().__init__(data_path)
            self.perm = np.random.permutation(self.size)

        def __getitem__(self, i):
            if i >= self.size:
                raise StopIteration
            return super().__getitem__(self.perm[i])


class CV2SampleLoader(SampleLoader):

    def read_file(self, file_path):
        img = cv2.imread(file_path)
        return img


class CV2SampleLoaderPerm(CV2SampleLoader):

    DATA_SET = ShuffledFilesDataSet


def common_pipeline(images):
    images = dali.fn.random_resized_crop(images, device="gpu", size =(224,224))
    rng = dali.fn.coin_flip(probability=0.5)
    images = dali.fn.crop_mirror_normalize(
        images, mirror=rng,
        device="gpu",
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        crop=(224, 224),
        mean=[125, 125, 125],
        std=[255, 255, 255])
    return images


def file_reader_pipeline(data_path, batch_size, num_threads, device_id, prefetch, reader_queue_depth, **kwargs,):
    pipe = dali.pipeline.Pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, prefetch_queue_depth=prefetch)
    with pipe:
        images, labels = dali.fn.file_reader(
            name="Reader",
            file_root=data_path,
            prefetch_queue_depth=reader_queue_depth,
            random_shuffle=True,)
        images = dali.fn.image_decoder(images, device="cpu", output_type=types.RGB)
        images = common_pipeline(images.gpu())
        pipe.set_outputs(images, labels)
    return pipe


class ExternalSourcePipeline(dali.pipeline.Pipeline):

    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.loader = CV2SampleLoaderPerm(data_path)

    def epoch_size(self, *args, **kwargs):
        return len(self.loader.data_set)


def external_source_pipeline(data_path, batch_size, num_threads, device_id, prefetch, reader_queue_depth, **kwargs,):
    pipe = ExternalSourcePipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, data_path=data_path)
    with pipe:
        images, labels = dali.fn.external_source(pipe.loader, batch=False, num_outputs=2)
        images = common_pipeline(images.gpu())
        pipe.set_outputs(images, labels)
    return pipe


def external_source_parallel_pipeline(data_path, batch_size, num_threads, device_id, prefetch, reader_queue_depth,
        py_workers_num=None, py_workers_init="fork"):
    pipe = ExternalSourcePipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, prefetch_queue_depth=prefetch,
        py_workers_init=py_workers_init, py_workers_num=py_workers_num, data_path=data_path)
    with pipe:
        images, labels = dali.fn.external_source(pipe.loader, num_outputs=2, batch=False, parallel=True, prefetch_queue_depth=reader_queue_depth)
        images = common_pipeline(images.gpu())
        pipe.set_outputs(images, labels)
    return pipe


TEST_DATA = [
    external_source_parallel_pipeline,
    file_reader_pipeline,
    external_source_pipeline,
]

def parse_args():

    parser = argparse.ArgumentParser(description='Compare external source vs filereader performance in RN50 data pipeline case')
    parser.add_argument('data_path', type=str, help='Directory path of training dataset')
    parser.add_argument('-b', '--batch_size', default=1024, type=int, metavar='N',
                        help='batch size (default: 1024)')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 3)')
    parser.add_argument('--py_workers', default=3, type=int, metavar='N',
                        help='number of python external source workers (default: 3)')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='Number of epochs to run')
    parser.add_argument('--assign_gpu', default=0, type=int, metavar='N',
                        help='Assign a given GPU. Cannot be used with --gpus')
    parser.add_argument('--worker_init', default='fork', choices=['fork', 'spawn'], type=str,
                        help='Python workers initialization method')
    parser.add_argument('--prefetch', default=2, type=int, metavar='N',
                        help='Pipeline cpu/gpu prefetch queue depth (default: 2)')
    parser.add_argument('--reader_queue_depth', default=1, type=int, metavar='N',
                        help='Depth of prefetching queue for file reading operators (FileReader/parallel ExternalSource) (default: 1)')
    parser.add_argument('--training', default=False, type=bool,
                        help='When specified to True pipeline is run alongside RN18 training, otherwise only data pipeline is run.')
    args = parser.parse_args()

    print("GPU ID: {}, batch: {}, epochs: {}, workers: {}, py_workers: {}, prefetch depth: {}, reader_queue_depth: {}, worker_init: {}"
        .format(args.assign_gpu, args.batch_size, args.epochs, args.workers, args.py_workers, args.prefetch, args.reader_queue_depth, args.worker_init))

    return args


def iteration_test(args):

    for pipe_factory in TEST_DATA:
        pipe = pipe_factory(
            batch_size=args.batch_size,
            num_threads=args.workers,
            device_id=args.assign_gpu,
            data_path=args.data_path,
            prefetch=args.prefetch,
            reader_queue_depth=args.reader_queue_depth,
            py_workers_init=args.worker_init,
            py_workers_num=args.py_workers
            )
        pipe.build()

        samples_no = pipe.epoch_size("Reader")
        expected_iters = samples_no // args.batch_size + (samples_no % args.batch_size != 0)

        print("RUN {}".format(pipe_factory.__name__))
        for i in range(args.epochs):
            if i == 0:
                print("Warm up")
            else:
                print("Test run " + str(i))
            data_time = AverageMeter()
            end = time.time()
            for j in range(expected_iters):
                try:
                    pipe.run()
                except StopIteration:
                    assert j == expected_iters - 1
                    break
                data_time.update(time.time() - end)
                if j % 10 == 0:
                    print("{} {}/ {}, avg time: {} [s], worst time: {} [s], speed: {} [img/s]".format(
                            pipe_factory.__name__,
                            j,
                            expected_iters,
                            data_time.avg,
                            data_time.max_val,
                            args.batch_size / data_time.avg,
                    ))
                end = time.time()
            pipe.reset()

        print("OK {}".format(pipe_factory.__name__))


def training_test(args):

    import torch
    import torch.nn as nn
    import torch.optim
    import torchvision.models as models
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    for pipe_factory in TEST_DATA:
        pipe = pipe_factory(
            batch_size=args.batch_size,
            num_threads=args.workers,
            device_id=args.assign_gpu,
            data_path=args.data_path,
            prefetch=args.prefetch,
            reader_queue_depth=args.reader_queue_depth,
            py_workers_init=args.worker_init,
            py_workers_num=args.py_workers
            )
        pipe.build()

        model =  models.resnet18().cuda()
        model.train()
        loss_fun = nn.CrossEntropyLoss().cuda()
        lr = 0.1 * args.batch_size / 256
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr,
            momentum=0.9,
        )

        samples_no = pipe.epoch_size("Reader")
        expected_iters = samples_no // args.batch_size + (samples_no % args.batch_size != 0)

        end = time.time()
        if pipe_factory == file_reader_pipeline:
            iterator = DALIClassificationIterator([pipe], reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP)
        else:
            iterator = DALIClassificationIterator([pipe], auto_reset=True)

        print("RUN {}".format(pipe_factory.__name__))
        losses = AverageMeter()
        for i in range(args.epochs):
            if i == 0:
                print("Warm up")
            else:
                print("Test run " + str(i))
            data_time = AverageMeter()
            for j, data in enumerate(iterator):
                inputs = data[0]["data"]
                target = data[0]["label"].squeeze(-1).cuda().long()
                if data[0]["label"].squeeze(-1).min() < 0:
                    print(data[0]["label"].squeeze(-1))
                outputs = model(inputs)
                loss = loss_fun(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                data_time.update(time.time() - end)

                if j % 50 == 0:
                    reduced_loss = loss.data
                    print(reduced_loss.item())
                    losses.update(reduced_loss.item())

                    print("{} {}/ {}, avg time: {} [s], worst time: {} [s], speed: {} [img/s], loss: {}, loss_avg: {}".format(
                            pipe_factory.__name__,
                            j,
                            expected_iters,
                            data_time.avg,
                            data_time.max_val,
                            args.batch_size / data_time.avg,
                            reduced_loss.item(),
                            losses.avg
                    ))
                end = time.time()
            if pipe_factory == file_reader_pipeline:
                iterator.reset()

        print("OK {}".format(pipe_factory.__name__))


if __name__ == "__main__":

    args = parse_args()
    if args.training:
        training_test(args)
    else:
        iteration_test(args)
