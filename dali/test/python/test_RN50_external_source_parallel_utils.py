# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    def get_sample(self, sample_idx, epoch_idx):
        if sample_idx >= self.size:
            raise StopIteration
        return self.files_map[sample_idx]


class ShuffledFilesDataSet(FilesDataSet):

    def __init__(self, data_path):
        super().__init__(data_path)
        self.rng = np.random.default_rng(seed=42)
        self.epoch_idx = 0
        self.perm = self.rng.permutation(self.size)

    def get_sample(self, sample_idx, epoch_idx):
        if self.epoch_idx != epoch_idx:
            self.perm = self.rng.permutation(self.size)
            self.epoch_idx = epoch_idx
        if sample_idx >= self.size:
            raise StopIteration
        return super().get_sample(self.perm[sample_idx], epoch_idx)


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
        file_path, class_no = self.data_set.get_sample(sample_info.idx_in_epoch, sample_info.iteration)
        return self.read_file(file_path), np.array([class_no])


class CV2SampleLoader(SampleLoader):

    def read_file(self, file_path):
        img = cv2.imread(file_path)
        return img


class CV2SampleLoaderPerm(CV2SampleLoader):

    DATA_SET = ShuffledFilesDataSet


def common_pipeline(images):
    images = dali.fn.random_resized_crop(images, device="gpu", size=(224, 224))
    rng = dali.fn.random.coin_flip(probability=0.5)
    images = dali.fn.crop_mirror_normalize(
        images, mirror=rng,
        device="gpu",
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        crop=(224, 224),
        mean=[125, 125, 125],
        std=[255, 255, 255])
    return images


def file_reader_pipeline(data_path, batch_size, num_threads, device_id, prefetch,
                         reader_queue_depth, **kwargs,):
    pipe = dali.pipeline.Pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id,
        prefetch_queue_depth=prefetch)
    with pipe:
        images, labels = dali.fn.readers.file(
            name="Reader",
            file_root=data_path,
            prefetch_queue_depth=reader_queue_depth,
            random_shuffle=True,)
        images = dali.fn.decoders.image(images, device="cpu", output_type=types.RGB)
        images = common_pipeline(images.gpu())
        pipe.set_outputs(images, labels)
    return pipe


class ExternalSourcePipeline(dali.pipeline.Pipeline):

    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.loader = CV2SampleLoaderPerm(data_path)

    def epoch_size(self, *args, **kwargs):
        return len(self.loader.data_set)


def external_source_pipeline(
        data_path, batch_size, num_threads, device_id, prefetch, reader_queue_depth, **kwargs,):
    pipe = ExternalSourcePipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, data_path=data_path)
    with pipe:
        images, labels = dali.fn.external_source(pipe.loader, batch=False, num_outputs=2)
        images = common_pipeline(images.gpu())
        pipe.set_outputs(images, labels)
    return pipe


def external_source_parallel_pipeline(
        data_path, batch_size, num_threads, device_id, prefetch, reader_queue_depth,
        py_num_workers=None, py_start_method="fork"):
    pipe = ExternalSourcePipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id,
        prefetch_queue_depth=prefetch, py_start_method=py_start_method,
        py_num_workers=py_num_workers, data_path=data_path)
    with pipe:
        images, labels = dali.fn.external_source(
            pipe.loader, num_outputs=2, batch=False, parallel=True,
            prefetch_queue_depth=reader_queue_depth)
        images = common_pipeline(images.gpu())
        pipe.set_outputs(images, labels)
    return pipe


def get_pipe_factories(test_pipes, parallel_pipe, file_reader_pipe, scalar_pipe):
    result = []
    if "parallel" in test_pipes:
        result.append(parallel_pipe)
    if "file_reader" in test_pipes:
        result.append(file_reader_pipe)
    if "scalar" in test_pipes:
        result.append(scalar_pipe)
    return result

def parse_test_arguments(supports_distributed):

    parser = argparse.ArgumentParser(
        description='Compare external source vs filereader performance in RN50 data pipeline case')
    parser.add_argument('data_path', type=str, help='Directory path of training dataset')
    parser.add_argument('-b', '--batch_size', default=1024, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 3)')
    parser.add_argument('--py_workers', default=3, type=int, metavar='N',
                        help='number of python external source workers (default: 3)')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='Number of epochs to run')
    parser.add_argument('--benchmark_iters', type=int, metavar='N',
                        help='Number of iterations to run in each epoch')
    parser.add_argument('--worker_init', default='fork', choices=['fork', 'spawn'], type=str,
                        help='Python workers initialization method')
    parser.add_argument('--prefetch', default=2, type=int, metavar='N',
                        help='Pipeline cpu/gpu prefetch queue depth')
    parser.add_argument(
        '--reader_queue_depth', default=1, type=int, metavar='N',
        help='Depth of prefetching queue for file reading operators (FileReader/parallel ExternalSource)')
    parser.add_argument(
        "--test_pipes", nargs="+", default=["parallel", "file_reader", "scalar"],
        help="Pipelines to be tested, allowed values: 'parallel', 'file_reader', 'scalar'")

    if supports_distributed:
        parser.add_argument('--local_rank', default=0, type=int,
            help="Id of the local rank in distributed scenario.")
    else:
        parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N',
            help='number of GPUs')
    args = parser.parse_args()

    if supports_distributed:
        print("GPU ID: {}, batch: {}, epochs: {}, workers: {}, py_workers: {}, prefetch depth: {}, reader_queue_depth: {}, worker_init: {}, test_pipes: {}" .format(
            args.local_rank, args.batch_size, args.epochs, args.workers, args.py_workers, args.prefetch, args.reader_queue_depth, args.worker_init, args.test_pipes))
    else:
        print("GPUS: {}, batch: {}, epochs: {}, workers: {}, py_workers: {}, prefetch depth: {}, reader_queue_depth: {}, worker_init: {}, test_pipes: {}" .format(
            args.gpus, args.batch_size, args.epochs, args.workers, args.py_workers, args.prefetch, args.reader_queue_depth, args.worker_init, args.test_pipes))

    return args
