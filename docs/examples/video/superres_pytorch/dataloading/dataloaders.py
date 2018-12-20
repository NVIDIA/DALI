import sys
import copy
from glob import glob
import math
import os

import torch
from torch.utils.data import DataLoader

from dataloading.datasets import imageDataset

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, num_threads, device_id, files):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.VideoReader(device="gpu", filenames=files, sequence_length=sequence_length,
                                     shard_id=device_id, num_shards=1, random_shuffle=True,
                                     initial_fill=16)


    def define_graph(self):
        output = self.input(name="Reader")
        return output

class DALILoader():
    def __init__(self, batch_size, file_root, sequence_length):
        container_files = os.listdir(file_root)
        container_files = [file_root + '/' + f for f in container_files]
        print(container_files)
        self.pipeline = VideoReaderPipeline(batch_size=batch_size,
                                            sequence_length=sequence_length,
                                            num_threads=2,
                                            device_id=0,
                                            files=container_files)
        print("after")
        self.pipeline.build()
        print("after build")
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
                                                         ["data"],
                                                         self.epoch_size,
                                                         auto_reset=True)
    def __len__(self):
        return self.epoch_size
    def __iter__(self):
        return self.dali_iterator.__iter__()


def get_loader(args, ds_type):
    if ds_type is not 'train' and ds_type is not 'val':
        raise ValueError("ds_type has to be either 'train' or 'val'")

    if args.loader == 'pytorch':

        if ds_type == 'train':
            dataset = imageDataset(
                args.frames,
                args.is_cropped,
                args.crop_size,
                os.path.join(args.root, 'train'),
                args.batchsize,
                args.world_size)

            loader = DataLoader(
                dataset,
                batch_size=args.batchsize,
                shuffle=(sampler is None),
                num_workers=10,
                pin_memory=True,
                sampler=sampler,
                drop_last=True)

            effective_bsz = args.batchsize * float(args.world_size)
            batches = math.ceil(len(dataset) / float(effective_bsz))

        if ds_type == 'val':
            dataset = imageDataset(
                args.frames,
                args.is_cropped,
                args.crop_size,
                os.path.join(args.root, 'val'),
                args.batchsize,
                args.world_size)

            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                sampler=sampler,
                drop_last=True)

            batches = math.ceil(len(dataset) / float(args.world_size))

        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    elif args.loader == 'DALI':

        print(ds_type + " loader init")
        loader = DALILoader(args.batchsize,
            os.path.join(args.root, ds_type),
            args.frames)
        print(ds_type + " loader getting length")
        batches = len(loader)
        sampler = None

    else:
        raise ValueError('%s is not a valid option for --loader' % args.loader)

    return loader, batches, sampler
