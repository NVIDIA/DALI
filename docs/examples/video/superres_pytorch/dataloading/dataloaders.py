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
    def __init__(self, batch_size, sequence_length, num_threads, device_id, files, crop_size):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.reader = ops.VideoReader(device="gpu", filenames=files, sequence_length=sequence_length, normalized=False,
                                     random_shuffle=True, image_type=types.RGB, dtype=types.UINT8, initial_fill=16)
        self.crop = ops.CropCastPermute(device="gpu", crop=crop_size, output_layout=types.NHWC, output_dtype=types.FLOAT)
        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])

    def define_graph(self):
        input = self.reader(name="Reader")
        cropped = self.crop(input, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.transpose(cropped)
        return output

class DALILoader():
    def __init__(self, batch_size, file_root, sequence_length, crop_size):
        container_files = os.listdir(file_root)
        container_files = [file_root + '/' + f for f in container_files]
        self.pipeline = VideoReaderPipeline(batch_size=batch_size,
                                            sequence_length=sequence_length,
                                            num_threads=2,
                                            device_id=0,
                                            files=container_files,
                                            crop_size=crop_size)
        self.pipeline.build()
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

            if args.world_size > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = torch.utils.data.RandomSampler(dataset)

            loader = DataLoader(
                dataset,
                batch_size=args.batchsize,
                shuffle=(sampler is None),
                num_workers=0,
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

            if args.world_size > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = torch.utils.data.RandomSampler(dataset)

            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                sampler=sampler,
                drop_last=True)

            batches = math.ceil(len(dataset) / float(args.world_size))

    elif args.loader == 'DALI':
        loader = DALILoader(args.batchsize,
            os.path.join(args.root, ds_type),
            args.frames,
            args.crop_size)
        batches = len(loader)
        sampler = None

    else:
        raise ValueError('%s is not a valid option for --loader' % args.loader)

    return loader, batches, sampler
