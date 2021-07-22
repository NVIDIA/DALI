import sys
import copy
from glob import glob
import math
import os

import torch
from torch.utils.data import DataLoader

from dataloading.datasets import imageDataset

from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin import pytorch
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def
def create_video_reader_pipeline(sequence_length, files, crop_size):
    images = fn.readers.video(device="gpu", filenames=files, sequence_length=sequence_length,
                              normalized=False, random_shuffle=True, image_type=types.RGB,
                              dtype=types.UINT8, initial_fill=16, pad_last_batch=True, name="Reader")
    images = fn.crop(images, crop=crop_size, dtype=types.FLOAT,
                     crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
                     crop_pos_y=fn.random.uniform(range=(0.0, 1.0)))

    images = fn.transpose(images, perm=[3, 0, 1, 2])

    return images


class DALILoader():
    def __init__(self, batch_size, file_root, sequence_length, crop_size):
        container_files = os.listdir(file_root)
        container_files = [file_root + '/' + f for f in container_files]
        self.pipeline = create_video_reader_pipeline(batch_size=batch_size,
                                                     sequence_length=sequence_length,
                                                     num_threads=2,
                                                     device_id=0,
                                                     files=container_files,
                                                     crop_size=crop_size)
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
                                                         ["data"],
                                                         reader_name="Reader",
                                                         last_batch_policy=pytorch.LastBatchPolicy.PARTIAL,
                                                         auto_reset=True)

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.dali_iterator.__iter__()


def get_loader(args, ds_type):
    if ds_type not in ('train', 'val'):
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
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
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
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
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
