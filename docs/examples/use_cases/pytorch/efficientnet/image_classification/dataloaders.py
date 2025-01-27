# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import torch
import numpy as np
from PIL import Image
from functools import partial

from image_classification.autoaugment import AutoaugmentImageNetPolicy

DATA_BACKEND_CHOICES = ["pytorch", "pytorch_optimized", "synthetic"]
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy
    import nvidia.dali.types as types

    from image_classification.dali import (
        training_pipe,
        training_pipe_external_source,
        validation_pipe,
        validation_pipe_external_source,
    )

    DATA_BACKEND_CHOICES.append("dali")
    DATA_BACKEND_CHOICES.append("dali_proxy")
except ImportError as e:
    print(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )

import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_jpeg_from_file(path, cuda=True):
    img_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


class DALIWrapper(object):
    @staticmethod
    def gen_wrapper(loader, num_classes, one_hot, memory_format):
        for data in loader:
            if memory_format == torch.channels_last:
                # If we requested the data in channels_last form, utilize the fact that DALI
                # can return it as NHWC. The network expects NCHW shape with NHWC internal memory,
                # so we can keep the memory and just create a view with appropriate shape and
                # strides reflacting that memory layouyt
                shape = data[0]["data"].shape
                stride = data[0]["data"].stride()

                # permute shape and stride from NHWC to NCHW
                def nhwc_to_nchw(t):
                    return t[0], t[3], t[1], t[2]

                input = torch.as_strided(
                    data[0]["data"],
                    size=nhwc_to_nchw(shape),
                    stride=nhwc_to_nchw(stride),
                )
            else:
                input = data[0]["data"].contiguous(memory_format=memory_format)
            target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            if one_hot:
                target = expand(num_classes, torch.float, target)
            yield input, target

        loader.reset()

    def __init__(self, loader, num_classes, one_hot, memory_format):
        self.loader = loader
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.memory_format = memory_format

    def __iter__(self):
        return DALIWrapper.gen_wrapper(
            self.loader,
            self.num_classes,
            self.one_hot,
            self.memory_format,
        )


def get_dali_train_loader(dali_device="gpu"):
    def gdtl(
        data_path,
        image_size,
        batch_size,
        num_classes,
        one_hot,
        interpolation="bilinear",
        augmentation="disabled",
        start_epoch=0,
        workers=5,
        _worker_init_fn=None,
        memory_format=torch.contiguous_format,
        **kwargs,
    ):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]

        output_layout = "HWC" if memory_format == torch.channels_last else "CHW"

        traindir = os.path.join(data_path, "train")

        pipeline_kwargs = {
            "batch_size": batch_size,
            "num_threads": workers,
            "device_id": rank % torch.cuda.device_count(),
            "seed": 12 + rank % torch.cuda.device_count(),
        }

        pipe = training_pipe(
            data_dir=traindir,
            interpolation=interpolation,
            image_size=image_size,
            output_layout=output_layout,
            automatic_augmentation=augmentation,
            dali_device=dali_device,
            rank=rank,
            world_size=world_size,
            **pipeline_kwargs,
        )

        train_loader = DALIClassificationIterator(
            pipe, reader_name="Reader", fill_last_batch=False
        )

        return (
            DALIWrapper(train_loader, num_classes, one_hot, memory_format),
            int(pipe.epoch_size("Reader") / (world_size * batch_size)),
        )

    return gdtl


def get_dali_val_loader():
    def gdvl(
        data_path,
        image_size,
        batch_size,
        num_classes,
        one_hot,
        interpolation="bilinear",
        crop_padding=32,
        workers=5,
        _worker_init_fn=None,
        memory_format=torch.contiguous_format,
        **kwargs,
    ):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]

        output_layout = "HWC" if memory_format == torch.channels_last else "CHW"

        valdir = os.path.join(data_path, "val")

        pipeline_kwargs = {
            "batch_size": batch_size,
            "num_threads": workers,
            "device_id": rank % torch.cuda.device_count(),
            "seed": 12 + rank % torch.cuda.device_count(),
        }

        pipe = validation_pipe(
            data_dir=valdir,
            interpolation=interpolation,
            image_size=image_size + crop_padding,
            image_crop=image_size,
            output_layout=output_layout,
            **pipeline_kwargs,
        )

        val_loader = DALIClassificationIterator(
            pipe, reader_name="Reader", fill_last_batch=False
        )

        return (
            DALIWrapper(val_loader, num_classes, one_hot, memory_format),
            int(pipe.epoch_size("Reader") / (world_size * batch_size)),
        )

    return gdvl


def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    # imgs is a torch tensor [c, h, w]
    h = imgs[0].size()[1]
    w = imgs[0].size()[2]

    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(
        tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda")
    )
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e


def as_memory_format(next_input, memory_format):
    if memory_format == torch.channels_last:
        shape = next_input.shape
        stride = next_input.stride()

        # permute shape and stride from NHWC to NCHW
        def nhwc_to_nchw(t):
            return t[0], t[3], t[1], t[2]

        next_input = torch.as_strided(
            next_input,
            size=nhwc_to_nchw(shape),
            stride=nhwc_to_nchw(stride),
        )
    elif memory_format == torch.contiguous_format:
        next_input = next_input.contiguous(memory_format=memory_format)
    return next_input


class PrefetchedWrapper(object):
    @staticmethod
    def prefetched_loader(loader, num_classes, one_hot, memory_format):
        stream = torch.cuda.Stream()
        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = as_memory_format(
                    next_input, memory_format=memory_format
                ).to(device="cuda")
                next_target = next_target.to(device="cuda")
                next_input = next_input.float()
                if one_hot:
                    next_target = expand(num_classes, torch.float, next_target)
            yield next_input, next_target

    def __init__(
        self, dataloader, start_epoch, num_classes, one_hot, memory_format=None
    ):
        self.dataloader = dataloader
        self.epoch = start_epoch
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.memory_format = memory_format

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler,
            torch.utils.data.distributed.DistributedSampler,
        ):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(
            self.dataloader, self.num_classes, self.one_hot, self.memory_format
        )

    def __len__(self):
        return len(self.dataloader)


def get_pytorch_train_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation="bilinear",
    augmentation=None,
    start_epoch=0,
    workers=5,
    _worker_init_fn=None,
    prefetch_factor=2,
    memory_format=torch.contiguous_format,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    traindir = os.path.join(data_path, "train")
    transform_list = [
        transforms.RandomResizedCrop(image_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation == "disabled":
        pass
    elif augmentation == "autoaugment":
        transform_list.append(AutoaugmentImageNetPolicy())
    else:
        raise NotImplementedError(
            f"Automatic augmentation: '{augmentation}' is not supported"
            " for PyTorch data loader."
        )

    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    train_dataset = datasets.ImageFolder(
        traindir, transforms.Compose(transform_list)
    )

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=partial(fast_collate, memory_format),
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return (
        PrefetchedWrapper(
            train_loader,
            start_epoch,
            num_classes,
            one_hot,
        ),
        len(train_loader),
    )


def get_pytorch_val_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation="bilinear",
    workers=5,
    _worker_init_fn=None,
    crop_padding=32,
    memory_format=torch.contiguous_format,
    prefetch_factor=2,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    valdir = os.path.join(data_path, "val")
    transform_list = [
        transforms.Resize(
            image_size + crop_padding, interpolation=interpolation
        ),
        transforms.CenterCrop(image_size),
    ]
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(transform_list),
    )

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=partial(fast_collate, memory_format),
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return PrefetchedWrapper(val_loader, 0, num_classes, one_hot), len(
        val_loader
    )


def fast_optimized_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    # imgs is a PILow image [w, h]
    w = imgs[0].size[0]
    h = imgs[0].size[1]

    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, targets


class PrefetchedOptimizedWrapper(object):
    @staticmethod
    def prefetched_loader(loader, num_classes, one_hot):
        mean = (
            torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )
        std = (
            torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                if one_hot:
                    next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, start_epoch, num_classes, one_hot):
        self.dataloader = dataloader
        self.epoch = start_epoch
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler,
            torch.utils.data.distributed.DistributedSampler,
        ):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedOptimizedWrapper.prefetched_loader(
            self.dataloader, self.num_classes, self.one_hot
        )

    def __len__(self):
        return len(self.dataloader)


def get_pytorch_optimized_train_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation="bilinear",
    augmentation=None,
    start_epoch=0,
    workers=5,
    _worker_init_fn=None,
    prefetch_factor=2,
    memory_format=torch.contiguous_format,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    traindir = os.path.join(data_path, "train")
    transform_list = [
        transforms.RandomResizedCrop(image_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation == "disabled":
        pass
    elif augmentation == "autoaugment":
        transform_list.append(AutoaugmentImageNetPolicy())
    else:
        raise NotImplementedError(
            f"Automatic augmentation: '{augmentation}' is not supported"
            " for PyTorch data loader."
        )

    train_dataset = datasets.ImageFolder(
        traindir, transforms.Compose(transform_list)
    )

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=partial(fast_optimized_collate, memory_format),
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return (
        PrefetchedOptimizedWrapper(
            train_loader,
            start_epoch,
            num_classes,
            one_hot,
        ),
        len(train_loader),
    )


def get_pytorch_optimize_val_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation="bilinear",
    workers=5,
    _worker_init_fn=None,
    crop_padding=32,
    memory_format=torch.contiguous_format,
    prefetch_factor=2,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    valdir = os.path.join(data_path, "val")
    transform_list = [
        transforms.Resize(
            image_size + crop_padding, interpolation=interpolation
        ),
        transforms.CenterCrop(image_size),
    ]
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(transform_list),
    )

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=partial(fast_optimized_collate, memory_format),
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return PrefetchedOptimizedWrapper(val_loader, 0, num_classes, one_hot), len(
        val_loader
    )


def read_file(path):
    return np.fromfile(path, dtype=np.uint8)


def read_filepath(path):
    return np.frombuffer(path.encode(), dtype=np.int8)


def get_dali_proxy_train_loader(dali_device="gpu"):
    def get_impl(
        data_path,
        image_size,
        batch_size,
        num_classes,
        one_hot,
        interpolation="bilinear",
        augmentation=None,
        start_epoch=0,
        workers=5,
        _worker_init_fn=None,
        prefetch_factor=2,
        memory_format=torch.contiguous_format,
    ):
        interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]

        output_layout = "HWC" if memory_format == torch.channels_last else "CHW"

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )

        pipeline_kwargs = {
            "batch_size": batch_size,
            "num_threads": workers,
            "device_id": rank % torch.cuda.device_count(),
            "seed": 12 + rank % torch.cuda.device_count(),
        }

        pipe = training_pipe_external_source(
            interpolation=interpolation,
            image_size=image_size,
            output_layout=output_layout,
            automatic_augmentation=augmentation,
            dali_device=dali_device,
            prefetch_queue_depth=8,
            **pipeline_kwargs,
        )

        dali_server = dali_proxy.DALIServer(pipe)

        train_dataset = datasets.ImageFolder(
            os.path.join(data_path, "train"),
            transform=dali_server.proxy,
            loader=read_filepath,
        )

        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True
            )
        else:
            train_sampler = None

        train_loader = dali_proxy.DataLoader(
            dali_server,
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            collate_fn=None,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )

        return (
            PrefetchedWrapper(
                train_loader, start_epoch, num_classes, one_hot, memory_format
            ),
            len(train_loader),
        )

    return get_impl


def get_dali_proxy_val_loader(dali_device="gpu"):
    def get_impl(
        data_path,
        image_size,
        batch_size,
        num_classes,
        one_hot,
        interpolation="bilinear",
        workers=5,
        _worker_init_fn=None,
        crop_padding=32,
        memory_format=torch.contiguous_format,
        prefetch_factor=2,
    ):
        interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]

        output_layout = "HWC" if memory_format == torch.channels_last else "CHW"

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        pipeline_kwargs = {
            "batch_size": batch_size,
            "num_threads": workers,
            "device_id": rank % torch.cuda.device_count(),
            "seed": 12 + rank % torch.cuda.device_count(),
        }

        pipe = validation_pipe_external_source(
            interpolation=interpolation,
            image_size=image_size + crop_padding,
            image_crop=image_size,
            output_layout=output_layout,
            **pipeline_kwargs,
        )

        dali_server = dali_proxy.DALIServer(pipe)
        val_dataset = datasets.ImageFolder(
            os.path.join(data_path, "val"),
            transform=dali_server.proxy,
            loader=read_filepath,
        )

        if torch.distributed.is_initialized():
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False
            )
        else:
            val_sampler = None

        val_loader = dali_proxy.DataLoader(
            dali_server,
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size,
            shuffle=(val_sampler is None),
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            collate_fn=None,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )

        return (
            PrefetchedWrapper(
                val_loader, 0, num_classes, one_hot, memory_format
            ),
            len(val_loader),
        )

    return get_impl


class SynteticDataLoader(object):
    def __init__(
        self,
        batch_size,
        num_classes,
        num_channels,
        height,
        width,
        one_hot,
        memory_format=torch.contiguous_format,
    ):
        input_data = (
            torch.randn(batch_size, num_channels, height, width)
            .contiguous(memory_format=memory_format)
            .cuda()
            .normal_(0, 1.0)
        )
        if one_hot:
            input_target = torch.empty(batch_size, num_classes).cuda()
            input_target[:, 0] = 1.0
        else:
            input_target = torch.randint(0, num_classes, (batch_size,))
        input_target = input_target.cuda()

        self.input_data = input_data
        self.input_target = input_target

    def __iter__(self):
        while True:
            yield self.input_data, self.input_target


def get_synthetic_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation=None,
    augmentation=None,
    start_epoch=0,
    workers=None,
    _worker_init_fn=None,
    memory_format=torch.contiguous_format,
    **kwargs,
):
    return (
        SynteticDataLoader(
            batch_size,
            num_classes,
            3,
            image_size,
            image_size,
            one_hot,
            memory_format=memory_format,
        ),
        -1,
    )
