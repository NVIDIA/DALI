#!/usr/bin/python3
# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim
import torchvision.models as models
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from torch.nn.parallel import DistributedDataParallel as DDP

from test_RN50_external_source_parallel_utils import (
    parse_test_arguments,
    external_source_parallel_pipeline,
    external_source_pipeline,
    file_reader_pipeline,
    get_pipe_factories,
)
from test_utils import AverageMeter

# This test requires significant amount of shared memory to be able to pass
# the batches between worker processes and the main process. If running in docker
# make sure that -shm-size is big enough.

# We place the parallel External Source as first as we need to fork before we call anything
# from cuda.
TEST_PIPES_FACTORIES = [
    external_source_parallel_pipeline,
    file_reader_pipeline,
    external_source_pipeline,
]


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


def training_test(args):
    """Run ExternalSource pipelines along RN18 network. Based on simplified RN50 Pytorch sample."""

    args.distributed = False
    args.world_size = 1
    args.gpu = 0
    args.distributed_initialized = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    test_pipe_factories = get_pipe_factories(
        args.test_pipes,
        external_source_parallel_pipeline,
        file_reader_pipeline,
        external_source_pipeline,
    )

    for pipe_factory in test_pipe_factories:
        pipe = pipe_factory(
            batch_size=args.batch_size,
            num_threads=args.workers,
            device_id=args.local_rank,
            data_path=args.data_path,
            prefetch_queue_depth=args.prefetch,
            reader_queue_depth=args.reader_queue_depth,
            py_start_method=args.worker_init,
            py_num_workers=args.py_workers,
            source_mode=args.source_mode,
            read_encoded=args.dali_decode,
        )

        # Start the pipeline workers first, before any CUDA call. The first pipeline factory
        # is the one with Parallel External Source that needs that.
        pipe.start_py_workers()

        if args.distributed and not args.distributed_initialized:
            args.gpu = args.local_rank
            torch.cuda.set_device(args.gpu)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            args.world_size = torch.distributed.get_world_size()
            args.distributed_initialized = True

        pipe.build()

        model = models.resnet18().cuda()
        if args.distributed:
            model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)
        model.train()
        loss_fun = nn.CrossEntropyLoss().cuda()
        lr = 0.1 * args.batch_size / 256
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)

        samples_no = pipe.epoch_size("Reader")
        if args.benchmark_iters is None:
            expected_iters = samples_no // args.batch_size + (samples_no % args.batch_size != 0)
        else:
            expected_iters = args.benchmark_iters

        if pipe_factory == file_reader_pipeline:
            iterator = DALIClassificationIterator(
                [pipe],
                reader_name="Reader",
                last_batch_policy=LastBatchPolicy.DROP,
                auto_reset=True,
            )
        else:
            iterator = DALIClassificationIterator(
                [pipe], size=samples_no * args.world_size, auto_reset=True
            )

        if args.local_rank == 0:
            print("RUN {}".format(pipe_factory.__name__))
        losses = AverageMeter()
        for i in range(args.epochs):
            if args.local_rank == 0:
                if i == 0:
                    print("Warm up")
                else:
                    print("Test run " + str(i))
            end = time.time()
            data_time = AverageMeter()
            for j, data in enumerate(iterator):
                inputs = data[0]["data"]
                target = data[0]["label"].squeeze(-1).cuda().long()
                outputs = model(inputs)
                loss = loss_fun(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if j % 50 == 0 and j != 0:
                    if args.distributed:
                        reduced_loss = reduce_tensor(loss.data)
                    else:
                        reduced_loss = loss.data

                    if args.local_rank == 0:
                        print(reduced_loss.item())
                    losses.update(reduced_loss.item())

                    torch.cuda.synchronize()

                    data_time.update((time.time() - end) / 50)
                    end = time.time()
                    if args.local_rank == 0:
                        template_string = (
                            "{} {}/ {}, avg time: {} [s], worst time: {} [s], "
                            "speed: {} [img/s], loss: {}, loss_avg: {}"
                        )
                        print(
                            template_string.format(
                                pipe_factory.__name__,
                                j,
                                expected_iters,
                                data_time.avg,
                                data_time.max_val,
                                args.batch_size * args.world_size / data_time.avg,
                                reduced_loss.item(),
                                losses.avg,
                            )
                        )
                if j >= expected_iters:
                    break

        print("OK {}".format(pipe_factory.__name__))


if __name__ == "__main__":
    args = parse_test_arguments(True)
    training_test(args)
