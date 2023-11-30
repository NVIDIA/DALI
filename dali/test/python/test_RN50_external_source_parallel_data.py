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

from test_utils import AverageMeter
import time

from test_RN50_external_source_parallel_utils import (
    parse_test_arguments,
    external_source_parallel_pipeline,
    external_source_pipeline,
    file_reader_pipeline,
    get_pipe_factories,
)


# This test requires significant amount of shared memory to be able to pass
# the batches between worker processes and the main process. If running in docker
# make sure that -shm-size is big enough.


def iteration_test(args):
    test_pipe_factories = get_pipe_factories(
        args.test_pipes,
        external_source_parallel_pipeline,
        file_reader_pipeline,
        external_source_pipeline,
    )
    for pipe_factory in test_pipe_factories:
        # TODO(klecki): We don't handle sharding in this test yet, would need to do it manually
        # for External Source pipelines
        pipes = [
            pipe_factory(
                batch_size=args.batch_size,
                num_threads=args.workers,
                device_id=gpu,
                data_path=args.data_path,
                prefetch_queue_depth=args.prefetch,
                reader_queue_depth=args.reader_queue_depth,
                py_start_method=args.worker_init,
                py_num_workers=args.py_workers,
                source_mode=args.source_mode,
                read_encoded=args.dali_decode,
            )
            for gpu in range(args.gpus)
        ]
        # First start the Python workers, so we fork without CUDA context.
        for pipe in pipes:
            pipe.start_py_workers()
        for pipe in pipes:
            pipe.build()

        samples_no = pipes[0].epoch_size("Reader")
        if args.benchmark_iters is None:
            expected_iters = samples_no // args.batch_size + (samples_no % args.batch_size != 0)
        else:
            expected_iters = args.benchmark_iters

        print("RUN {}".format(pipe_factory.__name__))
        for i in range(args.epochs):
            if i == 0:
                print("Warm up")
            else:
                print("Test run " + str(i))
            data_time = AverageMeter()
            end = time.time()
            frequency = 50
            for j in range(expected_iters):
                stop_iter = False
                for pipe in pipes:
                    try:
                        pipe.run()
                    except StopIteration:
                        assert j == expected_iters - 1
                        stop_iter = True
                if stop_iter:
                    break
                if j % frequency == 0 and j != 0:
                    data_time.update((time.time() - end) / frequency)
                    end = time.time()
                    print(
                        f"{pipe_factory.__name__} {j}/ {expected_iters}, "
                        f"avg time: {data_time.avg} [s], worst time: {data_time.max_val} [s], "
                        f"speed: {args.batch_size * args.gpus / data_time.avg} [img/s]"
                    )
            for pipe in pipes:
                pipe.reset()

        print("OK {}".format(pipe_factory.__name__))


if __name__ == "__main__":
    args = parse_test_arguments(False)
    iteration_test(args)
