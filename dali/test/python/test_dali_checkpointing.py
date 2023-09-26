# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali as dali
import nvidia.dali.fn as fn
import os
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from test_utils import get_dali_extra_path, compare_pipelines, check_batch

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')

warmup_epochs = 3
comparsion_iterations = 16
pipeline_args = {
    'batch_size': 10,
    'num_threads': 4,
    'checkpointing': True,
    'device_id': 0,
    'exec_async': True,
    'exec_pipelined': True,
}


def dump(pipe):
    print()
    print('-----------------------------------------------------')
    pipe.build()
    res = pipe.run()
    for i in range(len(res[0].shape())):
        print(res[0].at(i))
    # print(res[1].as_array())


# Checkpoints can be only accessed between the epochs
# Because of that, we need to calculate the exact epoch size
def calculate_iterations_in_epoch(pipe):
    reader_meta = pipe.reader_meta()
    try:
        epoch_size = reader_meta['Reader']['epoch_size_padded']
    except KeyError:
        # There is no reader in the pipeline
        epoch_size = 1

    # Round up, because pad_last_batch=True
    return (epoch_size + pipeline_args['batch_size'] - 1) // pipeline_args['batch_size']


def check_pipeline_checkpointing_native(pipeline_factory):
    pipe = pipeline_factory(**pipeline_args)
    pipe.build()

    iterations_in_epoch = calculate_iterations_in_epoch(pipe)
    for _ in range(warmup_epochs * iterations_in_epoch):
        pipe.run()

    restored = pipeline_factory(**pipeline_args, checkpoint_to_restore=pipe.checkpoint())
    compare_pipelines(pipe, restored, pipeline_args['batch_size'], comparsion_iterations)


def check_pipeline_checkpointing_iterator(pipeline_factory):
    pipe = pipeline_factory(**pipeline_args)
    pipe.build()

    iter = DALIClassificationIterator(pipe, reader_name='Reader', auto_reset=True)
    for _ in range(warmup_epochs):
        for _ in iter:
            pass

    restored = pipeline_factory(**pipeline_args, checkpoint_to_restore=iter.checkpoints()[0])
    restored.build()
    iter2 = DALIClassificationIterator(restored , reader_name='Reader', auto_reset=True)

    for out1, out2 in zip(iter, iter2):
        for d1, d2 in zip(out1, out2):
            for key in d1.keys():
                assert (d1[key] == d2[key]).all()


def test_simple_cpu_pipeline():
    @pipeline_def
    def pipeline_factory():
      data, label = fn.readers.file(name="Reader", file_root=images_dir, pad_last_batch=True, random_shuffle=True)
      decoded = fn.decoders.image_random_crop(data)
      return fn.resize(decoded, resize_x=120, resize_y=90), label

    check_pipeline_checkpointing_native(pipeline_factory)
    check_pipeline_checkpointing_iterator(pipeline_factory)

def test_simple_mixed_pipeline():
    @pipeline_def
    def pipeline_factory():
      data, label = fn.readers.file(name="Reader", file_root=images_dir, pad_last_batch=True, random_shuffle=True)
      decoded = fn.decoders.image_random_crop(data, device='mixed')
      return fn.resize(decoded, resize_x=120, resize_y=90), label

    check_pipeline_checkpointing_native(pipeline_factory)
    check_pipeline_checkpointing_iterator(pipeline_factory)

def test_rng_cpu_pipeline():
    @pipeline_def
    def pipeline_factory():
       random_data = fn.random.coin_flip(shape=[10])
       return fn.reductions.sum(random_data, axes=0)

    check_pipeline_checkpointing_native(pipeline_factory)
