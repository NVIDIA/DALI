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

import nvidia.dali.fn as fn
import os
from nvidia.dali.pipeline import pipeline_def
from test_utils import get_dali_extra_path, compare_pipelines

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')

def dump(pipe):
    print()
    print('-----------------------------------------------------')
    pipe.build()
    res = pipe.run()
    for i in range(len(res[0].shape())):
        print(res[0].at(i))
    # print(res[1].as_array())

# Epoch_size is calculated from reader meta if available.
# Otherwise, it must be passed.
def check_pipeline_checkpointing_native(pipeline_factory, epoch_size=None):
    warmup_epochs = 3
    comparsion_iterations = 16
    args = {
        'batch_size': 10,
        'num_threads': 4,
        'checkpointing': True,
        'device_id': 0,
        'exec_async': True,
        'exec_pipelined': True,
    }

    pipe = pipeline_factory(**args)
    assert pipe._checkpointing
    pipe.build()
    assert pipe._checkpointing

    # Checkpoints can be only accessed between the epochs
    # Because of that, we need to calculate the exact epoch size
    epoch_size = epoch_size if epoch_size is not None \
                            else pipe.reader_meta('Reader')['epoch_size_padded']
    iterations_in_epoch = (epoch_size + args['batch_size'] - 1) // args['batch_size']
    for _ in range(warmup_epochs * iterations_in_epoch):
        pipe.run()

    restored = pipeline_factory(**args, serialized_checkpoint=pipe.checkpoint())
    compare_pipelines(pipe, restored, args['batch_size'], comparsion_iterations)

def test_simple_cpu_pipeline():
    @pipeline_def
    def pipeline_factory():
      data, _ = fn.readers.file(name="Reader", file_root=images_dir, pad_last_batch=True, random_shuffle=True)
      return fn.decoders.image_random_crop(data)
      
    check_pipeline_checkpointing_native(pipeline_factory)

def test_simple_mixed_pipeline():
    @pipeline_def
    def pipeline_factory():
      data, label = fn.readers.file(name="Reader", file_root=images_dir, pad_last_batch=True, random_shuffle=True)
      return fn.decoders.image_random_crop(data, device='mixed')
      
    check_pipeline_checkpointing_native(pipeline_factory)

def test_rng_cpu_pipeline():
    @pipeline_def
    def pipeline_factory():
       random_data = fn.random.coin_flip(shape=[10])
       return fn.reductions.sum(random_data, axes=0)
      
    check_pipeline_checkpointing_native(pipeline_factory, epoch_size=1)
