# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from nvidia.dali import pipeline_def
import numpy as np
from test_utils import compare_pipelines
from test_utils import RandomDataIterator


def bricon_ref(input, brightness, brightness_shift, contrast, contrast_center):
  return brightness_shift + brightness * (contrast_center + contrast * (input - contrast_center))

def ref_operator(contrast_center):
  return lambda input, brightness, brightness_shift, contrast: \
                bricon_ref(input, brightness, brightness_shift, contrast, contrast_center)

def params():
  return fn.random.uniform(range=[0.5, 2.0], seed=123), \
         fn.random.uniform(range=[0.5, 2.0], seed=123), \
         fn.random.uniform(range=[0.4, 0.6], seed=123)


@pipeline_def(num_threads=12, device_id=0, seed=1234)
def bri_and_con_pipe(data_iterator, contrast_center, dev='cpu'):
  contrast, brightness, brightness_shift = params()
  inp = fn.external_source(source=data_iterator)
  if dev == 'gpu':
    inp = inp.gpu()
  inp = fn.contrast(inp, contrast=contrast, contrast_center=contrast_center)
  inp = fn.brightness(inp, brightness=brightness, brightness_shift=brightness_shift)
  return inp

@pipeline_def(num_threads=12, device_id=0, seed=1234)
def bricon_pipe(data_iterator, contrast_center, dev='cpu'):
  contrast, brightness, brightness_shift = params()
  inp = fn.external_source(source=data_iterator)
  if dev == 'gpu':
    inp = inp.gpu()
  inp = fn.brightness_contrast(inp, brightness=brightness, brightness_shift=brightness_shift,
                                    contrast=contrast, contrast_center=contrast_center)
  return inp

@pipeline_def(num_threads=12, device_id=0, seed=1234, exec_pipelined=False, exec_async=False)
def bricon_ref_pipe(data_iterator, contrast_center, dev='cpu'):
  contrast, brightness, brightness_shift = params()
  inp = fn.external_source(source=data_iterator)
  inp = fn.python_function(inp, brightness, brightness_shift, contrast,\
                           function=ref_operator(contrast_center))
  return inp


def test_equivalence():
  batch_size=32
  n_iters = 16
  ri1 = RandomDataIterator(batch_size, shape=(512, 256, 3), dtype=np.float32)
  ri2 = RandomDataIterator(batch_size, shape=(512, 256, 3), dtype=np.float32)
  for device in ['cpu', 'gpu']:
    pipe1 = bri_and_con_pipe(ri1, 0.4, device, batch_size=batch_size)
    pipe2 = bricon_pipe(ri2, 0.4, device, batch_size=batch_size)
    yield compare_pipelines, pipe1, pipe2, batch_size, n_iters

def test_vs_ref():
  batch_size=32
  n_iters = 16
  ri1 = RandomDataIterator(batch_size, shape=(512, 256, 3), dtype=np.float32)
  ri2 = RandomDataIterator(batch_size, shape=(512, 256, 3), dtype=np.float32)
  for device in ['cpu', 'gpu']:
    pipe1 = bricon_ref_pipe(ri1, 0.4, device, batch_size=batch_size)
    pipe2 = bricon_pipe(ri2, 0.4, device, batch_size=batch_size)
    yield compare_pipelines, pipe1, pipe2, batch_size, n_iters
