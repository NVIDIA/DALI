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
import nvidia.dali.types as types
import numpy as np
from test_utils import compare_pipelines
from test_utils import RandomDataIterator

def max_range(dtype):
  if dtype == np.half or dtype == np.single or dtype == np.double:
    return 1.0
  else:
    return np.iinfo(dtype).max

def type_range(dtype):
  if dtype in [np.half, np.single, np.double]:
    return (np.finfo(dtype).min, np.finfo(dtype).max)
  else:
    return (np.iinfo(dtype).min, np.iinfo(dtype).max)

def convert_sat(data, out_dtype):
  clipped = np.clip(data, *type_range(out_dtype))
  if out_dtype not in [np.half, np.single, np.double]:
    clipped = np.round(clipped)
  return clipped.astype(out_dtype)

def dali_to_np_type(dtype):
  if dtype == types.FLOAT:
    return np.single
  elif dtype == types.INT16:
    return np.short
  elif dtype == types.UINT8:
    return np.ubyte
  else:
    assert False

def bricon_ref(input, brightness, brightness_shift, contrast, contrast_center, out_dtype):
  output_range = max_range(out_dtype)
  input_range = max_range(input.dtype)
  # print('dtype: ', input.dtype, ' range ', input_range)
  norm = output_range / input_range
  output = brightness_shift * output_range + norm * brightness * (contrast_center + contrast * (input - contrast_center))
  return convert_sat(output, out_dtype)

def ref_operator(contrast_center, out_dtype):
  return lambda input, brightness, brightness_shift, contrast: \
                bricon_ref(input, brightness, brightness_shift, contrast, contrast_center, out_dtype)

def params():
  return fn.random.uniform(range=[0.0, 5.0], seed=123), \
         fn.random.uniform(range=[0.0, 5.0], seed=123), \
         fn.random.uniform(range=[-1.0, 1.0], seed=123)


@pipeline_def(num_threads=4, device_id=0, seed=1234)
def bri_and_con_pipe(data_iterator, contrast_center, dtype, dev='cpu'):
  contrast, brightness, brightness_shift = params()
  inp = fn.external_source(source=data_iterator)
  if dev == 'gpu':
    inp = inp.gpu()
  inp = fn.contrast(inp, contrast=contrast, contrast_center=contrast_center)
  inp = fn.brightness(inp, brightness=brightness, brightness_shift=brightness_shift, dtype=dtype)
  return inp

@pipeline_def(num_threads=4, device_id=0, seed=1234)
def bricon_pipe(data_iterator, contrast_center, dtype, dev='cpu'):
  contrast, brightness, brightness_shift = params()
  inp = fn.external_source(source=data_iterator)
  if dev == 'gpu':
    inp = inp.gpu()
  inp = fn.brightness_contrast(inp, brightness=brightness, brightness_shift=brightness_shift,
                               contrast=contrast, contrast_center=contrast_center, dtype=dtype)
  return inp

@pipeline_def(num_threads=4, device_id=0, seed=1234, exec_pipelined=False, exec_async=False)
def bricon_ref_pipe(data_iterator, contrast_center, dtype, dev='cpu'):
  contrast, brightness, brightness_shift = params()
  inp = fn.external_source(source=data_iterator)
  inp = fn.python_function(inp, brightness, brightness_shift, contrast,\
                           function=ref_operator(contrast_center, dali_to_np_type(dtype)))
  return inp

def check_equivalence(device, dtype):
  batch_size=32
  n_iters = 16
  ri1 = RandomDataIterator(batch_size, shape=(1, 1, 1), dtype=dali_to_np_type(dtype))
  ri2 = RandomDataIterator(batch_size, shape=(1, 1, 1), dtype=dali_to_np_type(dtype))
  contrast_center = 0.4 * max_range(dali_to_np_type(dtype))
  pipe1 = bri_and_con_pipe(ri1, contrast_center, dtype, device, batch_size=batch_size)
  pipe2 = bricon_pipe(ri2, contrast_center, dtype, device, batch_size=batch_size)
  if dtype in [np.half, np.single, np.double]:
    eps = 1e-4
  else:
    eps = 1
  compare_pipelines(pipe1, pipe2, batch_size, n_iters, eps=eps)

def test_equivalence():
  for device in ['cpu', 'gpu']:
    for dtype in [types.FLOAT, types.INT16, types.UINT8]:
      yield check_equivalence, device, dtype


def check_vs_ref(device, inp_dtype, out_dtype):
  batch_size=32
  n_iters = 16
  ri1 = RandomDataIterator(batch_size, shape=(1, 1, 1), dtype=dali_to_np_type(inp_dtype))
  ri2 = RandomDataIterator(batch_size, shape=(1, 1, 1), dtype=dali_to_np_type(inp_dtype))
  contrast_center = 0.4 * max_range(dali_to_np_type(inp_dtype))
  pipe1 = bricon_ref_pipe(ri1, contrast_center, out_dtype, device, batch_size=batch_size)
  pipe2 = bricon_pipe(ri2, contrast_center, out_dtype, device, batch_size=batch_size)
  if out_dtype in [np.half, np.single, np.double]:
    eps = 1e-4
  else:
    eps = 1
  compare_pipelines(pipe1, pipe2, batch_size, n_iters, eps=eps)



def test_vs_ref():
  for device in ['cpu', 'gpu']:
    for inp_dtype in [types.FLOAT, types.INT16, types.UINT8]:
      for out_dtype in [types.FLOAT, types.INT16, types.UINT8]:
        yield check_vs_ref, device, inp_dtype, out_dtype
