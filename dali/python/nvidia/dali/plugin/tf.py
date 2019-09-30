# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf
import os
import glob
from collections import Iterable
import re

import functools

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.framework import dtypes

from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import tensor_spec

_tf_plugins = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'libdali_tf*.so'))
_dali_tf_module = None
# Order: 'current', prebuilt for current TF version, prebuilt for other TF versions
_tf_version = re.search("(\d+.\d+).\d+", tf.__version__).group(1)
_tf_version_underscore = _tf_version.replace('.', '_')
_dali_tf_current = list(filter(lambda x: 'current' in x, _tf_plugins))
_dali_tf_prebuilt_tf_ver = list(filter(lambda x: _tf_version_underscore in x, _tf_plugins))
_dali_tf_prebuilt_others = list(filter(lambda x: 'current' not in x and _tf_version_underscore not in x, _tf_plugins))
_processed_tf_plugins = _dali_tf_current + _dali_tf_prebuilt_tf_ver + _dali_tf_prebuilt_others

first_error = None
for _libdali_tf in _processed_tf_plugins:
  try:
    _dali_tf_module = tf.load_op_library(_libdali_tf)
    break
  # if plugin is not compatible skip it
  except tf.errors.NotFoundError as error:
    if first_error == None:
      first_error = error
else:
  raise first_error or Exception('No matching DALI plugin found for installed TensorFlow version')

_dali_tf = _dali_tf_module.dali

def DALIIteratorWrapper(pipeline = None, serialized_pipeline = None, sparse = [],
                        shapes = [], dtypes = [], batch_size = -1, prefetch_queue_depth = 2, **kwargs):
  """
TF Plugin Wrapper

This operator works in the same way as DALI TensorFlow plugin, with the exception that is also accepts Pipeline objects as the input and serializes it internally. For more information, please look **TensorFlow Plugin API reference** in the documentation.
  """
  if type(prefetch_queue_depth) is dict:
      exec_separated = True
      cpu_prefetch_queue_depth = prefetch_queue_depth["cpu_size"]
      gpu_prefetch_queue_depth = prefetch_queue_depth["gpu_size"]
  elif type(prefetch_queue_depth) is int:
      exec_separated = False
      cpu_prefetch_queue_depth = -1 # dummy: wont' be used
      gpu_prefetch_queue_depth = prefetch_queue_depth

  if serialized_pipeline is None:
    serialized_pipeline = pipeline.serialize()

  # if batch_size is not provided we need to extract if from the shape arg
  if (not isinstance(shapes, Iterable) or len(shapes) == 0) and batch_size == -1:
    raise Exception('shapes and batch_size arguments cannot be empty, '
                    'please provide at leas one shape argument element with the BATCH size or set batch_size')

  if len(sparse) > 0 and sparse[0] and batch_size == -1:
    if isinstance(shapes[0], Iterable) and len(shapes[0]) == 1:
      shapes[0] = (shapes[0][0], 1)
    else:
      shapes[0] = (shapes[0], 1)

  # shapes and dtypes need to take into account that sparse tensor will produce 3 output tensors
  new_dtypes = []
  new_shapes = []
  for i in range(len(dtypes)):
    if i < len(sparse) and sparse[i]:
      # indices type of sparse tensor is tf.int64
      new_dtypes.append(tf.int64)
      new_dtypes.append(dtypes[i])
      # dense shape type of sparse tensor is tf.int64
      new_dtypes.append(tf.int64)
      if len(shapes) > i and len(shapes[i]) > 0:
        new_shapes.append((shapes[i][0], 1))
        new_shapes.append((shapes[i][0]))
      else:
        new_shapes.append(())
        new_shapes.append(())
      new_shapes.append(())
    else:
      new_dtypes.append(dtypes[i])
      if len(shapes) > i:
        new_shapes.append(shapes[i])

  # gpu_prefetch_queue_depth correspond to the global queue depth in the uniform case
  out = _dali_tf(serialized_pipeline=serialized_pipeline, shapes=new_shapes, dtypes=new_dtypes, sparse=sparse, batch_size=batch_size,
                 exec_separated=exec_separated, gpu_prefetch_queue_depth=gpu_prefetch_queue_depth, cpu_prefetch_queue_depth=cpu_prefetch_queue_depth, **kwargs)
  new_out = []
  j = 0
  for i in range(len(dtypes)):
    if i < len(sparse) and sparse[i]:
      new_out.append(tf.SparseTensor(indices=out[j], values=out[j + 1], dense_shape=out[j + 2]))
      j += 3
    else:
      new_out.append(out[j])
      j += 1
  return new_out

def DALIIterator():
    return DALIIteratorWrapper

# Vanilla raw operator legacy
def DALIRawIterator():
    return _dali_tf


def get_tf_minor_version():
  return tf.__version__.split('.')[1]


class DALIDatasetV2(dataset_ops.DatasetSource):
  """Creates a `DALIDataset` compatible with tf.data.Dataset from a DALI pipeline.

  Parameters
  ----------
  `pipeline` : `nvidia.dali.Pipeline` defining the augmentations to be performed. 
  `batch_size` : int,
      Batch size of the pipeline.
  `num_threads` : int,
      Number of CPU threads used by the pipeline.
  `device_id` : int,
      Id of GPU used by the pipeline.
  `exec_separated` : bool,
      Whether to execute the pipeline in a way that enables
      overlapping CPU and GPU computation, typically resulting
      in faster execution speed, but larger memory consumption.
  `prefetch_queue_depth` : int,
      depth of the executor queue. Deeper queue makes DALI more 
      resistant to uneven execution  time of each batch, but it also 
      consumes more memory for internal buffers.
      Value will be used with `exec_separated` set to False.
  `cpu_prefetch_queue_depth` : int,
      depth of the executor cpu queue. Deeper queue makes DALI more 
      resistant to uneven execution  time of each batch, but it also 
      consumes more memory for internal buffers.
      Value will be used with `exec_separated` set to True.
  `gpu_prefetch_queue_depth` : int,
      depth of the executor gpu queue. Deeper queue makes DALI more 
      resistant to uneven execution  time of each batch, but it also 
      consumes more memory for internal buffers.
      Value will be used with `exec_separated` set to True.
  `shapes`: `List` of tuples with the expected output shapes
  `dtypes`: `List` of `tf.DType` with the expected output types
  """
  def __init__(
    self,
    pipeline = '',
    batch_size = 1,
    num_threads = 4,
    device_id = 0,
    exec_separated = False,
    prefetch_queue_depth = 2,
    cpu_prefetch_queue_depth = 2,
    gpu_prefetch_queue_depth = 2,
    shapes = [], 
    dtypes = []):

    assert(len(shapes) == len(dtypes),
      "Different number of provided shapes and dtypes.")

    if exec_separated:
      assert(cpu_prefetch_queue_depth is not None,
        "With exec_separated == True cpu_prefetch_queue_depth cannot be None")
      assert(gpu_prefetch_queue_depth is not None,
        "With exec_separated == True gpu_prefetch_queue_depth cannot be None")
    else:
      assert(prefetch_queue_depth is not None,
        "With exec_separated == False prefetch_queue_depth cannot be None")

    output_classes = tuple(ops.Tensor for shape in shapes)

    self._pipeline = pipeline.serialize()
    self._batch_size = batch_size
    self._num_threads = num_threads
    self._device_id = device_id
    self._exec_separated = exec_separated
    self._prefetch_queue_depth = prefetch_queue_depth
    self._cpu_prefetch_queue_depth = cpu_prefetch_queue_depth
    self._gpu_prefetch_queue_depth = gpu_prefetch_queue_depth
    self._shapes = tuple(tf.TensorShape(shape) for shape in shapes)
    self._dtypes = tuple(dtype for dtype in dtypes)

    self._structure = structure.convert_legacy_structure(
      self._dtypes, self._shapes, output_classes)

    if get_tf_minor_version() == '14':
      super(DALIDatasetV2, self).__init__(self._as_variant_tensor())
    elif get_tf_minor_version() == '13':
      super(DALIDatasetV2, self).__init__()
    else:
      raise RuntimeError('Unsupported TensorFlow version detected at runtime. DALIDataset supports versions: 1.13, 1.14')


  @property
  def _element_structure(self):
    return self._structure


  # This function should not be removed or refactored.
  # It is needed for TF 1.13.1
  def _as_variant_tensor(self):
    return _dali_tf_module.dali_dataset(
      pipeline = self._pipeline,
      batch_size = self._batch_size,
      num_threads = self._num_threads,
      device_id = self._device_id,
      exec_separated = self._exec_separated,
      prefetch_queue_depth = self._prefetch_queue_depth,
      cpu_prefetch_queue_depth = self._cpu_prefetch_queue_depth,
      gpu_prefetch_queue_depth = self._gpu_prefetch_queue_depth,
      shapes = self._shapes, 
      dtypes = self._dtypes)


class DALIDatasetV1(dataset_ops.DatasetV1Adapter):
  """Creates a `DALIDataset` compatible with tf.data.Dataset from a DALI pipeline.

  Parameters
  ----------
  `pipeline` : `nvidia.dali.Pipeline` defining the augmentations to be performed. 
  `batch_size` : int,
      Batch size of the pipeline.
  `num_threads` : int,
      Number of CPU threads used by the pipeline.
  `device_id` : int,
      Id of GPU used by the pipeline.
  `exec_separated` : bool,
      Whether to execute the pipeline in a way that enables
      overlapping CPU and GPU computation, typically resulting
      in faster execution speed, but larger memory consumption.
  `prefetch_queue_depth` : int,
      depth of the executor queue. Deeper queue makes DALI more 
      resistant to uneven execution  time of each batch, but it also 
      consumes more memory for internal buffers.
      Value will be used with `exec_separated` set to False.
  `cpu_prefetch_queue_depth` : int,
      depth of the executor cpu queue. Deeper queue makes DALI more 
      resistant to uneven execution  time of each batch, but it also 
      consumes more memory for internal buffers.
      Value will be used with `exec_separated` set to True.
  `gpu_prefetch_queue_depth` : int,
      depth of the executor gpu queue. Deeper queue makes DALI more 
      resistant to uneven execution  time of each batch, but it also 
      consumes more memory for internal buffers.
      Value will be used with `exec_separated` set to True.
  `shapes`: `List` of tuples with the expected output shapes
  `dtypes`: `List` of `tf.DType` with the expected output types
  """


  @functools.wraps(DALIDatasetV2.__init__)
  def __init__(self, **kwargs):
    wrapped = DALIDatasetV2(**kwargs)
    super(DALIDatasetV1, self).__init__(wrapped)


# This is for TensorFlow 1.x compatibility
DALIDataset = DALIDatasetV1


DALIIterator.__doc__ = DALIIteratorWrapper.__doc__
DALIRawIterator.__doc__ = _dali_tf.__doc__
