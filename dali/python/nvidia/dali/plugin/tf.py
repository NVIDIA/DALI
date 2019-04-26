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

import os
import glob
from collections import Iterable

import functools

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.util.tf_export import tf_export

_tf_plugins = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'libdali_tf*.so'))
_dali_tf_module = None
# Order: 'current', prebuilt versions
_dali_tf_current = list(filter(lambda x: 'current' in x, _tf_plugins))
_dali_tf_prebuilt = list(filter(lambda x: 'current' not in x, _tf_plugins))
_processed_tf_plugins = _dali_tf_current + _dali_tf_prebuilt

for _libdali_tf in _processed_tf_plugins:
  try:
    _dali_tf_module = tf.load_op_library(_libdali_tf)
    break
  # if plugin is not compatible skip it
  except tf.errors.NotFoundError:
    pass
else:
  raise Exception('No matching DALI plugin found for installed TensorFlow version')

_dali_tf = _dali_tf_module.dali


def _PrepareDALIArgsHelper(pipeline = None, serialized_pipeline = None, sparse = [],
                     shapes = [], dtypes = [], batch_size = -1, prefetch_queue_depth = 2, **kwargs):
  """
  Internal helper that converts user input arguments into TensorFlow operator arguments.
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
  prepared_args = {'serialized_pipeline':serialized_pipeline , 'shapes':new_shapes, 'sparse':sparse, 'dtypes':dtypes, 'batch_size':batch_size,
                   'exec_separated':exec_separated, 'gpu_prefetch_queue_depth':gpu_prefetch_queue_depth, 'cpu_prefetch_queue_depth':cpu_prefetch_queue_depth}
  prepared_args.update(**kwargs)
  return prepared_args

def DALIIteratorWrapper(pipeline = None, serialized_pipeline = None, sparse = [],
                        shapes = [], dtypes = [], batch_size = -1, prefetch_queue_depth = 2, **kwargs):
  """
TF Plugin Wrapper

This operator works in the same way as DALI TensorFlow plugin, with the exception that is also accepts Pipeline objects as the input and serializes it internally. For more information, please look **TensorFlow Plugin API reference** in the documentation.
  """
  oldargs = locals().copy()
  oldargs.pop('kwargs', None)
  prepared_dali_args = _PrepareDALIArgsHelper(**oldargs)
  prepared_dali_args.update(**kwargs)

  # gpu_prefetch_queue_depth correspond to the global queue depth in the uniform case
  out = _dali_tf(**prepared_dali_args)
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


class DALIDatasetV2(dataset_ops.DatasetSource):
    def __init__(self,
                  pipeline=None,
                  batch_size=1,
                  sparse=[],
                  shapes=None,
                  dtypes=None,
                  prefetch_queue_depth=2,
                  num_threads=1,
                  **kwargs):
        """Creates a `DALIDataset`.
        Args:
        pipeline: A`nvidia.dali.Pipeline` defining the augmentation to be performed
        batch_size: `int` defining the number of samples in a batch
        shapes: A `List` of `tf.TensorShape` with the expected output shapes
        dtypes: A `List` of `tf.DType` with the expected output types
        devices: A `List` with the indexes of the devices to use
        prefetch_queue_depth: `int` with the amount of prefetched batches
        num_threads: `int` with the number of reader threads in the pipeline per GPU
        """
        oldargs = locals().copy()
        oldargs.pop('self', None)
        oldargs.pop('kwargs', None)
        oldargs.pop('__class__', None)
        self._dali_args = _PrepareDALIArgsHelper(**oldargs)
        self._dali_args.update(**kwargs)

        self._dtypes = self._dali_args['dtypes']
        self._shapes = self._dali_args['shapes']

        types = (self._dtypes[0], self._dtypes[1])
        shapes = (self._shapes[0], self._shapes[1])
        output_classes = (ops.Tensor, ops.Tensor)
        self._structure = structure.convert_legacy_structure(types, shapes, output_classes)

        super(DALIDatasetV2, self).__init__()
        # Change to this when TF updates
        #variant_tensor = self._as_variant_tensor
        #super(DALIDatasetV2, self).__init__(variant_tensor)

    @property
    def _element_structure(self):
        return self._structure

    def _as_variant_tensor(self):
        dali_args = self._dali_args
        return _dali_tf_module.dali_dataset(**dali_args)

class DALIDatasetV1(dataset_ops.DatasetV1Adapter):
  """Creates a `DALIDataset`.
  Args:
  pipeline: A`nvidia.dali.Pipeline` defining the augmentation to be performed
  batch_size: `int` defining the number of samples in a batch
  shapes: A `List` of `tf.TensorShape` with the expected output shapes
  dtypes: A `List` of `tf.DType` with the expected output types
  devices: A `List` with the indexes of the devices to use
  prefetch_queue_depth: `int` with the amount of prefetched batches
  num_threads: `int` with the number of reader threads in the pipeline per GPU
  """

  @functools.wraps(DALIDatasetV2.__init__)
  def __init__(self, **kwargs):
    wrapped = DALIDatasetV2(**kwargs)
    super(DALIDatasetV1, self).__init__(wrapped)

DALIDataset = DALIDatasetV1
#TODO(spanev) Replace when V2 is ready.

DALIIterator.__doc__ = DALIIteratorWrapper.__doc__
DALIRawIterator.__doc__ = _dali_tf.__doc__
