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
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape
from nvidia.dali import types

from collections import Iterable
from distutils.version import LooseVersion
import warnings

from nvidia.dali_tf_plugin import dali_tf_plugin
_dali_tf_module = dali_tf_plugin.load_dali_tf_plugin()
_dali_tf = _dali_tf_module.dali
_dali_tf.__doc__ = _dali_tf.__doc__ + """

    Please keep in mind that TensorFlow allocates almost all available device memory by default. This might cause errors in
    DALI due to insufficient memory. On how to change this behaviour please look into the TensorFlow documentation, as it may
    differ based on your use case.
"""

def serialize_pipeline(pipeline):
  try:
    return pipeline.serialize()
  except RuntimeError as e:
    raise RuntimeError("Error during pipeline initialization. Note that some operators "
                       "(e.g. Python Operators) cannot be used with "
                       "tensorflow data set API and DALIIterator.") from e


def DALIIteratorWrapper(pipeline = None, serialized_pipeline = None, sparse = [],
                        shapes = [], dtypes = [], batch_size = -1, prefetch_queue_depth = 2, **kwargs):
  """
  TF Plugin Wrapper

  This operator works in the same way as DALI TensorFlow plugin, with the exception that it also
  accepts Pipeline objects as an input, which are serialized internally. For more information,
  see :meth:`nvidia.dali.plugin.tf.DALIRawIterator`.
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
    serialized_pipeline = serialize_pipeline(pipeline)

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


def _get_tf_version():
  return LooseVersion(tf.__version__)


MIN_TENSORFLOW_VERSION = LooseVersion('1.15')
def dataset_compatible_tensorflow():
    return LooseVersion(tf.__version__) >= MIN_TENSORFLOW_VERSION
 
def dataset_distributed_compatible_tensorflow():
    return LooseVersion(tf.__version__) >= LooseVersion('2.5.0')


if dataset_compatible_tensorflow():
  from tensorflow.python.framework import ops
  from tensorflow.python.data.ops import dataset_ops
  from tensorflow.python.data.util import structure
  import functools

  def dataset_options():
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.autotune = False

    return options


  class _DALIDatasetV2(dataset_ops.DatasetSource):
    def __init__(
      self,
      pipeline,
      output_dtypes = None,
      output_shapes = None,
      fail_on_device_mismatch = True,
      *,
      batch_size = 1,
      num_threads = 4,
      device_id = 0,
      exec_separated = False,
      prefetch_queue_depth = 2,
      cpu_prefetch_queue_depth = 2,
      gpu_prefetch_queue_depth = 2,
      dtypes=None,
      shapes=None):

      output_shapes = self._handle_deprecation(output_shapes, shapes, "shapes")
      output_dtypes = self._handle_deprecation(output_dtypes, dtypes, "dtypes")

      if not self._check_output_dtypes(output_dtypes):
        raise TypeError(("`output_dtypes` should be provided as single tf.DType value " +
            "or a tuple of tf.DType values. Got value `{}` of type `{}`.") \
                .format(output_dtypes, type(output_dtypes)))

      if output_shapes is None:
        output_shapes = nest.map_structure(lambda _: tensor_shape.TensorShape(None), output_dtypes)
      else:
        output_shapes = nest.map_structure_up_to(output_dtypes, tensor_shape.as_shape, output_shapes)

      if not isinstance(output_dtypes, tuple):
        output_dtypes = (output_dtypes,)
        output_shapes = (output_shapes,)

      output_classes = nest.map_structure(lambda _: ops.Tensor, output_dtypes)

      self._pipeline = serialize_pipeline(pipeline)
      self._batch_size = batch_size
      self._num_threads = num_threads
      if device_id is None:
          device_id = types.CPU_ONLY_DEVICE_ID
      self._device_id = device_id
      self._exec_separated = exec_separated
      self._prefetch_queue_depth = prefetch_queue_depth
      self._cpu_prefetch_queue_depth = cpu_prefetch_queue_depth
      self._gpu_prefetch_queue_depth = gpu_prefetch_queue_depth
      self._output_shapes = output_shapes
      self._output_dtypes = output_dtypes
      self._fail_on_device_mismatch = fail_on_device_mismatch

      self._structure = structure.convert_legacy_structure(
        self._output_dtypes, self._output_shapes, output_classes)

      super(_DALIDatasetV2, self).__init__(self._as_variant_tensor())


    def _check_output_dtypes(self, output_dtypes):
      """Check whether output_dtypes is instance of tf.DType or tuple of tf.DType
      """
      if isinstance(output_dtypes, tf.DType):
        return True
      elif isinstance(output_dtypes, tuple) \
          and all(isinstance(dtype, tf.DType) for dtype in output_dtypes):
        return True
      else:
        return False

    def _handle_deprecation(self, supported_arg, deprecated_arg, name):
      if deprecated_arg is not None:
        if supported_arg is not None:
          raise ValueError(("Usage of `{name}` is deprecated in favor of `output_{name}`. " +
            "Both arguments were provided, but only `output_{name}` should be provided.").format(name=name))
        # show only this warning
        warnings.warn(("Use of argument `{name}` is deprecated. Please use `output_{name}` instead. " \
            + "`output_{name}` should be provided as a tuple or a single value.").format(name=name),
            Warning, stacklevel=2)
        if isinstance(deprecated_arg, list):
          return tuple(deprecated_arg)
        return deprecated_arg
      else:
        return supported_arg

    @property
    def element_spec(self):
      return self._structure


    @property
    def _element_structure(self):
      return self._structure


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
        output_shapes = self._output_shapes,
        output_dtypes = self._output_dtypes,
        fail_on_device_mismatch = self._fail_on_device_mismatch)


  if _get_tf_version() < LooseVersion('2.0'):
    class _DALIDatasetImpl(dataset_ops.DatasetV1Adapter):
      @functools.wraps(_DALIDatasetV2.__init__)
      def __init__(self, pipeline, **kwargs):
        self._wrapped = _DALIDatasetV2(pipeline, **kwargs)
        super(_DALIDatasetImpl, self).__init__(self._wrapped)
  else:
    _DALIDatasetImpl = _DALIDatasetV2

  class DALIDataset(dataset_ops._OptionsDataset):
    @functools.wraps(_DALIDatasetV2.__init__)
    def __init__(self, pipeline, **kwargs):
      dataset_impl = _DALIDatasetImpl(pipeline, **kwargs)
      super(DALIDataset, self).__init__(dataset_impl, dataset_options())

else:
  class DALIDataset:
    def __init__(
      self,
      pipeline,
      output_dtypes = None,
      output_shapes = None,
      fail_on_device_mismatch = True,
      *,
      batch_size = 1,
      num_threads = 4,
      device_id = 0,
      exec_separated = False,
      prefetch_queue_depth = 2,
      cpu_prefetch_queue_depth = 2,
      gpu_prefetch_queue_depth = 2,
      dtypes=None,
      shapes=None):
      raise RuntimeError('DALIDataset is not supported for detected version of TensorFlow.  DALIDataset supports versions: 1.15, 2.0')

DALIDataset.__doc__ =  """Creates a `DALIDataset` compatible with tf.data.Dataset from a DALI pipeline. It supports TensorFlow 1.15 and 2.0


    Please keep in mind that TensorFlow allocates almost all available device memory by default. This might cause errors in
    DALI due to insufficient memory. On how to change this behaviour please look into the TensorFlow documentation, as it may
    differ based on your use case.

    Parameters
    ----------
    `pipeline` : `nvidia.dali.Pipeline`
        defining the data processing to be performed.
    `output_dtypes`: `tf.DType` or `tuple` of `tf.DType`, default = None
        expected output types
    `output_shapes`: tuple of shapes, optional, default = None
        expected output shapes. If provided, must match arity of the `output_dtypes`.
        When set to None, DALI will infer the shapes on its own.
        Individual shapes can be also set to None or contain None to indicate unknown dimensions.
        If specified must be compatible with shape returned from DALI Pipeline
        and with `batch_size` argument which will be the outermost dimension of returned tensors.
        In case of `batch_size = 1` it can be omitted in the shape.
        DALI Dataset will try to match requested shape by squeezing 1-sized dimensions
        from shape obtained from Pipeline.
    `fail_on_device_mismatch` : bool, optional, default = True
        When set to `True` runtime check will be performed to ensure DALI device and TF device are
        both CPU or both GPU. In some contexts this check might be inaccurate. When set to `False`
        will skip the check but print additional logs to check the devices. Keep in mind that this
        may allow hidden GPU to CPU copies in the workflow and impact performance.
    `batch_size` : int, optional, default = 1
        batch size of the pipeline.
    `num_threads` : int, optional, default = 4
        number of CPU threads used by the pipeline.
    `device_id` : int, optional, default = 0
        id of GPU used by the pipeline.
        A None value for this parameter means that DALI should not use GPU nor CUDA runtime.
        This limits the pipeline to only CPU operators but allows it to run on any CPU capable machine.
    `exec_separated` : bool, optional, default = False
        Whether to execute the pipeline in a way that enables
        overlapping CPU and GPU computation, typically resulting
        in faster execution speed, but larger memory consumption.
    `prefetch_queue_depth` : int, optional, default = 2
        depth of the executor queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with `exec_separated` set to False.
    `cpu_prefetch_queue_depth` : int, optional, default = 2
        depth of the executor cpu queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with `exec_separated` set to True.
    `gpu_prefetch_queue_depth` : int, optional, default = 2
        depth of the executor gpu queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with `exec_separated` set to True.

    Returns
    -------
    `DALIDataset` object based on DALI pipeline and compatible with `tf.data.Dataset` API.

    """

DALIIterator.__doc__ = DALIIteratorWrapper.__doc__
DALIRawIterator.__doc__ = _dali_tf.__doc__
