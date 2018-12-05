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

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

_tf_plugins = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'libdali_tf*.so'))
_dali_tf_module = None
for _libdali_tf in _tf_plugins:
    try:
        _dali_tf_module = tf.load_op_library(_libdali_tf)
        break
    # if plugin is not compatible skip it
    except tf.errors.NotFoundError:
        pass
else:
    raise Exception('No matching DALI plugin found for installed TensorFlow version')

_dali_tf = _dali_tf_module.dali


def DALIIteratorWrapper(pipeline=None, serialized_pipeline=None, **kwargs):
    """
  TF Plugin Wrapper

  This operator works in the same way as DALI TensorFlow plugin, with the exception that is also accepts Pipeline objects as the input and serializes it internally. For more information, please look **TensorFlow Plugin API reference** in the documentation.
    """
    if serialized_pipeline is None:
        serialized_pipeline = pipeline.serialize()
    return _dali_tf(serialized_pipeline=serialized_pipeline, **kwargs)


def DALIIterator():
    return DALIIteratorWrapper


# Vanilla raw operator legacy
def DALIRawIterator():
    return _dali_tf


_dali_dataset = _dali_tf_module.dali_dataset


class DALIDataset(tf.data.Dataset):
    """A `Dataset` wrapping DALI iterators spread across a number of devices."""

    def __init__(self,
                 pipeline=None,
                 shapes=None,
                 dtypes=None,
                 devices=None,
                 prefetch_queue_depth=2,
                 num_threads=-1):
        """Creates a `DALIDataset`.
       Args:
         pipeline: A`nvidia.dali.Pipeline` defining the augmentation to be performed
         shapes: A `List` of `tf.TensorShape` with the expected output shapes
         dtypes: A `List` of `tf.DType` with the expected output types

       """
        super(DALIDataset, self).__init__()

        if pipeline:
            self._pipeline = pipeline.serialize()
        else:
            raise ValueError('No value provided for parameter \'pipeline\'')

        if shapes:
            self._shapes = shapes
        else:
            raise ValueError('No value provided for parameter \'shapes\'')

        if dtypes:
            self._dtypes = dtypes
        else:
            raise ValueError('No value provided for parameter \'dtypes\'')

        if devices:
            self._devices = devices
        else:
            raise ValueError('No value provided for parameter \'devices\'')

        self._prefetch_queue_threads = prefetch_queue_depth
        self._num_threads = num_threads

    def _as_variant_tensor(self):
        return _dali_dataset(serialized_pipeline=self._pipeline,
                             shapes=self._shapes,
                             dtypes=self._dtypes,
                             devices=self._devices,
                             prefetch_queue_depth=self._prefetch_queue_threads,
                             num_threads=self._num_threads)

    @property
    def output_types(self):
        return self._dtypes[0], self._dtypes[1]

    @property
    def output_shapes(self):
        # TODO: Tenporary shapes to be generalized
        return tensor_shape.TensorShape(self._shapes[0]),\
               tensor_shape.TensorShape(self._shapes[1])

    @property
    def output_classes(self):
        return ops.Tensor, ops.Tensor


DALIDataset.__doc__ = DALIDataset.__doc__
DALIIterator.__doc__ = DALIIteratorWrapper.__doc__
DALIRawIterator.__doc__ = _dali_tf.__doc__
