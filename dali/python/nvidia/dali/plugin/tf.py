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

def DALIIteratorWrapper(pipeline = None, serialized_pipeline = None, **kwargs):
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

dali_dataset_module = _dali_tf_module.dali_dataset

class DALIDataset(tf.data.Dataset):

  def __init__(self, image_type=None, label_type=None):
    super(DALIDataset, self).__init__()

    if image_type:
      self._image_type = image_type
    else:
      raise ValueError('No value provided for parameter \'image_type\'')

    if label_type:
      self._label_type = label_type
    else:
      raise ValueError('No value provided for parameter \'label_type\'')

  def _as_variant_tensor(self):
    return dali_dataset_module.dali_dataset()

  @property
  def output_types(self):
    return self._image_type, self._label_type

  @property
  def output_shapes(self):
    raise NotImplementedError

  @property
  def output_classes(self):
    raise NotImplementedError


DALIDataset.__doc__ = DALIDataset.__doc__
DALIIterator.__doc__ = DALIIteratorWrapper.__doc__
DALIRawIterator.__doc__ = _dali_tf.__doc__
