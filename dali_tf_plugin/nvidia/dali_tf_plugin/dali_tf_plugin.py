# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import re

_dali_tf = None

def load_dali_tf_plugin():
    global _dali_tf
    if _dali_tf is not None:
        return _dali_tf

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
    _dali_tf.__doc__ = _dali_tf.__doc__ + """

    Please keep in mind that TensorFlow allocates almost all available device memory by default. This might cause errors in
    DALI due to insufficient memory. On how to change this behaviour please look into the TensorFlow documentation, as it may
    differ based on your use case.
"""
    return _dali_tf
