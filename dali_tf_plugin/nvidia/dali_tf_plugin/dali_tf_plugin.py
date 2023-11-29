# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

_dali_tf_module = None


def load_dali_tf_plugin():
    global _dali_tf_module
    if _dali_tf_module is not None:
        return _dali_tf_module

    import nvidia.dali as dali  # Make sure DALI lib is loaded

    assert dali
    tf_plugins = glob.glob(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "libdali_tf*.so")
    )
    # Order: 'current', prebuilt for current TF version, prebuilt for other TF versions
    tf_version = re.search(r"(\d+.\d+).\d+", tf.__version__).group(1)
    tf_version_underscore = tf_version.replace(".", "_")
    dali_tf_current = list(filter(lambda x: "current" in x, tf_plugins))
    dali_tf_prebuilt_tf_ver = list(filter(lambda x: tf_version_underscore in x, tf_plugins))
    dali_tf_prebuilt_others = list(
        filter(lambda x: "current" not in x and tf_version_underscore not in x, tf_plugins)
    )
    processed_tf_plugins = dali_tf_current + dali_tf_prebuilt_tf_ver + dali_tf_prebuilt_others

    first_error = None

    for libdali_tf in processed_tf_plugins:
        try:
            _dali_tf_module = tf.load_op_library(libdali_tf)
            break
        # if plugin is not compatible skip it
        except tf.errors.NotFoundError as error:
            if first_error is None:
                first_error = error
    else:
        raise first_error or Exception(
            "No matching DALI plugin found for installed TensorFlow version"
        )

    return _dali_tf_module
