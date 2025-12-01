# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import backend as _b
from nvidia.dali import ops

_internal_schemas = ["_TFRecordReader", "readers___TFRecord"]


def tfrecord_enabled():
    """Check if the TFRecord Reader op is enabled by looking up if the internal implementation
    was registered in the backend.
    This call is valid after the backend ops were discovered (_load_ops() was called).
    """
    for internal_schema in _internal_schemas:
        if _b.TryGetSchema(internal_schema) is not None:
            return True
    return False


if tfrecord_enabled():

    def _get_impl(name, schema_name, internal_schema_name):

        class _TFRecordReaderImpl(
            ops.python_op_factory(name, schema_name, internal_schema_name, generated=False)
        ):
            """custom wrappers around ops"""

            def __init__(self, path, index_path, features, **kwargs):
                if isinstance(path, list):
                    self._path = path
                else:
                    self._path = [path]
                if isinstance(index_path, list):
                    self._index_path = index_path
                else:
                    self._index_path = [index_path]

                kwargs.update({"path": self._path, "index_path": self._index_path})
                self._features = features

                super().__init__(**kwargs)

            def __call__(self, *inputs, **kwargs):
                feature_names = []
                features = []
                for feature_name, feature in self._features.items():
                    feature_names.append(feature_name)
                    features.append(feature)
                    if not isinstance(feature, _b.tfrecord.Feature):
                        raise TypeError(
                            "Expected `nvidia.dali.tfrecord.Feature` for the "
                            f'"{feature_name}", but got {type(feature)}. '
                            "Use `nvidia.dali.tfrecord.FixedLenFeature` or "
                            "`nvidia.dali.tfrecord.VarLenFeature` "
                            "to define the features to extract."
                        )

                kwargs.update({"feature_names": feature_names, "features": features})

                # We won't have MIS as this op doesn't have any inputs (Reader)
                linear_outputs = super().__call__(*inputs, **kwargs)
                # We may have single, flattened output
                if not isinstance(linear_outputs, list):
                    linear_outputs = [linear_outputs]
                outputs = {}
                for feature_name, output in zip(feature_names, linear_outputs):
                    outputs[feature_name] = output

                return outputs

        return _TFRecordReaderImpl

    class TFRecordReader(_get_impl("_TFRecordReader", "TFRecordReader", "_TFRecordReader")):
        pass

    class TFRecord(_get_impl("_TFRecord", "readers__TFRecord", "readers___TFRecord")):
        pass
