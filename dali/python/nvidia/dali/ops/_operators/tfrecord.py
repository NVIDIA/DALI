# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class _TFRecordReaderImpl:
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
        self._schema = _b.GetSchema(self._internal_schema_name)
        self._spec = _b.OpSpec(self._internal_schema_name)
        self._device = "cpu"

        self._init_args, self._call_args = ops._separate_kwargs(kwargs)
        self._init_args.update({"path": self._path, "index_path": self._index_path})
        self._name = self._init_args.pop("name", None)
        self._preserve = self._init_args.get("preserve", False)

        for key, value in self._init_args.items():
            self._spec.AddArg(key, value)

        self._features = features

    @property
    def spec(self):
        return self._spec

    @property
    def schema(self):
        return self._schema

    @property
    def device(self):
        return self._device

    @property
    def preserve(self):
        return self._preserve

    def __call__(self, *inputs, **kwargs):
        # We do not handle multiple input sets for Reader as they do not have inputs
        args, arg_inputs = ops._separate_kwargs(kwargs)

        args = ops._resolve_double_definitions(args, self._init_args, keep_old=False)
        if self._name is not None:
            args = ops._resolve_double_definitions(args, {"name": self._name})  # restore the name

        self._preserve = self._preserve or args.get("preserve", False) or self._schema.IsNoPrune()

        feature_names = []
        features = []
        for feature_name, feature in self._features.items():
            feature_names.append(feature_name)
            features.append(feature)

        # Those arguments are added after the outputs are generated
        self.spec.AddArg("feature_names", feature_names)
        self.spec.AddArg("features", features)

        op_instance = ops._OperatorInstance(inputs, arg_inputs, args, self._init_args, self)

        outputs = {}
        for feature_name, output in zip(feature_names, op_instance.outputs):
            outputs[feature_name] = output

        return outputs


class TFRecordReader(_TFRecordReaderImpl, metaclass=ops._DaliOperatorMeta):
    _internal_schema_name = "_TFRecordReader"


class TFRecord(_TFRecordReaderImpl, metaclass=ops._DaliOperatorMeta):
    _internal_schema_name = "readers___TFRecord"
