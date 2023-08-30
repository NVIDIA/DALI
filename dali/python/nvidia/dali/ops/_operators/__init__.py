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

"""
Module dedicated for keeping internal implementation of Python wrappers and extensions
over the automatically generated operator bindings.

Some operators provided additional functionality, for example dictionary output in case of
TFRecord reader.

Typically, each such operator should be:
1. implemented in this module,
2. name should be  registered using one of the nvidia.dali.ops._registry.register_xxx_op(),
3. operator class should be reimported into the nvidia.dali.ops module,
4. the class should be reexposed in fn API via _wrap_op call.
"""
