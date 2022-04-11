# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect


def _discover_autoserialize(module, visited=[]):  # TODO test
    """
    TODO
    :param module:
    :return:
    """
    assert module is not None
    ret = []
    mems = inspect.getmembers(module)
    modules = []
    for mem in mems:
        obj = getattr(module, mem[0], None)
        if inspect.ismodule(obj) and mem[1] not in visited:
            modules.append(mem[0])
            visited.append(mem[1])
        elif inspect.isfunction(obj) and getattr(obj, 'autoserialize_me', False):
            ret.append(obj)
    for mod in modules:
        ret.extend(_discover_autoserialize(getattr(module, mod, None), visited))
    return ret


def invoke_autoserialize(module, filename):
    """
    TODO
    :param module:
    :param filename:
    :return:
    """
    autoserialize_me_functions = _discover_autoserialize(module)
    assert len(
        autoserialize_me_functions) == 1, f"Precisely one autoserialize function must exist in the module. Discovered: {autoserialize_me_functions}"
    dali_pipeline = autoserialize_me_functions[0]
    pipe = dali_pipeline()
    pipe.serialize(filename=filename)


def autoserialize(dali_pipeline):
    """
    Decorator, that marks a DALI pipeline (represented by :meth:`nvidia.dali.pipeline_def`) for
    autoserialization in [DALI Backend's](https://github.com/triton-inference-server/dali_backend#dali-triton-backend) model repository.

    For details about the autoserialization feature, please refer to the
    [DALI Backend documentation](https://github.com/triton-inference-server/dali_backend#autoserialization).

    To perform autoserialization, please refer to :meth:`invoke_autoserialize`.

    Only a ``pipeline_def`` can be decorated with ``autoserialize``.

    Only one ``pipeline_def`` may be decorated with ``autoserialize`` in given program.

    :param dali_pipeline: DALI Python model definition (``pipeline_def``)
    """
    assert getattr(dali_pipeline, "is_pipeline_def",
                   False), "Only `@pipeline_def` can be decorated with `@triton.autoserialize`."
    dali_pipeline.autoserialize_me = True
    return dali_pipeline
