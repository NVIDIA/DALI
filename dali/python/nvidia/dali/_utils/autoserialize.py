# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def _is_marked_autoserializable(object):
    return getattr(object, "_is_autoserialize", False)


def _discover_autoserialize(module, visited):
    """
    Traverses a module tree given by the head module and
    returns all functions that are marked with ``@autoserialize`` decorator.

    :param module: Module currently searched.
    :param visited: Paths to the ``__init__.py`` of the modules already searched.
    :return: All functions that are marked with ``@autoserialize`` decorator.
    """
    assert module is not None
    ret = []
    try:
        module_members = inspect.getmembers(module)
    except (ModuleNotFoundError, ImportError):
        # If any module can't be inspected, DALI will not be able to find the @autoserialize
        # anyway. We can just skip this module.
        return ret
    modules = []
    for name, path in module_members:
        obj = getattr(module, name, None)
        if inspect.ismodule(obj) and path not in visited:
            modules.append(name)
            visited.append(path)
        elif inspect.isfunction(obj) and _is_marked_autoserializable(obj):
            ret.append(obj)
    for mod in modules:
        ret.extend(_discover_autoserialize(getattr(module, mod, None), visited=visited))
    return ret


def invoke_autoserialize(head_module, filename):
    """
    Perform the autoserialization of a function marked by
        :meth:`nvidia.dali.plugin.triton.autoserialize`.

    Assuming, that user marked a function with ``@autoserialize`` decorator, the
    ``invoke_autoserialize`` is a utility function, which will actually perform
    the autoserialization.
    It discovers the ``@autoserialize`` function in a module tree denoted by provided
    ``head_module`` and saves the serialized DALI pipeline to the file in the ``filename`` path.

    Only one ``@autoserialize`` function may exist in a given module tree.

    :param head_module: Module, denoting the model tree in which the decorated function shall exist.
    :param filename: Path to the file, where the output of serialization will be saved.
    """
    autoserialize_functions = _discover_autoserialize(head_module, visited=[])
    if len(autoserialize_functions) > 1:
        raise RuntimeError(
            f"Precisely one autoserialize function must exist in the module. "
            f"Found {len(autoserialize_functions)}: {autoserialize_functions}."
        )
    if len(autoserialize_functions) < 1:
        raise RuntimeError(
            "Precisely one autoserialize function must exist in the module. Found none."
        )
    dali_pipeline = autoserialize_functions[0]
    dali_pipeline().serialize(filename=filename)
