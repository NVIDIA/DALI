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

import importlib
import sys
import types


def get_submodule(root, path):
    """Gets or creates submodule(s) of `root`.
    If the module path contains multiple parts, multiple modules are traversed or created

    Parameters
    ----------
        `root`
            module object or name of the root module
        `path`
            period-separated path of the submodule or a list/tuple of submodule names"""

    if isinstance(root, str):
        root = sys.modules[root]

    if not path:
        return root

    if isinstance(path, str):
        if str == "":
            return root
        path = path.split(".")

    module_name = root.__name__
    for part in path:
        m = getattr(root, part, None)
        module_name += "." + part
        if m is None:
            try:
                # Try importing existing module (if not loaded yet) to not overwrite it.
                m = importlib.import_module(module_name)
            except ModuleNotFoundError:
                m = sys.modules[module_name] = types.ModuleType(module_name)
            setattr(root, part, m)
        elif not isinstance(m, types.ModuleType):
            raise RuntimeError(
                f'The module {root} already contains an attribute "{part}", '
                f"which is not a module, but {m}"
            )
        root = m
    return root


def _adjust_operator_module(operator, api_module, submodule, impl_overwrite=None):
    """Adjust the __module__ of `operator` to point into the submodule of `api_module`
    pointed by the list of in `submodule`, for example:
        api_module = <nvidia.dali.ops module>
        submodule = ["experimental", "readers"]

    The original module where the operator code was generated is saved as `_impl_module` to allow
    access to it.

    If the operator has base class defined by hand, but it is generated with an automatic wrapper
    generator, we can point to the original implementation module via `impl_overwrite`.
    """
    module = get_submodule(api_module, submodule)
    operator._impl_module = operator.__module__ if impl_overwrite is None else impl_overwrite
    operator.__module__ = module.__name__
