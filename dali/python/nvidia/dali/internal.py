# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import sys
import types
import inspect

def get_submodule(root, path):
    """Gets or creates sumbodule(s) of `root`.
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
        if str == '':
            return root
        path = path.split('.')

    module_name = root.__name__
    for part in path:
        m = getattr(root, part, None)
        module_name += '.' + part
        if m is None:
            m = sys.modules[module_name] = types.ModuleType(module_name)
            setattr(root, part, m)
        elif not isinstance(m, types.ModuleType):
            raise RuntimeError("The module {} already contains an attribute \"{}\", which is not a module, but {}".format(
                root, part, m))
        root = m
    return root


def _discriminate_args(decorated_func, func_kwargs={}, parent_funcs=[], parent_names=[]):
    """Split args to those applicable to parent function and the decorated function.

    Parameters
    ----------
    decorated_func : callable
        Function that is decorated
    func_kawrgs : dict
        arguments passed to the factory (the one created after decorating decorated_func)
    parent_funcs : [callable], optional
        Function from which we "inherit" additional kwargs, by default Pipeline.__init__
        Parent functions shouldn't have distinct overlapping arguments.
        The first match will be returned if there is more than one.
    parent_names : [str], optional
        Name for the parent function used in error messages, by default "Pipeline constructor"

    Returns
    -------
    dict, dict
        Arguments belonging to decorated_func, Arguments belonging to parent_func
    """
    decorated_argspec = inspect.getfullargspec(decorated_func)
    parent_argspecs = [inspect.getfullargspec(func) for func in parent_funcs]

    fn_args = {}
    parent_args = [{} for _ in parent_funcs]

    if decorated_argspec.varkw is not None:
        raise TypeError(
            "Using variadic keyword argument `**{}` in graph-defining function is not allowed.".format(
                decorated_argspec.varkw))

    def is_arg(name, argspec):
        return name in argspec.args or name in argspec.kwonlyargs

    for name, value in func_kwargs.items():
        is_fn_arg = is_arg(name, decorated_argspec)
        is_parent_arg = [is_arg(name, parent_argspec) for parent_argspec in parent_argspecs]
        if is_fn_arg:
            fn_args[name] = value
            if any(is_parent_arg):
                pos = is_parent_arg.index(True)
                print(
                    "Warning: the argument `{}` shadows a {} argument of the same name.".format(
                        name, parent_names[pos]))
        elif any(is_parent_arg):
            pos = is_parent_arg.index(True)
            parent_args[pos][name] = value
        else:
            if len(parent_names) == 1:
                error_end = "nor the {}.".format(parent_names[0])
            else:
                error_end = "nor any of the: "
                for i, parent_name in enumerate(parent_names):
                    error_end += parent_name
                    error_end += ", " if (i < len(parent_names) - 1) else ""

            raise ValueError(("Argument `{}` passed to the pipeline defintion is not defined " +
                              "by the decorated function, {}.").format(name, error_end))

    return (fn_args, *parent_args)
