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

import inspect
import pickle  # nosec B403
import sys
import types
import marshal
import importlib


# Don't allow any reformatters turning it into regular `def` function.
# The whole point of this object is to have
# properties (name) specific to lambda.
dummy_lambda = lambda: 0  # noqa: E731


# unfortunately inspect.getclosurevars does not yield global names referenced by
# the code syntactically nested inside the function, this includes nested functions
# and list comprehension, for instance in case of [exp1 for exp2 in exp3] occurring inside
# a function, references from exp1 would be omitted


def get_global_references_from_nested_code(code, global_scope, global_refs):
    for constant in code.co_consts:
        if inspect.iscode(constant):
            closure = tuple(types.CellType(None) for _ in range(len(constant.co_freevars)))
            dummy_function = types.FunctionType(
                constant, global_scope, "dummy_function", closure=closure
            )
            global_refs.update(inspect.getclosurevars(dummy_function).globals)
            get_global_references_from_nested_code(constant, global_scope, global_refs)


def set_funcion_state(fun, state):
    fun.__globals__.update(state["global_refs"])
    fun.__defaults__ = state["defaults"]
    fun.__kwdefaults__ = state["kwdefaults"]


def function_unpickle(name, qualname, code, closure):
    code = marshal.loads(code)  # nosec B302
    global_scope = {"__builtins__": __builtins__}
    fun = types.FunctionType(code, global_scope, name, closure=closure)
    fun.__qualname__ = qualname
    return fun


def function_by_value_reducer(fun):
    cl_vars = inspect.getclosurevars(fun)
    code = marshal.dumps(fun.__code__)
    basic_def = (fun.__name__, fun.__qualname__, code, fun.__closure__)
    global_refs = dict(cl_vars.globals)
    get_global_references_from_nested_code(fun.__code__, fun.__globals__, global_refs)
    fun_context = {
        "global_refs": global_refs,
        "defaults": fun.__defaults__,
        "kwdefaults": fun.__kwdefaults__,
    }
    return function_unpickle, basic_def, fun_context, None, None, set_funcion_state


def module_unpickle(name, origin, submodule_search_locations):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, origin, submodule_search_locations=submodule_search_locations
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def module_reducer(module):
    spec = module.__spec__
    return module_unpickle, (spec.name, spec.origin, spec.submodule_search_locations)


def set_cell_state(cell, state):
    cell.cell_contents = state["cell_contents"]


def cell_unpickle():
    return types.CellType(None)


def cell_reducer(cell):
    return (
        cell_unpickle,
        tuple(),
        {"cell_contents": cell.cell_contents},
        None,
        None,
        set_cell_state,
    )


class DaliCallbackPickler(pickle.Pickler):
    def reducer_override(self, obj):
        if inspect.ismodule(obj):
            return module_reducer(obj)
        if isinstance(obj, types.CellType):
            return cell_reducer(obj)
        if inspect.isfunction(obj):
            if (
                isinstance(obj, type(dummy_lambda))
                and obj.__name__ == dummy_lambda.__name__
                or getattr(obj, "_dali_pickle_by_value", False)
            ):
                return function_by_value_reducer(obj)
            try:
                pickle.dumps(obj)
            except AttributeError as e:
                str_e = str(e)
                # For Python <3.12.5 and 3.12.5 respectively.
                if "Can't pickle local object" in str_e or "Can't get local object" in str_e:
                    return function_by_value_reducer(obj)
            except pickle.PicklingError as e:
                str_e = str(e)
                # For jupyter notebook issues and Python 3.12.5+ respectively
                if "it's not the same object as" in str_e or "Can't pickle local object" in str_e:
                    return function_by_value_reducer(obj)
        return NotImplemented
