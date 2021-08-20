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

import inspect
import pickle
import inspect
import sys
import types
import marshal
import importlib
import io

dummy_lambda = lambda : 0

def set_funcion_state(fun, state):
    fun.__globals__.update(state['global_refs'])
    fun.__defaults__ = state['defaults']
    fun.__kwdefaults__ = state['kwdefaults']

def function_unpickle(name, qualname, code, closure):
    code = marshal.loads(code)
    globs = {'__builtins__': __builtins__}
    fun = types.FunctionType(code, globs, name, closure=closure)
    fun.__qualname__ = qualname
    return fun

def function_by_value_reducer(fun):
    cl_vars = inspect.getclosurevars(fun)
    code = marshal.dumps(fun.__code__)
    basic_def = (fun.__name__, fun.__qualname__, code, fun.__closure__)
    fun_context = {
        'global_refs': cl_vars.globals,
        'defaults': fun.__defaults__,
        'kwdefaults': fun.__kwdefaults__
    }
    return function_unpickle, basic_def, fun_context, None, None, set_funcion_state

def module_unpickle(name, origin, submodule_search_locations):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, origin,
        submodule_search_locations=submodule_search_locations)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def module_reducer(module):
    spec = module.__spec__
    return module_unpickle, (spec.name, spec.origin, spec.submodule_search_locations)

def set_cell_state(cell, state):
    cell.cell_contents = state['cell_contents']

def cell_unpickle():
    return types.CellType(None)

def cell_reducer(cell):
    return (cell_unpickle, tuple(), {'cell_contents': cell.cell_contents}, None, None, set_cell_state)


class DaliCallbackPickler(pickle.Pickler):

    def reducer_override(self, obj):
        if inspect.ismodule(obj):
            return module_reducer(obj)
        if isinstance(obj, types.CellType):
            return cell_reducer(obj)
        if inspect.isfunction(obj):
            if isinstance(obj, type(dummy_lambda)) and obj.__name__ == dummy_lambda.__name__ or \
                    getattr(obj, '_dali_pickle_by_value', False):
                return function_by_value_reducer(obj)
            if '<locals>' in obj.__qualname__:
                try:
                    pickle.dumps(obj)
                except AttributeError as e:
                    if "Can't pickle local object" in str(e):
                        return function_by_value_reducer(obj)
        return NotImplemented


def dumps(obj, protocol=None, **kwargs):
    f = io.BytesIO()
    DaliCallbackPickler(f, protocol, **kwargs).dump(obj)
    return f.getvalue()


loads = pickle.loads


class CustomPickler:

    @classmethod
    def create(cls, py_callback_pickler):
        if py_callback_pickler is None or isinstance(py_callback_pickler, cls):
            return py_callback_pickler
        print(py_callback_pickler, dir(py_callback_pickler))
        if hasattr(py_callback_pickler, 'dumps') and hasattr(py_callback_pickler, 'loads'):
            return cls.of_reducer(py_callback_pickler)
        if isinstance(py_callback_pickler, (tuple, list)):
            params = [None] * 3
            for i, item in enumerate(py_callback_pickler):
                params[i] = item
            reducer, kwargs_dumps, kwargs_loads = params
            return cls.of_reducer(reducer, kwargs_dumps, kwargs_loads)
        raise ValueError("Unsupported py_callback_pickler value provided.")

    @classmethod
    def of_reducer(cls, reducer, dumps_kwargs=None, loads_kwargs=None):
        return cls(reducer.dumps, reducer.loads, dumps_kwargs, loads_kwargs)

    def __init__(self, dumps, loads, dumps_kwargs, loads_kwargs):
        self._dumps = dumps
        self._loads = loads
        self.dumps_kwargs = dumps_kwargs or {}
        self.loads_kwargs = loads_kwargs or {}

    def dumps(self, obj):
        return self._dumps(obj, **self.dumps_kwargs)

    def loads(self, obj):
        return self._loads(obj, **self.loads_kwargs)


def pickle_by_value(fun):
    """
    Hints parallel external source to serialize a decorated global function by value
    rather than by reference, which would be a default behavior of Python's pickler.
    """
    if inspect.isfunction(fun):
        setattr(fun, '_dali_pickle_by_value', True)
        return fun
    else:
        raise TypeError("Only functions can be explicitely set to be pickled by value")
