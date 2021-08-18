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

import pickle
import inspect
import sys
import types
import marshal
import importlib
import multiprocessing
from nvidia.dali.pickling import CustomPickler

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


class DaliForkingPickler(multiprocessing.reduction.ForkingPickler):

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


def dump(obj, file, protocol=2):
    DaliForkingPickler(file, protocol).dump(obj)


class DaliForkingPicklerReducer(multiprocessing.reduction.AbstractReducer):
    ForkingPickler = DaliForkingPickler
    register = DaliForkingPickler.register
    dump = dump


def register_reducers(mp, reducer):
    """
    If `reducer` implements multiprocessing.reduction.AbstractReducer then it is set in the multiprocessing 
    context `mp`, so that multiprocessing uses it over default Python pickler. In such case nothing is returned 
    and no additional pickling is required along the way, because custom reducer is expected to handle
    pickling accordingly.
    Alternatively, reducer might be a CustomPickler instance that will be returned from the function and
    used to create additional layer of pickling - external source callbacks will be first serialized using 
    provided CustomPickler and then passed to multiprocessing where, in the serialized form, they will be 
    forwarded by default Python pickler. CustomPickler must itself be picklable.
    Instead of providing CustomPickler instance directly, you can either provide module that contains 
    *dumps* and *loads* functions or a tuple where first item is the module and next optional two items are
    dictionaries of kwargs that should be passed to dumps and loads methods respectively.
    """
    if reducer is None:
        return
    if inspect.isclass(reducer) and issubclass(reducer, multiprocessing.reduction.AbstractReducer):
        # Python versions lower than 3.8 don't support customization of functions pickling 
        # as it is utilized in DaliForkingPicklerReducer
        version_info = sys.version_info
        if not issubclass(reducer, DaliForkingPicklerReducer) or\
                (version_info.major > 3 or (version_info.major == 3 and version_info.minor >= 8)):
            mp.reducer = reducer
        return
    if isinstance(reducer, CustomPickler):
        return reducer
    if hasattr(reducer, 'dumps') and hasattr(reducer, 'loads'):
        return CustomPickler.of_reducer(reducer)
    if isinstance(reducer, (tuple, list)):
        params = [None] * 3
        for i, item in enumerate(reducer):
            params[i] = item
        reducer, kwargs_dumps, kwargs_loads = params
        return CustomPickler.of_reducer(reducer, kwargs_dumps, kwargs_loads)
    raise ValueError("Unsupported reducer value provided.")

