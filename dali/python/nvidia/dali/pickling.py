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
from nvidia.dali._multiproc.reducers import register_dali_reducer


__all__ = ("register_dali_reducer", "CustomPickler", "pickle_by_value")

class CustomPickler:

    @classmethod
    def create(cls, py_callback_pickler):
        if py_callback_pickler is None or isinstance(py_callback_pickler, cls):
            return py_callback_pickler
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
    Hints parallel external source operator to serialize a decorated callback by value 
    rather than by reference, which would be a default behavior of Python's pickler.
    Decorator application is effective only with top-level functions used as a callback
    to ExternalSource working in parallel mode with Pipeline's `py_start_method` 
    specified to *spawn* running under Python 3.8 or above.
    It might be especially handy when working under Jupyter notebook, because of its limitation
    that processes *spawned* from the given notebook cannot import callbacks defined 
    in the same notebook. Pickling by value makes it possible to pass callback without 
    the need for imports from the main notebook in the spawned process.
    """
    if inspect.isfunction(fun):
        register_dali_reducer()
        setattr(fun, '_dali_pickle_by_value', True)
        return fun
    else:
        raise TypeError("Only functions can be explicitely set to be pickled by value")
