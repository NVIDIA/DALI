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

class CustomPickler:

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
    Use this decorator on a top-level function to hint DALI to serialize it by value
    (as lambdas and local functions are). It might be useful when working with
    external source in parallel mode if source callback cannot be imported by a worker process,
    for instance if both pipeline and source callback are defined in the same jupyter notebook
    and selected method of starting workers is set to spawn.
    """
    if inspect.isfunction(fun):
        setattr(fun, '_dali_pickle_by_value', True)
        return fun
    else:
        raise TypeError("Only functions can be explicitely set to be pickled by value")
