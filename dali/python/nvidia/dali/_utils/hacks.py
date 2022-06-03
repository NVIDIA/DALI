# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import collections.abc

_not_iterable = ()
_original_check = None


class _NotIterable:
    def __iter__(self):
        raise TypeError("The objects of type `", type(self), "` are not iterable.")


def _check_iterable(iterable, instance):
    if isinstance(instance, _not_iterable):
        return False
    return _original_check(iterable, instance)


def _hook_iterable_check():
    global _original_check
    global _not_iterable
    if _original_check is not None:
        return
    _original_check = type(collections.abc.Iterable).__instancecheck__
    type(collections.abc.Iterable).__instancecheck__ = _check_iterable
    if len(_not_iterable) == 0:
        _not_iterable = (_NotIterable,)


def not_iterable(cls, add_iter=True):
    """Makes an object non-iterable by raising a TypeError in __iter__ and suppressing
    the detection of the object as an instance of collections.abc.Iterable.
    """
    _hook_iterable_check()
    if add_iter:
        cls.__iter__ = _NotIterable.__iter__
    global _not_iterable
    s = set(_not_iterable)
    s.add(cls)
    _not_iterable = tuple(s)
