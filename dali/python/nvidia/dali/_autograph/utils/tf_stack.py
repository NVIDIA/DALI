# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pylint: disable=g-bad-name
import collections
import inspect
import threading
from types import ModuleType
from typing import List, Union


# Generally such lookups should be done using `threading.local()`. See
# https://blogs.gnome.org/jamesh/2008/06/11/tls-python/ for a detailed
# explanation of why. However the transform stacks are expected to be empty
# when a thread is joined, so reusing the key does not introduce a correctness
# issue. Moreover, get_ident is faster than storing and retrieving a unique
# key in a thread local store.
_get_thread_key = threading.get_ident


# TODO(mdan): Move these to C++ as well.
# Moving to C++ can further avoid extra copies made by get_effective_map.
_source_mapper_stacks = collections.defaultdict(lambda: [SentinelMapper()])
_source_filter_stacks = collections.defaultdict(lambda: [SentinelFilter()])


class StackTraceTransform(object):
    """Base class for stack trace transformation functions."""

    _stack_dict = None  # Subclasses should override
    _thread_key = None

    def __enter__(self):
        # Any given instance is assumed to be used by a single thread, which reduces
        # expensive thread local lookups.
        if self._thread_key is None:
            self._thread_key = _get_thread_key()
        else:
            assert self._thread_key == _get_thread_key(), "Shared across threads?"

        stack = self._stack_dict[self._thread_key]
        self.parent = stack[-1]
        stack.append(self)
        self.update()
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        top = self._stack_dict[self._thread_key].pop()
        assert top is self, "Concurrent access?"

    def update(self):
        raise NotImplementedError("subclasses need to override this")


class StackTraceMapper(StackTraceTransform):
    """Allows remapping traceback information to different source code."""

    _stack_dict = _source_mapper_stacks

    def __init__(self):
        self.internal_map = {}

    def update(self):
        self.internal_map.clear()
        self.internal_map.update(self.get_effective_source_map())

    def get_effective_source_map(self):
        """Returns a map (filename, lineno) -> (filename, lineno, function_name, line)."""
        raise NotImplementedError("subclasses need to override this")


EMPTY_DICT = {}


class SentinelMapper(StackTraceMapper):

    def get_effective_source_map(self):
        return EMPTY_DICT


class StackTraceFilter(StackTraceTransform):
    """Allows filtering traceback information by removing superfluous frames."""

    _stack_dict = _source_filter_stacks

    def __init__(self):
        self.internal_set = set()

    def update(self):
        self.internal_set.clear()
        self.internal_set.update(self.get_filtered_filenames())

    def get_filtered_filenames(self):
        raise NotImplementedError("subclasses need to override this")

    def is_filtered(self, filename):
        raise NotImplementedError("subclasses need to override this")

    def __bool__(self):
        return bool(self.internal_set)


EMPTY_SET = frozenset()


class SentinelFilter(StackTraceFilter):

    def get_filtered_filenames(self):
        return EMPTY_SET


class CurrentModuleFilter(StackTraceFilter):
    """Filters stack frames from the module where this is used (best effort)."""

    def __init__(self):
        super().__init__()
        filter_filename = None
        outer_f = None
        f = inspect.currentframe()
        try:
            if f is not None:
                # The current frame is __init__. The first outer frame should be the
                # caller.
                outer_f = f.f_back
                if outer_f is not None:
                    filter_filename = inspect.getsourcefile(outer_f)
            self._filename = filter_filename
            # This may be called repeatedly: once on entry by the superclass, then by
            # each child context manager.
            self._cached_set = None
        finally:
            # Avoid reference cycles, see:
            # https://docs.python.org/3.7/library/inspect.html#the-interpreter-stack
            del f
            del outer_f

    def get_filtered_filenames(self):
        if self._cached_set is not None:
            return self._cached_set

        filtered_filenames = frozenset((self._filename,))
        if self.parent is not None:
            filtered_filenames |= self.parent.get_filtered_filenames()
        self._cached_set = filtered_filenames
        return filtered_filenames

    def is_filtered(self, filename):
        return filename in self.internal_set


class CustomModuleFilter(StackTraceFilter):
    """Filters stack frames from the modules that were listed for this filter.

    We detect the top directory of given module and filter all frames from that path or its subpath.
    """

    def __init__(self, module_filter: Union[List[ModuleType], ModuleType]):
        super().__init__()
        self._filtered_filenames = set()
        if not isinstance(module_filter, list):
            module_filter = [module_filter]
        for module in module_filter:
            try:
                module_file = inspect.getfile(module)
                init_py = "__init__.py"
                if module_file.endswith(init_py):
                    module_file = module_file[: -len(init_py)]
                self._filtered_filenames.add(module_file)
            except TypeError as e:
                raise TypeError(f"{module} is a built-in module and cannot be filtered.") from e
        self._cached_set = None

    def get_filtered_filenames(self):
        if self._cached_set is not None:
            return self._cached_set

        filtered_filenames = frozenset(self._filtered_filenames)
        if self.parent is not None:
            filtered_filenames |= self.parent.get_filtered_filenames()
        self._cached_set = filtered_filenames
        return filtered_filenames

    def is_filtered(self, filename):
        for frame_filter_entry in self.internal_set:
            if filename.startswith(frame_filter_entry):
                return True
        return False


def get_frame_map():
    thread_key = _get_thread_key()
    return _source_mapper_stacks[thread_key][-1].internal_map


def get_frame_filter():
    thread_key = _get_thread_key()
    return _source_filter_stacks[thread_key][-1]
