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
"""Functions used to extract and analyze stacks.  Faster than Python libs."""
# pylint: disable=g-bad-name
import collections
import inspect
import threading
import traceback
import pkgutil

import pprint

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
        self.internal_map = {}  # _tf_stack.PyBindSourceMap() #TODO

    def update(self):
        # pp = pprint.PrettyPrinter(indent=4)
        # print(
        #     f"TODO: update(): {pp.pformat(self.internal_map)} "
        #     "\n<-\n"
        #     f"{pp.pformat(self.get_effective_source_map())}"
        # )
        # self.internal_map.update_to(tuple(self.get_effective_source_map().items()))
        # Based on the update_to implementation in tf_stack.cc, we rewrite it to pure Python.
        # The get_effective_source_map() recalculates recursively if it doesn't have the cached
        # value
        self.internal_map.clear()
        self.internal_map.update(self.get_effective_source_map())

    def get_effective_source_map(self):
        """Returns a map (filename, lineno) -> (filename, lineno, function_name)."""
        raise NotImplementedError("subclasses need to override this")


EMPTY_DICT = {}


class SentinelMapper(StackTraceMapper):
    def get_effective_source_map(self):
        return EMPTY_DICT


class StackTraceFilter(StackTraceTransform):
    """Allows filtering traceback information by removing superfluous frames."""

    _stack_dict = _source_filter_stacks

    def __init__(self):
        self.internal_set = set()  # _tf_stack.PyBindFileSet() # TODO

    def update(self):
        # pp = pprint.PrettyPrinter(indent=4)
        # print(
        #     f"TODO: update(): {pp.pformat(self.internal_set)} \n
        # <-\n {pp.pformat(self.get_filtered_filenames())}"
        # )
        # self.internal_set.update_to(set(self.get_filtered_filenames()))
        self.internal_set.clear()
        self.internal_set.update(self.get_filtered_filenames())

    def get_filtered_filenames(self):
        raise NotImplementedError("subclasses need to override this")


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


class CustomModuleFilter(StackTraceFilter):
    """Filters stack frames from the modules that were listed for this filter"""

    def __init__(self, module_filter: "Union[List[module], module]"):
        super().__init__()
        self._filtered_filenames = set()
        if not isinstance(module_filter, list):
            module_filter = [module_filter]
        for module in module_filter:
            try:
                module_file = inspect.getfile(module)
                init_py = "/__init__.py"
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


def _collapse_callstack(stack_summary):
    """With autograph it may appear as if we have several execution points within the same
    function. This function leaves only the latest entry for a given function invoked in that
    file.
    """
    seen_functions = set()
    rev_result = []
    for i in range(len(stack_summary) - 1, -1, -1):
        seen_elem = stack_summary[i].filename, stack_summary[i].name
        if seen_elem in seen_functions:
            continue
        seen_functions.add(seen_elem)
        rev_result.append(stack_summary[i])
    return list(reversed(rev_result))


def extract_stack(
    skip_bottom_frames=0, skip_top_frames=0, filter_modules=True, collapse_callstack=True
):
    # Returns a StackSummary which inherits from list, and contains traceback.FrameSummary
    # objects. Frame summary contains filename, lineno, name and line (string representing context).
    # The source mapper contains:
    # (loc.filename, loc.lineno)] = (
    #             origin.loc.filename,
    #             origin.loc.lineno,
    #             origin.function_name,
    #         )
    # The filter is a set of module names - how to use it?

    # -1 so we drop extract_stack frame
    stack_summary = traceback.extract_stack()[skip_bottom_frames : -1 - skip_top_frames]

    thread_key = _get_thread_key()
    frame_map = _source_mapper_stacks[thread_key][-1].internal_map
    frame_filter = _source_filter_stacks[thread_key][-1].internal_set
    # print(f"TODO: extract_stack(): {tb_stack} {frame_map} {frame_filter}")]
    origin_stack_summary = []
    for frame_entry in stack_summary:
        ag_entry = (frame_entry.filename, frame_entry.lineno)

        if ag_entry in frame_map:
            origin_info = frame_map[ag_entry]
            origin_frame_entry = traceback.FrameSummary(
                filename=origin_info[0],
                lineno=origin_info[1],
                name=origin_info[2],
                line=frame_entry.line,
            )
        else:
            origin_frame_entry = frame_entry

        # if frame_entry.filename not in frame_filter:
        skip = False
        if filter_modules:
            for frame_filter_entry in frame_filter:
                if frame_entry.filename.startswith(frame_filter_entry):
                    skip = True
                    break
        if not skip:
            origin_stack_summary.append(origin_frame_entry)
    if collapse_callstack:
        origin_stack_summary = _collapse_callstack(origin_stack_summary)
    # pp = pprint.PrettyPrinter(indent=4)
    # print(
    #     f"Extracting stack:\nFrame filter:\n{pp.pformat(frame_filter)}\n"
    #     f"Old:\n{pp.pformat(stack_summary)}\nNew:\n{pp.pformat(origin_stack_summary)}"
    # )
    return origin_stack_summary


# def extract_stack(stacklevel=1):
#     """An eager-friendly alternative to traceback.extract_stack.

#     Args:
#       stacklevel: number of initial frames to skip when producing the stack.

#     Returns:
#       A list-like FrameSummary containing StackFrame-like objects, which are
#       namedtuple-like objects with the following fields: filename, lineno, name,
#       line, meant to masquerade as traceback.FrameSummary objects.
#     """
#     thread_key = _get_thread_key()
#     return _tf_stack.extract_stack(
#         _source_mapper_stacks[thread_key][-1].internal_map,
#         _source_filter_stacks[thread_key][-1].internal_set,
#         stacklevel,
#     )
