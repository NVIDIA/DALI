# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import traceback
from nvidia.dali._autograph.utils.tf_stack import get_frame_map, get_frame_filter
from nvidia.dali._autograph import is_frame_ag_call_entrypoint, is_frame_ag_call_unconverted


_origin_trace_enabled = True
# Processing options, mainly for debugging purposes
_collapse_ag_frames = True
_filter_ag_frames = True
_remap_ag_frames = True


def set_tracing(*, enabled: bool = None, options={}):
    """Enable or disable tracing of operator origin information.

    The default will change in the future, this API will remain private.

    Available options:
    * collapse_ag_frames: bool -
    * filter_ag_frames: bool - filter frames from AutoGraph and conditionals implementation
    * remap_ag_frames: bool - map AG produced frames back to user code

    """
    global _origin_trace_enabled
    global _collapse_ag_frames
    global _filter_ag_frames
    global _remap_ag_frames
    _collapse_ag_frames = options.get("collapse_ag_frames", _collapse_ag_frames)
    _filter_ag_frames = options.get("filter_ag_frames", _filter_ag_frames)
    _remap_ag_frames = options.get("remap_ag_frames", _remap_ag_frames)
    if enabled is not None:
        _origin_trace_enabled = enabled


def is_tracing_enabled():
    return _origin_trace_enabled


def get_stack_depth():
    """Get the number of stack frames up to the call of this function."""
    # Walking through frames is at least order of magnitude faster
    if hasattr(sys, "_getframe"):
        depth = 0
        frame = sys._getframe(1)
        # To be safe against unwanted reference cycles according to inspect module docs,
        # we force remove the reference to the frame.
        try:
            while frame:
                depth = depth + 1
                frame = frame.f_back
            return depth
        finally:
            del frame
    return len(traceback.extract_stack())


def _is_matching_function(prev_frame_summary, new_frame_summary):
    """Heuristic check if the frame summaries describe the same function.
    Any of the arguments can be none, this means we don't have a match
    """
    if prev_frame_summary is None or new_frame_summary is None:
        return False
    return (
        prev_frame_summary.name == new_frame_summary.name
        and prev_frame_summary.filename == new_frame_summary.filename
    )


def _filter_autograph_frames(stack_summary, frame_map, frame_filter):
    origin_stack_summary = []
    is_ag_function_call_start = False
    current_function_region = None

    for frame_entry in stack_summary:
        # The frame_map maps filename:lineno pairs of AG-produced code back to FrameSummary of
        # original code.
        ag_entry = (frame_entry.filename, frame_entry.lineno)

        if _remap_ag_frames and ag_entry in frame_map:
            origin_info = frame_map[ag_entry]
            origin_frame_entry = traceback.FrameSummary(
                filename=origin_info[0],
                lineno=origin_info[1],
                name=origin_info[2],
                line=origin_info[3],
            )
        else:
            origin_frame_entry = frame_entry

        # We will skip code that is related to internal AutoGraph and conditionals implementation,
        # leaving us with code produced by transforming user-code
        # See api.py:converted_call for details on filtering.
        skip = frame_filter.is_filtered(origin_frame_entry.filename)

        # Detect repeated appearance of function transformed by AG
        # AutoGraph is wrapping a function call - entry point
        # We always start with a converted call to a pipeline_def.
        if is_frame_ag_call_entrypoint(frame_entry):
            is_ag_function_call_start = True
            current_function_region = None
        # It quits to a non-AG code, treat it as normal from now-on
        if is_frame_ag_call_unconverted(frame_entry):
            is_ag_function_call_start = False
            current_function_region = None
        # User code - not filtered out
        if not skip:
            # We are in the first part of the converted_func call that is not skipped
            # (as we are in user code, remember the function)
            if is_ag_function_call_start:
                is_ag_function_call_start = False
                current_function_region = origin_frame_entry
                origin_stack_summary.append(origin_frame_entry)
            else:
                # If we are in the same function region, we replace previous entry so we keep only
                # the last one
                assert origin_stack_summary
                if _is_matching_function(origin_stack_summary[-1], current_function_region):
                    if _collapse_ag_frames:
                        origin_stack_summary.pop()
                else:
                    current_function_region = None
                origin_stack_summary.append(origin_frame_entry)
        elif not _filter_ag_frames:
            origin_stack_summary.append(origin_frame_entry)
    return origin_stack_summary


def extract_stack(start_frame=0, end_frame=None):
    """Extract list of FrameSummary object from current stack, in the range [start_frame:end_frame],
    where 0 is the first frame. If AutoGraph was used, the FrameSummary entries are filtered
    and remapped back to the user code.

    Returns
    -------
    List[FrameSummary]
    """
    # Returns a StackSummary which inherits from list, and contains traceback.FrameSummary
    # objects. Frame summary contains filename, lineno, name and line (string representing context).
    # -1 so we drop extract_stack frame
    stack_depth = get_stack_depth()
    limit = stack_depth - start_frame
    end_frame = end_frame - start_frame if end_frame is not None else -1
    stack_summary = traceback.extract_stack(limit=limit)[:end_frame]

    # If those are empty, AutoGraph transformations were not used, we can return as is
    frame_map = get_frame_map()
    frame_filter = get_frame_filter()
    if not frame_map and not frame_filter:
        return stack_summary

    return _filter_autograph_frames(stack_summary, frame_map, frame_filter)


def preprocess_stack_summary(stack_summary):
    """
    Split the list of FrameSummary into 4 separate list of each components.
    Preprocess the output of AST (trim the whitespace).

    Parameters
    ----------
    stack_summary : list[FrameSummary]

    Returns
    -------
    List[str], List[int], List[str], List[str]
        [filename], [lineno], [name], [line]
    """
    filename_stack = [frame_summary.filename for frame_summary in stack_summary]
    lineno_stack = [frame_summary.lineno for frame_summary in stack_summary]
    name_stack = [frame_summary.name for frame_summary in stack_summary]
    line_stack = [frame_summary.line.strip() for frame_summary in stack_summary]
    return filename_stack, lineno_stack, name_stack, line_stack
