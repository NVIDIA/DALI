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

import traceback
from nvidia.dali._autograph.utils.tf_stack import get_frame_map, get_frame_filter
from nvidia.dali._autograph import is_frame_converted_call, is_frame_call_unconverted


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

    # Returns a StackSummary which inherits from list, and contains traceback.FrameSummary
    # objects. Frame summary contains filename, lineno, name and line (string representing context).
    # The source mapper contains:
    # (loc.filename, loc.lineno)] = (
    #             origin.loc.filename,
    #             origin.loc.lineno,
    #             origin.function_name,
    #             origin.source_code_line,
    #         )

    origin_stack_summary = []
    is_ag_function_call_start = False
    current_function_region = None

    for frame_entry in stack_summary:
        # The frame_map maps filename:lineno pairs of AG-produced code back to FrameSummary of
        # original code.
        ag_entry = (frame_entry.filename, frame_entry.lineno)

        if ag_entry in frame_map:
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
        skip = frame_filter.is_filtered(frame_entry.filename)

        # AutoGraph is wrapping a function call
        if is_frame_converted_call(frame_entry):
            is_ag_function_call_start = True
        # It quits to a non-AG code, treat it as normal from now-on
        if is_frame_call_unconverted(frame_entry):
            is_ag_function_call_start = False
            current_function_region = None
        # We are in the first part of the converted_func call (as we are in user code, remember
        # the function)
        if is_ag_function_call_start and not skip:
            is_ag_function_call_start = False
            current_function_region = origin_frame_entry
            skip = True
            origin_stack_summary.append(origin_frame_entry)

        if not skip:
            # If we are in the same function region, we replace previous entry so we keep only the
            # last one.
            assert origin_stack_summary
            if _is_matching_function(origin_stack_summary[-1], current_function_region):
                origin_stack_summary.pop()
            origin_stack_summary.append(origin_frame_entry)
    return origin_stack_summary


def extract_stack(skip_bottom_frames=0, skip_top_frames=0):
    """Extract list of FrameSummary object, optionally skipping the bottom and top ones from the place
    of the call. If AutoGraph was used, the FrameSummary entries are filtered and remapped back
    to the user code.'

    Returns
    -------
    List[FrameSummary]
    """
    # -1 so we drop extract_stack frame
    stack_summary = traceback.extract_stack()[skip_bottom_frames : -1 - skip_top_frames]

    # If those are empty, AutoGraph transformations were not used, we can return as is
    frame_map = get_frame_map()
    frame_filter = get_frame_filter()
    if not frame_map and not frame_filter:
        return stack_summary

    return _filter_autograph_frames(stack_summary, frame_map, frame_filter)


def separate_stack_summary(stack_summary):
    """Split the list of FrameSummary into 4 separate list of each components

    Parameters
    ----------
    stack_summary : _type_
        _description_

    Returns
    -------
    List(str), List(int), List(str), List(str)
        [filename], [lineno], [name], [line]
    """
    filename_stack = [frame_summary.filename for frame_summary in stack_summary]
    lineno_stack = [frame_summary.lineno for frame_summary in stack_summary]
    name_stack = [frame_summary.name for frame_summary in stack_summary]
    line_stack = [frame_summary.line for frame_summary in stack_summary]
    return filename_stack, lineno_stack, name_stack, line_stack
