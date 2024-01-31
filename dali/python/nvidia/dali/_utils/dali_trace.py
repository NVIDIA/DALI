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


def extract_stack(
    skip_bottom_frames=0, skip_top_frames=0, filter_modules=True, collapse_callstack=False
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
