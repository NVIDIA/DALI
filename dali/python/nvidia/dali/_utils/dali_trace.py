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
    #             origin.source_code_line,
    #         )
    # The frame map and filter are only populated when we use AG, so without conditional mode
    # we will simply extract part of the stack.
    frame_map = get_frame_map()
    frame_filter = get_frame_filter()

    # -1 so we drop extract_stack frame
    stack_summary = traceback.extract_stack()[skip_bottom_frames : -1 - skip_top_frames]

    if not frame_map and not frame_filter:
        return stack_summary

    # print(f"TODO: extract_stack(): {tb_stack} {frame_map} {frame_filter}")]
    origin_stack_summary = []
    current_function_region = None
    for i, frame_entry in enumerate(stack_summary):
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

        # if frame_entry.filename not in frame_filter:
        skip = False
        if filter_modules:
            for frame_filter_entry in frame_filter:
                if frame_entry.filename.startswith(frame_filter_entry):
                    skip = True
                    break
        # Transformed function, so it's not something from filtered AutoGraph modules.
        # It can be from AutoAugment.
        if (frame_entry.filename.endswith("nvidia/dali/_autograph/impl/api.py") and frame_entry.name == "converted_call"):
            prepare_for_ag_call_region = True
        if (frame_entry.filename.endswith("nvidia/dali/_autograph/impl/api.py") and frame_entry.name == "_call_unconverted"):
            prepare_for_ag_call_region = False

        if prepare_for_ag_call_region and not skip:
            prepare_for_ag_call_region = False
            current_function_region = origin_frame_entry
            skip = True
            origin_stack_summary.append(origin_frame_entry)



        # if origin_frame_entry.name == f"autograph__{frame_entry.name}":
        #     # we start the function region here
        #     print(f"Start of function: {origin_frame_entry.name}")
        #     current_function_region = origin_frame_entry.name
        if not skip:
            if origin_stack_summary and origin_stack_summary[-1].name == current_function_region.name and origin_stack_summary[-1].filename == current_function_region.filename:
                origin_stack_summary.pop()
            origin_stack_summary.append(origin_frame_entry)
        print(f"\n\nProcessing: [{i}] {skip=}:\n{frame_entry=}\n->\n{origin_frame_entry=}")
    # if collapse_callstack:
    #     origin_stack_summary = _collapse_callstack(origin_stack_summary)
    # pp = pprint.PrettyPrinter(indent=4)
    # print(
    #     f"Extracting stack:\nFrame filter:\n{pp.pformat(frame_filter)}\n"
    #     f"Old:\n{pp.pformat(stack_summary)}\nNew:\n{pp.pformat(origin_stack_summary)}"
    # )
    return origin_stack_summary



# TODO(klecki!!!!): CHECK HOW THIS BEHAVES WITH allowed calls and automatic augments.
