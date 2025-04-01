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

import numpy as np
import traceback
import re
import fnmatch

from nvidia.dali import pipeline_def, fn, ops, Pipeline
from nvidia.dali.auto_aug import auto_augment, augmentations
from nvidia.dali.auto_aug.core import augmentation, Policy
from nvidia.dali.pipeline import do_not_convert
from nose2.tools import params
from test_utils import load_test_operator_plugin
from nvidia.dali.pipeline.experimental import pipeline_def as pipeline_def_experimental


load_test_operator_plugin()


op_mode = "dali.fn"
extracted_stacks = []
base_frame = 0


def capture_python_traces(fun, full_stack=False):
    """Run `fun` and collect all stack traces (in Python mode) in order of occurrence.
    Running the pipeline definition as Python serves as baseline for comparing the correctness
    of traces captured by DALI.

    Parameters
    ----------
    fun : callable
        Function to trace, typically we capture traces starting from the frame of this call
    full_stack : bool, optional
        Whether to capture the whole stack, by default False
    """
    global base_frame
    base_frame = len(traceback.extract_stack()) if not full_stack else 0

    global op_mode
    op_mode_bkp = op_mode
    op_mode = "python"
    fun()
    op_mode = op_mode_bkp
    result = ["".join(traceback.format_list(tb)) for tb in extracted_stacks]
    extracted_stacks.clear()
    return result


def get_python_trace_here(full_stack=False):
    python_tbs = capture_python_traces(origin_trace, full_stack=full_stack)
    # Remove last 6 lines, keep the last endline
    python_tbs = ["\n".join(tb.split("\n")[:-7]) + "\n" for tb in python_tbs]
    return python_tbs


def extract_trace_from_tl(out):
    """Extract string representing traces from the test operator that returns it as u8 tensor."""
    # Extract data from first sample
    arr = np.array(out[0])
    # View it as list of characters and decode to string
    return arr.view(f"S{arr.shape[0]}")[0].decode("utf-8")


def capture_dali_traces(pipe_def):
    """Run the pipeline and extract all the traces returned as the outputs of the pipeline"""
    p = pipe_def()
    outputs = p.run()
    result = [extract_trace_from_tl(output) for output in outputs]
    return result


def origin_trace():
    """Either return trace using test operator or capture it via Python API"""
    if op_mode == "dali.fn":
        return fn.origin_trace_dump()
    if op_mode == "dali.ops":
        return ops.OriginTraceDump()()  # Yup, super obvious __init__ + __call__
    # elif op_mode == "python":
    # Skip last frame as it differs from calling the fn.origin_trace_dump above.
    extracted_stacks.append(traceback.extract_stack()[base_frame:-1])
    return None


def origin_trace_glob(op_mode):
    return f"""*File "*/test_operator_origin_trace.py", line *, in origin_trace
    return {"fn.origin_trace_dump()" if op_mode == "dali.fn" else "ops.OriginTraceDump()()"}*
"""


def compare_traces(dali_tbs, python_tbs, end_glob=origin_trace_glob):
    assert len(dali_tbs) == len(python_tbs)
    regex = fnmatch.translate(end_glob(op_mode))
    for dali_tb, python_tb in zip(dali_tbs, python_tbs):
        err = f"Comparing dali_tb:\n{dali_tb}\nvs python_tb:\n{python_tb}"
        # Check if we skipped just one frame in python_tb (2 lines)
        assert dali_tb.count("\n") == python_tb.count("\n") + 2, err
        assert dali_tb.startswith(python_tb), err
        assert re.match(regex, dali_tb), f"{dali_tb} didn't match the expected end traceback"


test_modes = ["dali.fn", "dali.ops"]


@params(*test_modes)
def test_trace_almost_trivial(test_mode):
    global op_mode
    op_mode = test_mode

    def pipe():
        return origin_trace()

    python_tbs = capture_python_traces(pipe)

    dali_regular_pipe = pipeline_def(batch_size=2, num_threads=1, device_id=0)(pipe)
    dali_regular_tbs = capture_dali_traces(dali_regular_pipe)
    compare_traces(dali_regular_tbs, python_tbs)


def test_trace_almost_trivial_debug():
    global op_mode
    op_mode = "dali.fn"

    @pipeline_def_experimental(batch_size=2, num_threads=1, device_id=0, debug=True)
    def pipe():
        return origin_trace()

    p = pipe()
    (out,) = p.run()
    assert out.shape() == [(0,), (0,)], "Debug doesn't carry trace information"


@params(*test_modes)
def test_trace_scope(test_mode):
    global op_mode
    op_mode = test_mode

    pipe = Pipeline(batch_size=2, num_threads=1, device_id=0)

    with pipe:
        pipe.set_outputs(origin_trace())

    dali_tbs = capture_dali_traces(lambda: pipe)

    glob = f"""  File "*/test_operator_origin_trace.py", line *, in test_trace_scope
    pipe.set_outputs(origin_trace())
  File "*/test_operator_origin_trace.py", line *, in origin_trace
    return {"fn.origin_trace_dump()" if test_mode == "dali.fn" else "ops.OriginTraceDump()()"}*
"""
    regex = fnmatch.translate(glob)
    assert re.match(regex, dali_tbs[0]), f"{dali_tbs[0]} didn't match expected traceback"


@params(*test_modes)
def test_trace_push_current(test_mode):
    global op_mode
    op_mode = test_mode

    pipe = Pipeline(batch_size=2, num_threads=1, device_id=0)
    Pipeline.push_current(pipe)
    dn = origin_trace()
    pipe.set_outputs(dn)
    Pipeline.pop_current()

    dali_tbs = capture_dali_traces(lambda: pipe)

    glob = f"""  File "*/test_operator_origin_trace.py", line *, in test_trace_push_current
    dn = origin_trace()
  File "*/test_operator_origin_trace.py", line *, in origin_trace
    return {"fn.origin_trace_dump()" if test_mode == "dali.fn" else "ops.OriginTraceDump()()"}*
"""
    regex = fnmatch.translate(glob)
    assert re.match(regex, dali_tbs[0]), f"{dali_tbs[0]} didn't match expected traceback"


@params(*test_modes)
def test_trace_recursive(test_mode):
    global op_mode
    op_mode = test_mode

    def recursive_helper(n=2):
        if n:
            return recursive_helper(n - 1)
        else:
            # using list comprehension or similar will give slightly different results in AG
            # We can revisit and maybe tune the filtering/remapping?
            return origin_trace()

    def pipe():
        if 0:
            x = origin_trace()
        else:
            x = recursive_helper()
        return x

    python_tbs = capture_python_traces(pipe)

    dali_regular_pipe = pipeline_def(batch_size=2, num_threads=1, device_id=0)(pipe)
    dali_regular_tbs = capture_dali_traces(dali_regular_pipe)
    compare_traces(dali_regular_tbs, python_tbs)

    dali_cond_pipe = pipeline_def(
        batch_size=2, num_threads=1, device_id=0, enable_conditionals=True
    )(pipe)
    dali_cond_tbs = capture_dali_traces(dali_cond_pipe)
    compare_traces(dali_cond_tbs, python_tbs)


@params(*test_modes)
def test_trace_recursive_do_not_convert(test_mode):
    global op_mode
    op_mode = test_mode

    @do_not_convert
    def recursive_helper(n=2):
        if n:
            return recursive_helper(n - 1)
        else:
            return origin_trace()

    def pipe():
        return recursive_helper()

    python_tbs = capture_python_traces(pipe)

    dali_cond_pipe = pipeline_def(
        batch_size=2, num_threads=1, device_id=0, enable_conditionals=True
    )(pipe)
    dali_cond_tbs = capture_dali_traces(dali_cond_pipe)
    compare_traces(dali_cond_tbs, python_tbs)


@params(*test_modes)
def test_trace_auto_aug(test_mode):
    global op_mode
    op_mode = test_mode

    # TODO(klecki): AutoGraph loses mapping for the trace_aug and points to a transformed file.
    # Find out if we can somehow propagate that code mapping back. @do_not_convert helps with
    # keeping regular code.
    @do_not_convert
    def my_custom_policy() -> Policy:
        @augmentation
        def trace_aug(data, _):
            return origin_trace()

        return Policy(
            name="SimplePolicy",
            num_magnitude_bins=11,
            sub_policies=[
                [(trace_aug, 1.0, None), (augmentations.identity, 0.8, None)],
            ],
        )

    @pipeline_def(batch_size=2, num_threads=1, device_id=0, enable_conditionals=True)
    def pipe():
        images = np.full((100, 100, 3), 42, dtype=np.uint8)

        augmented_images = auto_augment.apply_auto_augment(my_custom_policy(), images)

        return augmented_images

    dali_cond_tbs = capture_dali_traces(pipe)

    # It's not really feasible to generate the baseline for comparison, so we just record
    # start and end of the expected trace, so not to rely on the callstack of the implementation
    stacktrace_glob = f"""  File "*/test_operator_origin_trace.py", line *, in pipe
    augmented_images = auto_augment.apply_auto_augment(my_custom_policy(), images)
  File "*nvidia/dali/auto_aug/auto_augment.py", line *, in apply_auto_augment
*
  File "*/test_operator_origin_trace.py", line *, in trace_aug
    return origin_trace()
  File "*/test_operator_origin_trace.py", line *, in origin_trace
    return {"fn.origin_trace_dump()" if test_mode == "dali.fn" else "ops.OriginTraceDump()()"}*
"""
    regex = fnmatch.translate(stacktrace_glob)
    assert re.match(regex, dali_cond_tbs[0]), f"{dali_cond_tbs[0]} didn't match expected traceback"


@params(*test_modes)
def test_trace_outside_pipeline(test_mode):
    global op_mode
    op_mode = test_mode

    if test_mode == "dali.fn":
        dn = fn.origin_trace_dump()
    else:
        dn = ops.OriginTraceDump()()

    python_tbs = get_python_trace_here(full_stack=True)

    @pipeline_def(batch_size=2, num_threads=1, device_id=0)
    def pipe():
        return dn

    def glob(op_mode):
        return f"""*File "*/test_operator_origin_trace.py", line *, in test_trace_outside_pipeline
    dn = {"fn.origin_trace_dump()" if op_mode == "dali.fn" else "ops.OriginTraceDump()()"}*"""

    dali_tbs = capture_dali_traces(pipe)
    compare_traces(
        dali_tbs,
        python_tbs,
        end_glob=glob,
    )
