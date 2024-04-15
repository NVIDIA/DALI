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


def _repack_list(sets, fn):
    """Repack list from [[a, b, c], [a', b', c'], ....]
    to [fn(a, a', ...), fn(b, b', ...), fn(c, c', ...)]
    where fn can be `tuple` or `list`
    Assume that all elements of input have the same length
    """
    output_list = []
    arg_list_len = len(sets[0])
    for i in range(arg_list_len):
        output_list.append(fn(input_set[i] for input_set in sets))
    return output_list


def _repack_output_sets(outputs):
    """Repack and "transpose" the output sets, from groups of outputs of individual operators
    to interleaved groups of consecutive outputs, that is from:
    [[out0, out1, out2], [out0', out1', out2'], ...] produce:
    [[out0, out0', ...], [out1, out1', ...], [out2, out2', ...]]


    Assume that all elements of input have the same length
    If the inputs were 1-elem lists, it is flattened, that is:
    [[out0], [out0'], [out0''], ...] -> [out0, out0', out0'', ...]
    """
    if len(outputs) > 1 and len(outputs[0]) == 1:
        output = []
        for elem in outputs:
            output.append(elem[0])
        return output
    return _repack_list(outputs, list)


def _build_input_sets(inputs, op_name):
    """Detect if the list of positional inputs [Inp_0, Inp_1, Inp_2, ...], represents Multiple
    Input Sets (MIS) to operator and prepare lists of regular DataNode-only positional inputs to
    individual operator instances.

    If all Inp_i are DataNodes there are no MIS involved.
    If any of Inp_i is a list of DataNodes, this is considered a MIS. In that case, non-list
    Inp_i is repeated to match the length of the one that is a list, and those lists are regrouped,
    for example:

    inputs = [a, b, [x, y, z], [u, v, w]]

    # "a" and "b" are repeated to match the length of [x, y, z]:
    -> [[a, a, a], [b, b, b], [x, y, z], [u, v, w]]

    # input sets are rearranged, so they form a regular tuples of DataNodes suitable to being passed
    # to one Operator Instance.
    -> [(a, b, x, u), (a, b, y, v), (a, b, z, w)]

    This allows to create 3 operator instances, each with 4 positional inputs.

    Parameters
    ----------
    inputs : List of positional inputs
        The inputs are either DataNodes or lists of DataNodes indicating MIS.
    op_name : str
        Name of the invoked operator, for error reporting purposes.
    """

    def _detect_multiple_input_sets(inputs):
        """Check if any of inputs is a list, indicating a usage of MIS."""
        return any(isinstance(input, list) for input in inputs)

    def _safe_len(input):
        if isinstance(input, list):
            return len(input)
        else:
            return 1

    def _check_common_length(inputs):
        """Check if all list representing multiple input sets have the same length and return it"""
        arg_list_len = max(_safe_len(input) for input in inputs)
        for input in inputs:
            if isinstance(input, list):
                if len(input) != arg_list_len:
                    raise ValueError(
                        f"All argument lists for Multiple Input Sets used "
                        f"with operator `{op_name}` must have "
                        f"the same length"
                    )
        return arg_list_len

    def _unify_lists(inputs, arg_list_len):
        """Pack single _DataNodes into lists, so they are treated as Multiple Input Sets
        consistently with the ones already present

        Parameters
        ----------
        arg_list_len : int
            Number of MIS.
        """
        result = ()
        for input in inputs:
            if isinstance(input, list):
                result = result + (input,)
            else:
                result = result + ([input] * arg_list_len,)
        return result

    def _repack_input_sets(inputs):
        """Zip the list from [[arg0, arg0', arg0''], [arg1', arg1'', arg1''], ...]
        to [(arg0, arg1, ...), (arg0', arg1', ...), (arg0'', arg1'', ...)]
        """
        return _repack_list(inputs, tuple)

    input_sets = []
    if _detect_multiple_input_sets(inputs):
        arg_list_len = _check_common_length(inputs)
        packed_inputs = _unify_lists(inputs, arg_list_len)
        input_sets = _repack_input_sets(packed_inputs)
    else:
        input_sets = [inputs]

    return input_sets
