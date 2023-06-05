# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""This module contains the implementation of DALI if statement.

It initializes AutoGraph with the DaliOperatorOverload that provides the overload for the if_stmt
and adjust the filtered modules so DALI code is not converted.

The if_stmt provides access to both branches as callables and the set_state/get_state functions
that allows to capture and adjust all symbols modified within those branches. This allows to
checkpoint the state and visit the code of both branches.

if_stmt highlights which state variables are considered the outputs of the if/else pair - we can
use the state captured after visiting if and else branches and produce fn._conditional.merge
nodes for all of them.

When visiting the if/else scopes, we are tracking tha path that we took and the predicates that
were used via the _ConditionStack. As it is not easy to detect which state variables would be
consumed as inputs to DALI operators, we inject additional code to the operator function.
Every time a DataNode is consumed, we look up in which scope it was produced and travel the
path from that point to the current scope in the _ConditionStack, applying necessary splits.
All the return values are registered to the current scope for further lookups.
"""

import warnings


def check_nesting_support():
    """Function to check if nesting conditionals is supported in current environment.

    Note:
        This is due to lack of support for dm-tree package in Python 3.6. When Python 3.6
        support is dropped this can be removed.

    Returns:
        _bool: Is nesting conditionals supported.
    """
    try:
        import tree        # noqa: F401
        return True
    except Exception:
        warnings.warn(
            "Nesting conditionals requires Python 3.7+ and dm-tree package present in the system.")
        return False


if check_nesting_support():
    # This is needed to forward private definitions
    from nvidia.dali._conditionals_impl import *        # noqa: F401, F403
    from nvidia.dali._conditionals_impl import _autograph        # noqa: F401
    from nvidia.dali._conditionals_impl import _DataNode        # noqa: F401
    from nvidia.dali._conditionals_impl import _Branch        # noqa: F401
    from nvidia.dali._conditionals_impl import _StackEntry        # noqa: F401
    from nvidia.dali._conditionals_impl import _ConditionStack        # noqa: F401
    from nvidia.dali._conditionals_impl import _cond_manager        # noqa: F401
    from nvidia.dali._conditionals_impl import _cond_true        # noqa: F401
    from nvidia.dali._conditionals_impl import _cond_false        # noqa: F401
    from nvidia.dali._conditionals_impl import _cond_merge        # noqa: F401
    from nvidia.dali._conditionals_impl import _verify_branch_outputs        # noqa: F401
    from nvidia.dali._conditionals_impl import _OVERLOADS        # noqa: F401
    from nvidia.dali._conditionals_impl import _data_node_repr        # noqa: F401
else:
    # This is needed to forward private definitions
    from nvidia.dali._conditionals_impl_legacy import *        # noqa: F401, F403
    from nvidia.dali._conditionals_impl_legacy import _autograph        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _DataNode        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _Branch        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _StackEntry        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _ConditionStack        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _cond_manager        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _cond_true        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _cond_false        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _cond_merge        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _verify_branch_outputs        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _OVERLOADS        # noqa: F401
    from nvidia.dali._conditionals_impl_legacy import _data_node_repr        # noqa: F401
