# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""This module implements operators that AutoGraph overloads.

Note that "operator" is used loosely here, and includes control structures like
conditionals and loops, implemented in functional form, using for example
closures for the body.
"""

# Naming conventions:
#  * operator names match the name usually used for the respective Python
#    idiom; examples: for_stmt, list_append
#  * operator arguments match either of:
#    - the corresponding Python AST attribute (e.g. the condition of an if
#      statement is called test) if the operator represents an AST construct
#    - the names used in the Python docs, if the operator is a function (e.g.
#      list_ and x for append, see
#      https://docs.python.org/3.7/tutorial/datastructures.html)
#
# All operators may accept a final argument named "opts", of a type that
# subclasses namedtuple and contains any arguments that are only required
# for some specializations of the operator.

from autograph.operators.conditional_expressions import if_exp
from autograph.operators.control_flow import for_stmt
from autograph.operators.control_flow import if_stmt
from autograph.operators.control_flow import while_stmt
from autograph.operators.data_structures import list_append
from autograph.operators.data_structures import list_pop
from autograph.operators.data_structures import list_stack
from autograph.operators.data_structures import ListPopOpts
from autograph.operators.data_structures import ListStackOpts
from autograph.operators.data_structures import new_list
from autograph.operators.exceptions import assert_stmt
from autograph.operators.logical import and_
from autograph.operators.logical import eq
from autograph.operators.logical import not_
from autograph.operators.logical import not_eq
from autograph.operators.logical import or_
from autograph.operators.py_builtins import float_
from autograph.operators.py_builtins import int_
from autograph.operators.py_builtins import len_
from autograph.operators.py_builtins import print_
from autograph.operators.py_builtins import range_
from autograph.operators.slices import get_item
from autograph.operators.slices import GetItemOpts
from autograph.operators.slices import set_item
from autograph.operators.variables import ld
from autograph.operators.variables import ldu
from autograph.operators.variables import Undefined
from autograph.operators.variables import UndefinedReturnValue
