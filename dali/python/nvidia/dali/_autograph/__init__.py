# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Conversion of eager-style Python into user-customized graph code based on
TensorFlow conversion.

AutoGraph transforms a subset of Python which operates on user-defined objects
into equivalent user-defined graph code. When executing the graph, it has the same
effect as if you ran the original code in eager mode. This AutoGraph fork introduces
customization points for the detection of user-defined objects and operator overloads.
The customization point can be controlled by inheriting from OperatorBase and passing
it to the initialize_autograph function.
Python code which doesn't operate on user-defined objects remains functionally
unchanged, but keep in mind that AutoGraph only executes such code at trace
time, and generally will not be consistent with eager execution.

For more information, see the
[AutoGraph reference documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
python/autograph/g3doc/reference/index.md),
and the [tf.function guide](https://www.tensorflow.org/guide/function#autograph_transformations).
"""

# TODO(mdan): Bring only the relevant symbols to the top level.
from nvidia.dali._autograph import operators
from nvidia.dali._autograph import utils
from nvidia.dali._autograph.core.converter import ConversionOptions
from nvidia.dali._autograph.core.converter import Feature
from nvidia.dali._autograph.impl.api import initialize_autograph
from nvidia.dali._autograph.impl.api import AutoGraphError
from nvidia.dali._autograph.impl.api import convert
from nvidia.dali._autograph.impl.api import converted_call
from nvidia.dali._autograph.impl.api import do_not_convert
# from nvidia.dali._autograph.impl.api import StackTraceMapper
from nvidia.dali._autograph.impl.api import to_code
from nvidia.dali._autograph.impl.api import to_graph
from nvidia.dali._autograph.lang.directives import set_element_type
from nvidia.dali._autograph.lang.directives import set_loop_options
from nvidia.dali._autograph.utils import ag_logging
from nvidia.dali._autograph.utils.all_utils import _remove_undocumented
from nvidia.dali._autograph.utils.hooks import OperatorBase

# TODO(mdan): Revisit this list once we finalize the generated code mechanism.
_allowed_symbols = [
    # Main API
    'AutoGraphError',
    'ConversionOptions',
    'Feature',
    # 'StackTraceMapper',
    'convert',
    'converted_call',
    'do_not_convert',
    'to_code',
    'to_graph',
    # Overloaded operators
    'operators',
    # Python language "extensions"
    'set_element_type',
    'set_loop_options',
    'stack',
    'tensor_list',
    # Utilities: to be removed
    'utils',
]


_remove_undocumented(__name__, _allowed_symbols)
