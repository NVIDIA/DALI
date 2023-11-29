# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Support for wrapping converted functions bodies with auxiliary logic."""

from nvidia.dali._autograph.core import ag_ctx
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.operators import variables


# TODO(mdan): Move this into operators - it represents a function definition.


class FunctionScope(object):
    """Context manager that wraps the body of a converted function.

    This context manager handles various operations related to the scope of a
    function:
      * optional TF name scopes - these name scopes match the name of the
          function, for easy visualization in tensorBoard;
      * optional automatic control dependencies - this adds the same mechanism
          for control dependencies that is used by `@tf.function`; it can be
          optionally enabled when using `tf.autograph.to_graph`;
      * tracking of autograph conversion state (whether it's enabled by the user,
          conversion options);
    """

    def __init__(self, function_name, scope_name, options):
        self.name = scope_name
        self.options = options

        if options.user_requested:
            self.autograph_ctx = ag_ctx.ControlStatusCtx(ag_ctx.Status.ENABLED, options)
        self.callopts = options.call_options()

    def _sanitize(self, name):
        """See https://www.tensorflow.org/api_docs/python/tf/Graph#name_scope."""
        # TensorFlow doesn't like leading underscores at the top level.
        if name and name.startswith("_"):
            name = "fn" + name
        return name

    def __enter__(self):
        if self.options.user_requested:
            self.autograph_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.options.user_requested:
            self.autograph_ctx.__exit__(exc_type, exc_val, exc_tb)

    def ret(self, value, did_return):
        """Marks a value as returned from the function guarded by the scope."""
        del did_return

        if isinstance(value, variables.UndefinedReturnValue):
            return None

        return value


def with_function_scope(thunk, scope_name, options):
    """Inline version of the FunctionScope context manager."""
    with FunctionScope("lambda_", scope_name, options) as scope:
        return thunk(scope)
