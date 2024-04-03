// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/operators/python_function/jax_function.h"
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include <utility>

namespace dali {

DALI_SCHEMA(_JaxFunction)
    .DocStr(R"code(Runs the `function` callback, passing in the specified arguments as JAX arrays.

The callback can return 0 or more outputs, all of which must be JAX arrays.

All the inputs and outputs must have the same device placement/sharding.
If the inputs/outputs are placed on GPU(s), the JAX and DALI internal streams
will be synchronized, there is no need to synchronize the launched JAX functions with the host.
)code")
    .NumInput(0, 256)
    .AddArg("function_id", R"code(Id of the python function)code", DALI_INT64)
    .AddOptionalArg("num_outputs", R"code(The number of outputs returned by the `function`.

Function can return no output, in that case the `num_outputs` must be set to 0.
If the `num_outputs` is 1 (the default), callback should return a single JAX array,
for `num_outputs` > 1, callback should return a tuple of JAX arrays.
)code",
                    1)
    .OutputFn([](const OpSpec &spec) { return spec.GetArgument<int>("num_outputs"); })
    .AddOptionalArg<std::vector<TensorLayout>>("output_layouts",
                                               R"code(The layouts of returned tensors.

It can be either a list of strings for all of `num_outputs` respective outputs or a single string
to be set to all of the outputs.

Please note, in DALI, the outermost batch extent is implicit, the layout should
take into account only the sample dimensions.

If the argument is not specified, the `function` has the same number of inputs and outputs and
the dimensionality of respective inputs and outputs is preserved, the layout will be propagated
from the input to the output.)code",
                                               nullptr)
    .Unserializable()
    .MakeInternal()
    .MakeDocHidden();


DALI_REGISTER_OPERATOR(_JaxFunction, JaxFunction<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(_JaxFunction, JaxFunction<GPUBackend>, GPU);

}  // namespace dali
