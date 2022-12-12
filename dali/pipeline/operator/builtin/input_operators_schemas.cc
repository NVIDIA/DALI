// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/op_schema.h"

namespace dali {

/*
 * This file defines the schemas for InputOperator and everything that
 * derives from it.
 *
 * The purpose of putting them in one file is to prevent SIOF.
 * On the event that they are in separate compilation units, the InputOperatorBase
 * schema might be registered after the remaining schemas. In that case, the particular
 * input operator (e.g. ExternalSource) calls `AddParent` on a schema, that's not
 * registered yet. This resolves in runtime error when importing DALI.
 */


DALI_SCHEMA(InputOperatorBase)
                .DocStr(R"doc(
A base for any operator that forwards in-memory data to DALI pipeline.)doc")
                .NumInput(0)
                .NumOutput(0)
                .AddOptionalArg("blocking", R"code(
If ``True``, this operator will block until the data is available (e.g. by calling ``feed_input``).
If ``False``, the operator will raise an error, if the data is not available.
)code", true)
                .AddOptionalArg("no_copy", R"code(
Determines whether DALI should copy the buffer when ``feed_input`` is called.

If set to True, DALI passes the user's memory directly to the pipeline, instead of copying it.
It is the user's responsibility to keep the buffer alive and unmodified until it is
consumed by the pipeline.

The buffer can be modified or freed again after the outputs of the relevant iterations
have been consumed. Effectively, it happens after ``prefetch_queue_depth`` or
``cpu_queue_depth * gpu_queue_depth`` (when they are not equal) iterations following
the ``feed_input`` call.

The memory location must match the specified ``device`` parameter of the operator.
For the CPU, the provided memory can be one contiguous buffer or a list of contiguous Tensors.
For the GPU, to avoid extra copy, the provided buffer must be contiguous. If you provide a list
of separate Tensors, there will be an additional copy made internally, consuming both memory
and bandwidth.
)code", false);


// This schema is partially internal. We want it to be listed int the supported_ops,
// but it is explicitly not loaded by the Op Factory. Instead the Python wrapper classes
// access it directly.
// C++ operators should access this operator directly as well.
DALI_SCHEMA(ExternalSource)
                .DocStr(R"code(Allows externally provided data to be passed as an input to the pipeline.

  This is a backend for `ExternalSource` operator. For Python functionality, refer to
  nvidia.dali.fn.external_source operator documentation.

  This operator can be used with C and C++ APIs by either directly specifying it with OpSpec
  or by the Pipeline::AddExternalInput method.)code")
                .NumInput(0)
                .NumOutput(1)
                .AddParent("InputOperatorBase")
                .AddOptionalTypeArg("dtype", R"code(Input data type.

The operator will validate that the fetched data is of the provided type.
If the argument is omitted or ``DALIDataType.NO_TYPE`` is passed, the operator will infer
the type based on the provided data.

This argument will be required starting from DALI 2.0.)code")
                .AddOptionalArg<int>("ndim", R"code(Number of dimensions in the input.

The dimensionality of the data provided to the operator will be verified against this value.
Number of dimensions can be also inferred from the ``layout`` argument if provided.

If the ``layout`` argument is provided, the ``ndim`` must match the number
of dimensions in the layout.

Specifying the input dimensionality will be required starting from DALI 2.0)code", nullptr)
                .AddOptionalArg<TensorLayout>("layout",
                                              "If provided, sets the layout of the data.", nullptr);

}  // namespace dali
