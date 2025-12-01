// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/builtin/external_source.h"
#include <functional>
#include <list>

namespace dali {

template<>
void ExternalSource<CPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  auto &thread_pool = ws.GetThreadPool();
  ForwardCurrentData(output, data_id_, thread_pool);
  output.SetLayout(layout_);
  SetDepletedOperatorTrace(ws, !(repeats_last_ || HasDataInQueue()));
}


template<>
void ExternalSource<GPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<GPUBackend>(0);
  cudaStream_t stream_used = ws.has_stream() ? ws.stream() : 0;
  ForwardCurrentData(output, data_id_, stream_used);
  output.SetLayout(layout_);
  SetDepletedOperatorTrace(ws, !(repeats_last_ || HasDataInQueue()));
}


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
                .AddOptionalTypeArg("dtype", R"code(Input data type.

The operator will validate that the fetched data is of the provided type.
If the argument is omitted or ``DALIDataType.NO_TYPE`` is passed, the operator will infer
the type based on the provided data.

This argument will be required starting from DALI 2.0.)code")
                .AddOptionalArg<int>("ndim", R"code(Number of dimensions in the input.

The dimensionality of the data provided to the operator will be verified against this value.
Number of dimensions can be also inferred from the ``layout`` argument if provided.

If the ``layout`` argument is provided, the `ndim` must match the number
of dimensions in the layout.

Specifying the input dimensionality will be required starting from DALI 2.0)code", nullptr)
                .AddOptionalArg<TensorLayout>("layout",
                                              "If provided, sets the layout of the data.", nullptr)
                .AddOptionalArg("repeat_last", R"(If set, the last batch is re-fed when running
the operator and no new data was provided since the previous run.)", false)
                .AddParent("InputOperatorBase");


DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<GPUBackend>, GPU);

}  // namespace dali
