// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/generic/reduce/reduce.h"
#include "dali/operators/generic/reduce/sum.h"
#include "dali/operators/generic/reduce/mean.h"
#include "dali/operators/generic/reduce/root_mean_square.h"
#include "dali/operators/generic/reduce/mean_square.h"
#include "dali/operators/generic/reduce/reduce_with_mean_input.h"


namespace dali {

DALI_SCHEMA(ReduceBase)
  .AddOptionalArg<std::vector<int>>(
    "axes",
    R"code(Axis or axes along which reduction is performed.

Not providing any axis results in reduction of all elements.)code",
   nullptr)
  .AddOptionalArg<TensorLayout>("axis_names", R"code(Name(s) of the axis or axes along which the reduction is performed.

The input layout is used to translate the axis names to axis indices, for example ``axis_names="HW"`` with input
layout `"FHWC"` is equivalent to specifying ``axes=[1,2]``. This argument cannot be used together with ``axes``.)code",
    nullptr)
  .AddOptionalArg(
    "keep_dims",
    "If True, maintains original input dimensions.",
    false);

DALI_SCHEMA(ReduceWithOutputType)
  .AddOptionalArg("dtype",
    R"code(Output data type. This type is used to accumulate the result.)code",
    DALI_NO_TYPE)
  .AddParent("ReduceBase");

  DALI_SCHEMA(ReduceWithMeanInput)
  .AddOptionalArg("ddof",
    R"code(Delat Degrees of Freedom. Adjusts the divisor used in calculations, which is `N - ddof`.)code",
    0)
  .AddParent("ReduceBase");

DALI_SCHEMA(reductions__StdDev)
  .DocStr("Gets standard deviation of elements along provided axes.")
  .NumInput(2)
  .NumOutput(1)
  .AddParent("ReduceWithMeanInput");

DALI_SCHEMA(reductions__Variance)
  .DocStr("Gets variance of elements along provided axes.")
  .NumInput(2)
  .NumOutput(1)
  .AddParent("ReduceWithMeanInput");

DALI_SCHEMA(reductions__Mean)
  .DocStr("Gets mean of elements along provided axes.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceWithOutputType");

DALI_SCHEMA(reductions__MeanSquare)
  .DocStr("Gets mean square of elements along provided axes.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceWithOutputType");

DALI_SCHEMA(reductions__RMS)
  .DocStr("Gets root mean square of elements along provided axes.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceWithOutputType");

DALI_SCHEMA(reductions__Sum)
  .DocStr("Gets sum of elements along provided axes.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceWithOutputType");

DALI_SCHEMA(reductions__Min)
  .DocStr("Gets minimal input element along provided axes.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceBase");

DALI_SCHEMA(reductions__Max)
  .DocStr("Gets maximal input element along provided axes.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceBase");

using MeanCPU = MeanOp<kernels::MeanCPU, CPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Mean, MeanCPU, CPU);

using MeanGPU = MeanOp<kernels::MeanGPU, GPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Mean, MeanGPU, GPU);

using MeanSquareCPU = MeanSquareOp<kernels::MeanSquareCPU, CPUBackend>;
DALI_REGISTER_OPERATOR(reductions__MeanSquare, MeanSquareCPU, CPU);

using MeanSquareGPU = MeanSquareOp<kernels::MeanSquareGPU, GPUBackend>;
DALI_REGISTER_OPERATOR(reductions__MeanSquare, MeanSquareGPU, GPU);

using RootMeanSquareCPU = RootMeanSquareOp<kernels::RootMeanSquareCPU, CPUBackend>;
DALI_REGISTER_OPERATOR(reductions__RMS, RootMeanSquareCPU, CPU);

using RootMeanSquareGPU = RootMeanSquareOp<kernels::RootMeanSquareGPU, GPUBackend>;
DALI_REGISTER_OPERATOR(reductions__RMS, RootMeanSquareGPU, GPU);

using SumCPU = SumOp<kernels::SumCPU, CPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Sum, SumCPU, CPU);

using SumGPU = SumOp<kernels::SumGPU, GPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Sum, SumGPU, GPU);

using MinCPU = ReduceOp<kernels::MinCPU, CPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Min, MinCPU, CPU);

using MinGPU = ReduceOp<kernels::MinGPU, GPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Min, MinGPU, GPU);

using MaxCPU = ReduceOp<kernels::MaxCPU, CPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Max, MaxCPU, CPU);

using MaxGPU = ReduceOp<kernels::MaxGPU, GPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Max, MaxGPU, GPU);

using StdCPU = ReduceWithMeanInput<kernels::StdDevCPU, CPUBackend>;
DALI_REGISTER_OPERATOR(reductions__StdDev, StdCPU, CPU);

using StdGPU = ReduceWithMeanInput<kernels::StdDevGPU, GPUBackend>;
DALI_REGISTER_OPERATOR(reductions__StdDev, StdGPU, GPU);

using VarianceCPU = ReduceWithMeanInput<kernels::VarianceCPU, CPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Variance, VarianceCPU, CPU);

using VarianceGPU = ReduceWithMeanInput<kernels::VarianceGPU, GPUBackend>;
DALI_REGISTER_OPERATOR(reductions__Variance, VarianceGPU, GPU);

}  // namespace dali
