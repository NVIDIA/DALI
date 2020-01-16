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

#include "dali/operators/math/normalize/normalize.h"
#include "dali/kernels/reduce/reduce.h"
#include "dali/kernels/normalize/normalize_cpu.h"
#include "dali/core/static_switch.h"

namespace dali {

DALI_SCHEMA(Normalize)
  .DocStr(R"(Normalizes the input by removing mean and dividing by standard deviation.

The mean and standard deviation can be calculated internally for specified subset of axes or
can be externally provided as *mean* and *stddev* arguments.)")
  .NumInput(1)
  .AddOptionalArg("batch", "If True, the mean and standard deviation are calculated across tensors "
    "in the batch. This also requires that the input sample shapes in the non-averaged axes match.",
    false)
  .AddOptionalArg<float>("mean", "Mean value to subtract from the data. It can be either a scalar "
    "or a batch of tensors with same dimesnionality as the input and the shape in each dimesnion "
    "must either match that of the input or be equal to 1 (in which case the value will be "
    "broadcast in this dimension). If not specified, the mean is calculated from the input.",
    0.0f, true)
  .AddOptionalArg<float>("stddev", "Stanrad deviation value to scale the data. For shape "
    "constraints, see *mean* argument. If not specified, the mean is calculated from the input.",
    0.0f, true)
  .AddOptionalArg<int>("axes", "Indices if axes along which the input is normalized. By default, "
    "all axes are used. Axes can also be specified by name, see *axes_names*.", -1, false)
  .AddOptionalArg<TensorLayout>("axis_names", "Names of the axes in the input - axis indices "
    "are taken from the input layout. This argument cannot be used together with *axes*.", "")
  .AddOptionalArg("shift", "The value to which the mean will map in the output. Useful for "
    "unsigned output types.", 0.0f, false)
  .AddOptionalArg("scale", "The scaling factor applied to the output. Useful for integral "
    "output types", 1.0f, false)
  .AddOptionalArg("dtype", "Output type. When using integral types, use *shift* and *scale* to "
    "improve usage of the output type dynamic range. If dtype is an integral type, out of range "
    "values are clamped, and non-integer values are rounded to nearest integer.", DALI_FLOAT);

DALI_REGISTER_OPERATOR(Normalize, Normalize<CPUBackend>, CPU);

#define NormTypes (int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float)

template <>
void Normalize<CPUBackend>::SetupBackend(const HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  int nsamples = input.size();
  int nthreads = ws.GetThreadPool().size();

  const TypeInfo &Float = TypeTable::GetTypeInfo(DALI_FLOAT);
  mean_.Resize(param_shape_);
  mean_.set_type(Float);
  inv_stddev_.Resize(param_shape_);
  inv_stddev_.set_type(Float);

  TYPE_SWITCH(input_type_, type2id, InputType, NormTypes, (
    TYPE_SWITCH(output_type_, type2id, OutputType, NormTypes, (
      using Kernel = kernels::NormalizeCPU<OutputType, InputType, float>;
      kmgr_.Resize<Kernel>(nthreads, nsamples);
    ), (DALI_FAIL("Normalize: unsupported output type")))  // NOLINT
  ), (DALI_FAIL("Normalize: unsupported input type")));   // NOLINT
}

template <>
void Normalize<CPUBackend>::RunImpl(HostWorkspace &ws) {
  ThreadPool &TP = ws.GetThreadPool();

  TP.WaitForWork();
}


}  // namespace dali
