// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <tuple>
#include <vector>
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operators/crop/slice_base.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/util/half.hpp"
#include "dali/core/static_switch.h"

namespace dali {
namespace detail {

template <typename OutputType, typename InputType>
void RunHelper(Tensor<CPUBackend> &output,
               const Tensor<CPUBackend> &input,
               const std::vector<int64_t> &slice_anchor,
               const std::vector<int64_t> &slice_shape) {
  std::size_t number_of_dims = input.shape().size();
  VALUE_SWITCH(number_of_dims, NumDims, (3, 4), (
    kernels::KernelContext ctx;
    auto in_view = view<const InputType, NumDims>(input);

    kernels::SliceArgs<NumDims> slice_args;
    auto &anchor = slice_args.anchor;
    auto &shape = slice_args.shape;
    for (std::size_t d = 0; d < NumDims; d++) {
      anchor[d] = slice_anchor[d];
      shape[d] = slice_shape[d];
    }

    kernels::SliceCPU<OutputType, InputType, NumDims> kernel;
    kernels::KernelRequirements req = kernel.Setup(ctx, in_view, slice_args);

    output.set_type(TypeInfo::Create<OutputType>());
    output.SetLayout(input.GetLayout());
    output.Resize(req.output_shapes[0][0].shape.to_vector());

    auto out_view = view<OutputType, NumDims>(output);
    kernel.Run(ctx, out_view, in_view, slice_args);
  ), // NOLINT
  (
    DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims));
  )); // NOLINT
}

}  // namespace detail

DALI_SCHEMA(SliceBase)
    .DocStr(R"code(Base implementation for `Slice`, `Crop` and related operators)code")
    .MakeInternal()
    .AddOptionalArg("output_dtype",
      R"code(Output data type. By default same data type as the input will be used)code",
      DALI_NO_TYPE);

template <>
void SliceBase<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  this->DataDependentSetup(ws, idx);
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);
  auto data_idx = ws->data_idx();

  if (input_type_ == DALI_FLOAT16 || output_type_ == DALI_FLOAT16) {
    DALI_ENFORCE(input_type_ == output_type_,
      "type conversion is not supported for half precision floats");
    detail::RunHelper<float16_cpu, float16_cpu>(
      output, input, slice_anchors_[data_idx], slice_shapes_[data_idx]);
    return;
  }

  DALI_TYPE_SWITCH(input_type_, InputType,
    DALI_TYPE_SWITCH(output_type_, OutputType,
      detail::RunHelper<OutputType, InputType>(
        output, input, slice_anchors_[data_idx], slice_shapes_[data_idx]);
    )
  )
}

}  // namespace dali
