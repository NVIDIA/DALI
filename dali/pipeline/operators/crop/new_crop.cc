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
#include "dali/pipeline/operators/crop/new_crop.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/util/half.hpp"
#include "dali/core/static_switch.h"

namespace dali {

DALI_SCHEMA(NewCrop)
    .DocStr(R"code(Crops image with a given window dimensions and window position (upper left corner). **Experimental** Use `Crop` instead)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg(
        "image_type",
        R"code(The color space of input and output image)code",
        DALI_RGB, false)
    .AddParent("CropAttr");

namespace detail {

template <typename InputType, typename OutputType>
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
    output.Resize(req.output_shapes[0][0].shape);

    auto out_view = view<OutputType, NumDims>(output);
    kernel.Run(ctx, out_view, in_view, slice_args);
  ), // NOLINT
  (
    DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims));
  )); // NOLINT
}

}  // namespace detail

template <>
void NewCrop<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  CropAttr::ProcessArguments(ws);
  const auto &input = ws->Input<CPUBackend>(0);
  input_type_ = input.type().id();
  if (output_type_ == DALI_NO_TYPE)
    output_type_ = input_type_;
}

template <>
void NewCrop<CPUBackend>::DataDependentSetup(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);

  const DALITensorLayout in_layout = input.GetLayout();
  DALI_ENFORCE(in_layout == DALI_NHWC || in_layout == DALI_NCHW
            || in_layout == DALI_NFHWC || in_layout == DALI_NFCHW,
    "Unexpected data layout");
  DALITensorLayout out_layout = in_layout;

  auto data_idx = ws->data_idx();
  DataDependentSetup(data_idx, in_layout, input.shape());
  auto &slice_shape = slice_shapes_[data_idx];

  auto &output = ws->Output<CPUBackend>(idx);
  output.SetLayout(out_layout);
}

template <>
void NewCrop<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  DataDependentSetup(ws, idx);
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

// Register operator
DALI_REGISTER_OPERATOR(NewCrop, NewCrop<CPUBackend>, CPU);

}  // namespace dali
