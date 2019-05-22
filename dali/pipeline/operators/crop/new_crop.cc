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

template <typename InputType, typename OutputType, std::size_t D>
void RunHelper(Tensor<CPUBackend> &output,
               const Tensor<CPUBackend> &input,
               const std::array<int64_t, D> &slice_anchor,
               const std::array<int64_t, D> &slice_shape) {
  kernels::KernelContext ctx;
  auto in_view = view<const InputType, D>(input);

  kernels::SliceArgs<D> slice_args = {slice_anchor, slice_shape};

  kernels::SliceCPU<OutputType, InputType, D> kernel;
  kernels::KernelRequirements kernel_req = kernel.Setup(ctx, in_view, slice_args);

  auto out_view = view<OutputType, D>(output);
  kernel.Run(ctx, out_view, in_view, slice_args);
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
  // TODO(janton) : support other layouts
  DALI_ENFORCE(in_layout == DALI_NHWC || in_layout == DALI_NCHW,
    "Unexpected data layout");
  DALITensorLayout out_layout = in_layout;

  auto data_idx = ws->data_idx();
  DataDependentSetup(data_idx, in_layout, input.shape());
  auto &slice_shape = slice_shapes_[data_idx];
  Dims output_shape = { slice_shape[0], slice_shape[1], slice_shape[2] };

  auto &output = ws->Output<CPUBackend>(idx);
  output.SetLayout(out_layout);
  output.Resize(output_shape);
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
    detail::RunHelper<float16_cpu, float16_cpu, 3>(
      output, input, slice_anchors_[data_idx], slice_shapes_[data_idx]);
    return;
  }

  DALI_TYPE_SWITCH(input_type_, InputType,
    DALI_TYPE_SWITCH(output_type_, OutputType,
      detail::RunHelper<OutputType, InputType, 3>(
        output, input, slice_anchors_[data_idx], slice_shapes_[data_idx]);
    )
  )
}

// Register operator
DALI_REGISTER_OPERATOR(NewCrop, NewCrop<CPUBackend>, CPU);

}  // namespace dali
