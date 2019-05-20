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

#include <vector>
#include "dali/image/transform.h"
#include "dali/kernels/slice/slice_gpu.cuh"
#include "dali/core/static_switch.h"
#include "dali/pipeline/operators/crop/new_crop.h"
#include "dali/pipeline/data/views.h"

namespace dali {

namespace detail {

template <typename InputType, typename OutputType>
void RunHelper(TensorList<GPUBackend>& output,
               const TensorList<GPUBackend>& input,
               const std::vector<std::vector<int64_t>>& slice_anchors,
               const std::vector<std::vector<int64_t>>& slice_shapes,
               cudaStream_t stream) {
  std::size_t number_of_dims = input.shape().size();
  VALUE_SWITCH(number_of_dims, NumDims, (3, 4), (
    kernels::SliceGPU<OutputType, InputType, NumDims> kernel;

    kernels::KernelContext ctx;
    ctx.gpu.stream = stream;
    auto in_view = view<const InputType, NumDims>(input);

    std::vector<kernels::SliceArgs<NumDims>> slice_args;
    slice_args.reserve(slice_anchors.size());
    for (std::size_t i = 0; i < slice_anchors.size(); i++) {
      std::array<int64_t, NumDims> anchor, shape;
      const auto& slice_anchor = slice_anchors[i];
      const auto& slice_shape = slice_shapes[i];
      for (std::size_t d = 0; d < NumDims; d++) {
        anchor[d] = slice_anchor[d];
        shape[d] = slice_shape[d];
      }
      slice_args.push_back({anchor, shape});
    }

    kernels::KernelRequirements kernel_req = kernel.Setup(ctx, in_view, slice_args);

    auto out_view = view<OutputType, NumDims>(output);
    kernel.Run(ctx, out_view, in_view, slice_args);
  ),  // NOLINT
  (
    DALI_FAIL("Not supported number of dimensions");
  ));  // NOLINT
}

}  // namespace detail

template <>
void NewCrop<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  CropAttr::ProcessArguments(ws);
  const auto &input = ws->Input<GPUBackend>(0);
  input_type_ = input.type().id();
  if (output_type_ == DALI_NO_TYPE)
    output_type_ = input_type_;
}

template <>
void NewCrop<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);

  const DALITensorLayout in_layout = input.GetLayout();
  // TODO(janton) : support other layouts
  DALI_ENFORCE(in_layout == DALI_NHWC || in_layout == DALI_NCHW
            || in_layout == DALI_NFHWC || in_layout == DALI_NFCHW,
    "Unexpected data layout");
  DALITensorLayout out_layout = in_layout;

  std::vector<Dims> output_shape(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    DataDependentSetup(i, in_layout, input.tensor_shape(i));
    auto &slice_shape = slice_shapes_[i];
    if (in_layout == DALI_NFHWC || in_layout == DALI_NFCHW) {
      output_shape[i] = { slice_shape[0], slice_shape[1], slice_shape[2], slice_shape[3] };
    } else {
      output_shape[i] = { slice_shape[0], slice_shape[1], slice_shape[2] };
    }
  }
  auto &output = ws->Output<GPUBackend>(idx);
  output.Resize(output_shape);
  output.SetLayout(out_layout);
}

template <>
void NewCrop<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  DataDependentSetup(ws, idx);
  const auto &input = ws->Input<GPUBackend>(idx);
  auto &output = ws->Output<GPUBackend>(idx);

  if (input_type_ == DALI_FLOAT16 || output_type_ == DALI_FLOAT16) {
    DALI_ENFORCE(input_type_ == output_type_,
      "type conversion is not supported for half precision floats");
    detail::RunHelper<float16, float16>(
      output, input, slice_anchors_, slice_shapes_, ws->stream());
    return;
  }

  DALI_TYPE_SWITCH(input_type_, InputType,
    DALI_TYPE_SWITCH(output_type_, OutputType,
      detail::RunHelper<OutputType, InputType>(
        output, input, slice_anchors_, slice_shapes_, ws->stream());
    )
  )
}

// Register operator
DALI_REGISTER_OPERATOR(NewCrop, NewCrop<GPUBackend>, GPU);

}  // namespace dali
