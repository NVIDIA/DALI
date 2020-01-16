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
#include "dali/operators/generic/slice/slice_base.h"
#include "dali/pipeline/data/views.h"

namespace dali {
namespace detail {

template <typename OutputType, typename InputType>
void RunHelper(TensorList<GPUBackend>& output,
               const TensorList<GPUBackend>& input,
               const std::vector<std::vector<int64_t>>& slice_anchors,
               const std::vector<std::vector<int64_t>>& slice_shapes,
               cudaStream_t stream,
               kernels::ScratchpadAllocator &scratch_alloc) {
  std::size_t number_of_dims = input.tensor_shape(0).size();
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

    kernels::KernelRequirements req = kernel.Setup(ctx, in_view, slice_args);

    output.set_type(TypeInfo::Create<OutputType>());
    output.Resize(req.output_shapes[0]);

    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx.scratchpad = &scratchpad;

    auto out_view = view<OutputType, NumDims>(output);
    kernel.Run(ctx, out_view, in_view, slice_args);
  ),  // NOLINT
  (
    DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims));
  ));  // NOLINT
}

}  // namespace detail


template <>
void SliceBase<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  this->DataDependentSetup(ws);
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  TYPE_SWITCH(input_type_, type2id, InputType, SLICE_TYPES, (
    if (input_type_ == output_type_) {
      detail::RunHelper<InputType, InputType>(
        output, input, slice_anchors_, slice_shapes_, ws.stream(), scratch_alloc_);
    } else {
      TYPE_SWITCH(output_type_, type2id, OutputType, (float, float16, uint8_t), (
        detail::RunHelper<OutputType, InputType>(
          output, input, slice_anchors_, slice_shapes_, ws.stream(), scratch_alloc_);
      ), DALI_FAIL(make_string("Not supported output type:", output_type_));); // NOLINT
    }
  ), DALI_FAIL(make_string("Not supported input type:", input_type_));); // NOLINT

  output.SetLayout(InputLayout(ws, 0));
}

}  // namespace dali
