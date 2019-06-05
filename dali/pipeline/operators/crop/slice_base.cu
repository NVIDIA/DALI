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
#include "dali/pipeline/operators/crop/slice_base.h"
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
    std::vector<Dims> out_shapes;
    to_dims_vec(out_shapes, req.output_shapes[0]);

    output.set_type(TypeInfo::Create<OutputType>());
    output.SetLayout(input.GetLayout());
    output.Resize(out_shapes);

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
void SliceBase<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  this->DataDependentSetup(ws, idx);
  const auto &input = ws->Input<GPUBackend>(idx);
  auto &output = ws->Output<GPUBackend>(idx);

  if (input_type_ == DALI_FLOAT16 || output_type_ == DALI_FLOAT16) {
    DALI_ENFORCE(input_type_ == output_type_,
      "type conversion is not supported for half precision floats");
    detail::RunHelper<float16, float16>(
      output, input, slice_anchors_, slice_shapes_, ws->stream(), scratch_alloc_);
    return;
  }

  DALI_TYPE_SWITCH(input_type_, InputType,
    DALI_TYPE_SWITCH(output_type_, OutputType,
      detail::RunHelper<OutputType, InputType>(
        output, input, slice_anchors_, slice_shapes_, ws->stream(), scratch_alloc_);
    )
  )
}

}  // namespace dali
