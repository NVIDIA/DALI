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

#include <utility>
#include <vector>
#include "dali/pipeline/operators/fused/crop_mirror_normalize.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_gpu.cuh"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

namespace detail {

template <typename OutputType, typename InputType>
void RunHelper(TensorList<GPUBackend> &output,
               const TensorList<GPUBackend> &input,
               const std::vector<std::vector<int64_t>>& slice_anchors,
               const std::vector<std::vector<int64_t>>& slice_shapes,
               const std::vector<int> &horizontal_flip,
               bool pad_output,
               const std::vector<float> &mean,
               const std::vector<float> &inv_std_dev,
               DALITensorLayout input_layout,
               DALITensorLayout output_layout,
               cudaStream_t stream,
               kernels::ScratchpadAllocator &scratch_alloc) {
  std::size_t number_of_dims = input.tensor_shape(0).size();
  VALUE_SWITCH(number_of_dims, NumDims, (3, 4), (
    kernels::SliceFlipNormalizePermuteGPU<OutputType, InputType, NumDims> kernel;
    kernels::KernelContext ctx;
    ctx.gpu.stream = stream;
    auto in_view = view<const InputType, NumDims>(input);

    std::vector<kernels::SliceFlipNormalizePermuteArgs<NumDims>> per_sample_args;
    per_sample_args.reserve(slice_anchors.size());
    for (std::size_t i = 0; i < slice_anchors.size(); i++) {
      per_sample_args.emplace_back(slice_shapes[i]);
      auto &args = per_sample_args[i];
      const auto& slice_anchor = slice_anchors[i];
      for (std::size_t d = 0; d < NumDims; d++) {
        args.anchor[d] = slice_anchor[d];
      }

      if (horizontal_flip[i]) {
        args.flip[horizontal_dim_idx(input_layout)] = true;
      }

      if (pad_output) {
        args.padded_shape[channels_dim(input_layout)] = 4;
      }

      if (input_layout != output_layout) {
        args.permuted_dims = permuted_dims<NumDims>(input_layout, output_layout);
      }

      const bool should_normalize =
         !std::all_of(mean.begin(), mean.end(), [](float x){ return x == 0.0f; })
      || !std::all_of(inv_std_dev.begin(), inv_std_dev.end(), [](float x){ return x == 1.0f; });
      if (should_normalize) {
        args.mean = mean;
        args.inv_stddev = inv_std_dev;
        args.normalization_dim = channels_dim(input_layout);
      }
    }

    kernels::KernelRequirements req = kernel.Setup(ctx, in_view, per_sample_args);
    std::vector<Dims> out_shapes;
    to_dims_vec(out_shapes, req.output_shapes[0]);

    output.set_type(TypeInfo::Create<OutputType>());
    output.SetLayout(input_layout);
    output.Resize(out_shapes);

    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx.scratchpad = &scratchpad;

    auto out_view = view<OutputType, NumDims>(output);
    kernel.Run(ctx, out_view, in_view, per_sample_args);
  ),  // NOLINT
  (
    DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims));
  ));  // NOLINT
}

}  // namespace detail

template <>
void CropMirrorNormalize<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  for (int sample_idx = 0; sample_idx < batch_size_; sample_idx++) {
    SetupSample(sample_idx, input_layout_, input.tensor_shape(sample_idx));
    mirror_[sample_idx] = spec_.GetArgument<int>("mirror", ws, sample_idx);
  }
  auto &output = ws->Output<GPUBackend>(idx);
  output.SetLayout(output_layout_);
}

template<>
void CropMirrorNormalize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  this->DataDependentSetup(ws, idx);
  const auto &input = ws->Input<GPUBackend>(idx);
  auto &output = ws->Output<GPUBackend>(idx);

  DALI_TYPE_SWITCH_WITH_FP16_GPU(input_type_, InputType,
    DALI_TYPE_SWITCH_WITH_FP16_GPU(output_type_, OutputType,
      detail::RunHelper<OutputType, InputType>(
        output, input, slice_anchors_, slice_shapes_,
        mirror_, pad_output_, mean_vec_, inv_std_vec_,
        input_layout_, output_layout_,
        ws->stream(), scratch_alloc_);
    )
  )
}

DALI_REGISTER_OPERATOR(CropMirrorNormalize, CropMirrorNormalize<GPUBackend>, GPU);

}  // namespace dali
