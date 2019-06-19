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

#include "dali/pipeline/operators/fused/crop_mirror_normalize.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_cpu.h"
#include "dali/util/half.hpp"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(CropMirrorNormalize)
  .DocStr(R"code(Perform fused cropping, normalization, format conversion
(NHWC to NCHW) if desired, and type casting.
Normalization takes input image and produces output using formula:

  output = (input - mean) / std

Note that not providing any crop argument will result into mirroring and
normalization only.
)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AllowSequences()
  .AddOptionalArg("output_dtype",
    R"code(Output data type.)code", DALI_FLOAT)
  .AddOptionalArg("output_layout",
    R"code(Output tensor data layout)code", DALI_NCHW)
  .AddOptionalArg(
    "pad_output",
    R"code(Whether to pad the output to number of channels being multiple of 4.)code", false)
  .AddOptionalArg("mirror",
    R"code(Mask for horizontal flip.
- `0` - do not perform horizontal flip for this image
- `1` - perform horizontal flip for this image.
)code",
    0, true)
  .AddOptionalArg("mean",
    R"code(Mean pixel values for image normalization.)code",
    std::vector<float>{0.0f})
  .AddOptionalArg("std",
    R"code(Standard deviation values for image normalization.)code",
    std::vector<float>{1.0f})
  .AddParent("Crop");

DALI_REGISTER_OPERATOR(CropMirrorNormalize, CropMirrorNormalize<CPUBackend>, CPU);

namespace detail {

template <typename OutputType, typename InputType>
void RunHelper(Tensor<CPUBackend> &output,
               const Tensor<CPUBackend> &input,
               const std::vector<int64_t> &slice_anchor,
               const std::vector<int64_t> &slice_shape,
               bool horizontal_flip,
               bool pad_output,
               const std::vector<float> &mean,
               const std::vector<float> &inv_std_dev) {
  std::size_t number_of_dims = input.shape().size();
  auto input_layout = input.GetLayout();
  auto output_layout = output.GetLayout();
  VALUE_SWITCH(number_of_dims, Dims, (3, 4), (
    auto in_view = view<const InputType, Dims>(input);

    kernels::SliceFlipNormalizePermuteArgs<Dims> args(slice_shape);
    for (std::size_t d = 0; d < Dims; d++) {
      args.anchor[d] = slice_anchor[d];
    }

    if (pad_output) {
      args.padded_shape[channels_dim(input_layout)] = 4;
    }

    if (horizontal_flip) {
      args.flip[horizontal_dim_idx(input_layout)] = true;
    }

    // Check if permutation is needed
    if (input_layout != output_layout) {
      args.permuted_dims = permuted_dims<Dims>(input_layout, output_layout);
    }

    const bool should_normalize =
         !std::all_of(mean.begin(), mean.end(), [](float x){ return x == 0.0f; })
      || !std::all_of(inv_std_dev.begin(), inv_std_dev.end(), [](float x){ return x == 1.0f; });
    if (should_normalize) {
      args.mean = mean;
      args.inv_stddev = inv_std_dev;
      args.normalization_dim = channels_dim(input_layout);
    }

    kernels::SliceFlipNormalizePermuteCPU<OutputType, InputType, Dims> kernel;
    kernels::KernelContext ctx;
    kernels::KernelRequirements req = kernel.Setup(ctx, in_view, args);

    output.set_type(TypeInfo::Create<OutputType>());
    output.SetLayout(input.GetLayout());
    output.Resize(req.output_shapes[0][0].shape.to_vector());

    auto out_view = view<OutputType, Dims>(output);
    kernel.Run(ctx, out_view, in_view, args);
  ), // NOLINT
  (
    DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims));
  )); // NOLINT
}

}  // namespace detail

template <>
void CropMirrorNormalize<CPUBackend>::DataDependentSetup(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  SetupSample(ws->data_idx(), input_layout_, input.shape());
  mirror_[ws->data_idx()] = spec_.GetArgument<int>("mirror", ws, ws->data_idx());

  auto &output = ws->Output<CPUBackend>(idx);
  output.SetLayout(output_layout_);
}

template <>
void CropMirrorNormalize<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  this->DataDependentSetup(ws, idx);
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);
  auto data_idx = ws->data_idx();

  DALI_TYPE_SWITCH_WITH_FP16_CPU(input_type_, InputType,
    DALI_TYPE_SWITCH_WITH_FP16_CPU(output_type_, OutputType,
      detail::RunHelper<OutputType, InputType>(
        output, input, slice_anchors_[data_idx], slice_shapes_[data_idx],
        mirror_[data_idx], pad_output_, mean_vec_, inv_std_vec_);
    )
  )
}

}  // namespace dali
