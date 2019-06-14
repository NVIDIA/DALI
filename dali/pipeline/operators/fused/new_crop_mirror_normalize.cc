// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/fused/new_crop_mirror_normalize.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_cpu.h"
#include "dali/util/half.hpp"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(NewCropMirrorNormalize)
  .DocStr(R"code(Perform fused cropping, normalization, format conversion
(NHWC to NCHW) if desired, and type casting.
Normalization takes input image and produces output using formula:

  output = (input - mean) / std
)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
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

DALI_REGISTER_OPERATOR(NewCropMirrorNormalize, NewCropMirrorNormalize<CPUBackend>, CPU);

namespace detail {

size_t horizontal_dim_idx(DALITensorLayout layout) {
  switch (layout) {
    case DALI_NHWC:
      return 1;
    case DALI_NCHW:
      return 2;
    case DALI_NFHWC:
      return 2;
    case DALI_NFCHW:
      return 3;
    default:
      DALI_FAIL("not supported layout: " + std::to_string(layout));
  }
}

template <size_t Dims>
std::array<int64_t, Dims> permuted_dims(DALITensorLayout in_layout,
                                        DALITensorLayout out_layout) {
  std::array<int64_t, Dims> perm_dims;
  for (size_t d = 0; d < Dims; d++) {
    perm_dims[d] = d;
  }

  if (in_layout != out_layout) {
    if (in_layout == DALI_NHWC && out_layout == DALI_NCHW) {
      perm_dims[0] = 2;
      perm_dims[1] = 0;
      perm_dims[2] = 1;
    } else if (in_layout == DALI_NCHW && out_layout == DALI_NHWC) {
      perm_dims[0] = 1;
      perm_dims[1] = 2;
      perm_dims[2] = 0;
    } else if (in_layout == DALI_NFHWC && out_layout == DALI_NFCHW) {
      perm_dims[1] = 3;
      perm_dims[2] = 1;
      perm_dims[3] = 2;
    } else if (in_layout == DALI_NFCHW && out_layout == DALI_NFHWC) {
      perm_dims[1] = 2;
      perm_dims[2] = 3;
      perm_dims[3] = 1;
    } else {
      DALI_FAIL("layout conversion from " + std::to_string(in_layout) + " to "
        + std::to_string(out_layout) + " not supported");
    }
  }

  return perm_dims;
}

size_t channels_dim(DALITensorLayout in_layout) {
  switch (in_layout) {
    case DALI_NHWC:
      return 2;
    case DALI_NCHW:
      return 0;
    case DALI_NFHWC:
      return 3;
    case DALI_NFCHW:
      return 1;
    default:
      DALI_FAIL("not supported layout: " + std::to_string(in_layout));
  }
}

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
void NewCropMirrorNormalize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  input_layout_ = ws->Input<CPUBackend>(0).GetLayout();
  if (output_layout_ == DALI_SAME) {
    output_layout_ = input_layout_;
  }
  DALI_ENFORCE(input_layout_ == DALI_NHWC || input_layout_ == DALI_NCHW
            || input_layout_ == DALI_NFHWC || input_layout_ == DALI_NFCHW,
    "Unexpected data layout");

  input_type_ = ws->Input<CPUBackend>(0).type().id();
  if (output_type_ == DALI_NO_TYPE) {
    output_type_ = input_type_;
  }
  CropAttr::ProcessArguments(ws);

  mirror_[ws->data_idx()] = spec_.GetArgument<int>("mirror", ws, ws->data_idx());
}

template <>
void NewCropMirrorNormalize<CPUBackend>::DataDependentSetup(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  SetupSample(ws->data_idx(), input_layout_, input.shape());

  auto &output = ws->Output<CPUBackend>(idx);
  output.SetLayout(output_layout_);
}

template <>
void NewCropMirrorNormalize<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
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
