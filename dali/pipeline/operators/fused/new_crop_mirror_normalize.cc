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

namespace dali {

namespace detail {

template <typename OutputType, typename InputType>
void RunHelper(Tensor<CPUBackend> &output,
               const Tensor<CPUBackend> &input,
               const std::vector<int64_t> &slice_anchor,
               const std::vector<int64_t> &slice_shape,
               bool horizontal_flip,
               ) {
  std::size_t number_of_dims = input.shape().size();
  VALUE_SWITCH(number_of_dims, NumDims, (3, 4), (
    kernels::KernelContext ctx;
    auto in_view = view<const InputType, NumDims>(input);

    kernels::SliceFlipNormalizePermuteArgs<NumDims> args;
    auto &anchor = args.anchor;
    auto &shape = args.shape;
    for (std::size_t d = 0; d < NumDims; d++) {
      anchor[d] = slice_anchor[d];
      shape[d] = slice_shape[d];
    }

    if (horizontal_flip) {
      args.should_flip = true;
      size_t horizontal_dim = 0;
      switch (input_layout_) {
        case DALI_NHWC:
          horizontal_dim = 1;
          break;
        case DALI_NCHW:
          horizontal_dim = 2;
          break;
        case DALI_NFHWC:
          horizontal_dim = 2;
          break;
        case DALI_NFCHW:
          horizontal_dim = 3;
          break;
        default:
          DALI_FAIL("not supported layout: " + std::to_string(input_layout_));
      }
      args.flip[horizontal_dim] = true;
    }

    // Check if permutation is needed
    if (input_layout_ != output_layout_) {
      switch (input_layout_) {
        case DALI_NHWC:
          horizontal_dim = 1;
          break;
        case DALI_NCHW:
          horizontal_dim = 2;
          break;
        case DALI_NFHWC:
          horizontal_dim = 2;
          break;
        case DALI_NFCHW:
          horizontal_dim = 3;
          break;
        default:
          DALI_FAIL("not supported layout: " + std::to_string(input_layout_));
      }
    }

    kernels::SliceFlipNormalizePermuteCPU<OutputType, InputType, NumDims> kernel;
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
  .AddArg("mean",
    R"code(Mean pixel values for image normalization.)code", DALI_FLOAT_VEC)
  .AddArg("std",
    R"code(Standard deviation values for image normalization.)code", DALI_FLOAT_VEC)
  .AddParent("Crop");

/*template <typename Out>
void RunHelper(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(0);
  auto &output = ws->Output<CPUBackend>(idx);

  Out *output_ptr = output.template mutable_data<Out>();
  const int stride = input.dim(1) * C_;
  const int mirror_image = !has_mirror_ ? mirror_.data<int>()[0] :
     spec_.GetArgument<int>("mirror", ws, ws->data_idx());

  vector<Index> input_shape = input.shape();
  DALI_ENFORCE(input_shape.size() == 3,
      "Expects 3-dimensional image input.");

  int H = input_shape[0];
  int W = input_shape[1];
  auto coord = CropAttr::GetCropWindowGenerator(ws->data_idx())(H, W);
  int crop_offsets = coord.y*W*C_ + coord.x*C_;
}*/

template <>
void NewCropMirrorNormalize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  if (output_layout_ == DALI_SAME) {
    output_layout_ = ws->Input<CPUBackend>(0).GetLayout();
  }

  if (output_type_ == DALI_NO_TYPE) {
    output_type_ = ws->Input<CPUBackend>(0).type().id();
  }
  CropAttr::ProcessArguments(ws);
}

template <>
void NewCropMirrorNormalize<CPUBackend>::DataDependentSetup(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);

  const DALITensorLayout in_layout = input.GetLayout();
  DALI_ENFORCE(in_layout == DALI_NHWC || in_layout == DALI_NCHW
            || in_layout == DALI_NFHWC || in_layout == DALI_NFCHW,
    "Unexpected data layout");
  DALITensorLayout out_layout = in_layout;

  auto data_idx = ws->data_idx();
  SetupSample(data_idx, in_layout, input.shape());
  auto &slice_shape = slice_shapes_[data_idx];

  auto &output = ws->Output<CPUBackend>(idx);
  output.SetLayout(out_layout);
}


template <>
void NewCropMirrorNormalize<CPUBackend>::DataDependentSetup(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);

  // DALITensorLayout outLayout;
  // output.Resize(GetOutShape(input.GetLayout(), &outLayout));
  // output.SetLayout(outLayout);
}

template <>
void NewCropMirrorNormalize<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
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

DALI_REGISTER_OPERATOR(NewCropMirrorNormalize, NewCropMirrorNormalize<CPUBackend>, CPU);

}  // namespace dali
