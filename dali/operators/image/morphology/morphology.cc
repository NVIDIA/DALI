// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/morphology/morphology.h"

namespace dali {

bool Morphology::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto sh = input.shape();
  output_desc.resize(1);
  output_desc[0] = {sh, input.type()};
  return true;
}

void Morphology::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());

  kernels::DynamicScratchpad scratchpad(AccessOrder(ws.stream()));
  auto mask = AcquireTensorArgument<int32_t>(ws, scratchpad, mask_arg_,
                                             TensorShape<1>(2), nvcvop::GetDataType<int32_t>(2));
  auto anchor = AcquireTensorArgument<int32_t>(ws, scratchpad, anchor_arg_,
                                               TensorShape<1>(2), nvcvop::GetDataType<int32_t>(2));

  if (!op_workspace_ || (op_workspace_.capacity() < input.num_samples())) {
    int current_size = (op_workspace_) ? op_workspace_.capacity() : 0;
    op_workspace_ =
        nvcv::ImageBatchVarShape(std::max(current_size * 2, input.num_samples()));
  }
  op_workspace_.clear();
  nvcvop::AllocateImagesLike(op_workspace_, input, scratchpad);

  auto input_images = GetInputBatch(ws, 0);
  auto output_images = GetOutputBatch(ws, 0);
  cvcuda::Morphology op{};
  op(ws.stream(), input_images, output_images, op_workspace_, morph_type_, mask, anchor, iteration_,
     border_mode_);
}

DALI_SCHEMA(Morphology)
  .AddOptionalArg("mask_size", "Size of the structuring element.",
                  std::vector<int32_t>({3, 3}), true, true)
  .AddOptionalArg("anchor",
                  "Sets the anchor point of the structuring element. Default value (-1, -1)"
                  " uses the element's center as the anchor point.",
                  std::vector<int32_t>({-1, -1}), true, true)
  .AddOptionalArg("iterations",
                  "Number of times to execute the operation, typically set to 1. "
                  "Setting to a value higher than 1 is equivelent to increasing the mask size "
                  "by (mask_width - 1, mask_height -1) for every additional iteration.",
                  1, false, false)
  .AddOptionalArg("border_mode",
                  "Border mode to be used when accessing elements outside input image.",
                  "constant");

DALI_SCHEMA(experimental__Dilate)
  .AddParent("Morphology")
  .DocStr("Performs a dilation operation on the input image.")
  .NumInput(1)
  .NumOutput(1)
  .InputDox(0, "input", "TensorList",
            "Input data. Must be images in HWC or CHW layout, or a sequence of those.")
  .AllowSequences()
  .InputLayout({"HW", "HWC", "FHWC", "CHW", "FCHW"});

DALI_SCHEMA(experimental__Erode)
  .AddParent("Morphology")
  .DocStr("Performs an erosion operation on the input image.")
  .NumInput(1)
  .NumOutput(1)
  .InputDox(0, "input", "TensorList",
            "Input data. Must be images in HWC or CHW layout, or a sequence of those.")
  .AllowSequences()
  .InputLayout({"HW", "HWC", "FHWC", "CHW", "FCHW"});


DALI_REGISTER_OPERATOR(experimental__Dilate, Dilate, GPU);

DALI_REGISTER_OPERATOR(experimental__Erode, Erode, GPU);

}  // namespace dali
