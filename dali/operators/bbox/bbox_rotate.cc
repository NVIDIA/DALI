// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bbox_rotate.h"

namespace dali {

DALI_SCHEMA(BBoxRotate)
    .DocStr(
        R"code(Transforms bounding boxes so that the boxes remain in the same place in the image after
the image is rotated. Boxes that land outside the image with `keep_size=True` will be removed from the
output, a mask of valid boxes will be returned that can be used to remove other data such as class labels.

Box coordinates must be (0,1) normalized and the image is assumed to be center rotated with `fn.rotate`.
)code")
    .NumInput(1, 2)
    .InputDox(
        0, "bboxes", "2D TensorList of float",
        R"code(Relative coordinates of the bounding boxes that are represented as a 2D tensor, where the
first dimension refers to the index of the bounding box, and the second dimension refers to the index
of the coordinate.
)code")
    .InputDox(
        1, "labels", "1D TensorList of int",
        R"code(Class labels for the bounding boxes. These should be provided if the `keep_size` argument is
set to True as boxes may be truncated, this ensures the corresponding labels will also be removed.
)code")
    .NumOutput(1)
    .AdditionalOutputsFn([](const OpSpec& spec) {
      return spec.NumRegularInput() - 1;  // +1 if labels are provided
    })
    .AddArg("ltrb", R"code(True for ``ltrb`` or False for ``xywh``.)code", DALI_BOOL, false)
    .AddArg("angle", R"code(Rotation angle in degrees.)code", DALI_FLOAT, true)
    .AddOptionalArg(
        "keep_size",
        R"code(If true, the bounding box output coordinates will assume the image canvas size was also kept 
(see `nvidia.dali.fn.rotate`).)code",
        false, false)
    .AddOptionalArg("mode",
                    R"code(Mode of the bounding box transformation. Possible values are:
- ``expand``: expands the bounding box to definitely enclose the target, but may be larger than rotated target.
- ``fixed``: retains the original size of the bounding box, but may be smaller than the rotated target. 
- ``halfway``: halfway between the expanded size and the original size.
)code",
                    dali::string("expand"), false);

void rotateBoxesKernel(ConstSampleView<CPUBackend> inputBoxes, SampleView<CPUBackend> rotateBuffer,
                       SampleView<CPUBackend> outputBoxes, float angle,
                       std::optional<ConstSampleView<CPUBackend>> inputLabels,
                       std::optional<SampleView<CPUBackend>> outputLabels) {}

template <>
void BBoxRotate<CPUBackend>::RunImpl(Workspace& ws) {
  const auto& inputBoxes = ws.Input<CPUBackend>(0);

  auto& tpool = ws.GetThreadPool();
  for (int sampleIdx = 0; sampleIdx < inputBoxes.num_samples(); ++sampleIdx) {
    tpool.AddWork([&, sampleIdx](int) {
      std::optional<ConstSampleView<CPUBackend>> inputLabels;
      std::optional<SampleView<CPUBackend>> outputLabels;
      if (ws.NumInput() == 2) {
        inputLabels = ws.Input<CPUBackend>(1)[sampleIdx];
        outputLabels = ws.Output<CPUBackend>(1)[sampleIdx];
      }
      float angle;
      if (spec_.HasTensorArgument("angle")) {
        angle = ws.ArgumentInput("angle")[sampleIdx].data<float>()[0];
      } else {
        angle = spec_.GetArgument<float>("angle");
      }
      rotateBoxesKernel(inputBoxes[sampleIdx], bbox_rotate_buffer_[sampleIdx],
                        ws.Output<CPUBackend>(0)[sampleIdx], angle, inputLabels, outputLabels);
    });
  }
  tpool.RunAll();
}

DALI_REGISTER_OPERATOR(BBoxRotate, BBoxRotate<CPUBackend>, CPU);

}  // namespace dali