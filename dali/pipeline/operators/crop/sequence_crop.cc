// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/crop/sequence_crop.h"
#include "dali/pipeline/basic/crop.h"
#include "dali/pipeline/basic/tensor.h"
#include "dali/pipeline/basic/type_switch.h"

namespace dali {

DALI_SCHEMA(SequenceCrop)
    .DocStr(R"code(Perform a random crop on a sequecne.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddParent("Crop")
    .EnforceInputLayout(DALI_NHWC);

void SequenceCrop::RunImpl(Workspace<CPUBackend> *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto *output = ws->Output<CPUBackend>(idx);

  DALITensorLayout outLayout = output_layout_ == DALI_SAME ? input.GetLayout() : output_layout_;
  output->SetLayout(outLayout);
  // output->Resize(GetOutShape(input.GetLayout(), &outLayout));
  auto layout_id = layoutToTypeId(output->GetLayout());
  using CropOutputTypes = std::tuple<uint8_t, int16_t, int32_t, int64_t, float>;
  using CropOutputPermTypes =
      std::tuple<dali_index_sequence<0, 1, 2>, dali_index_sequence<2, 0, 1>>;

  // Check if we use u8, RGB or Greyscale
  // CheckParam(input, "CropCPUBackend");

  const int threadIdx = ws->thread_idx();
  const int h_start = per_sample_crop_[threadIdx].first;
  const int w_start = per_sample_crop_[threadIdx].second;

  // TODO(klecki): simplification - do not handle float16

  const int dataIdx = ws->data_idx();
  type_switch<basic::SequenceCropSizeHelper, CropOutputTypes, CropOutputPermTypes>::Run(
      output_type_, layout_id, ws, idx, h_start, w_start, crop_height_[dataIdx], crop_width_[dataIdx]);

  type_switch<basic::SequenceCropRunHelper, CropOutputTypes, CropOutputPermTypes>::Run(
      output_type_, layout_id, ws, idx, h_start, w_start, crop_height_[dataIdx], crop_width_[dataIdx]);
}

const std::vector<Index> SequenceCrop::CheckShapes(const SampleWorkspace *ws) {
  const auto &input = ws->Input<CPUBackend>(0);

  DALI_ENFORCE(input.ndim() == 4, "Operator expects 4-dimensional sequence image input.");

  // enforce that all shapes match
  for (int i = 1; i < ws->NumInput(); ++i) {
    const auto &other_input = ws->Input<CPUBackend>(i);
    DALI_ENFORCE(other_input.ndim() == 4, "Operator expects 4-dimensional sequence image input.");
    for (int j = 1; j < 4; j++) {
      DALI_ENFORCE(input.dim(j) == other_input.dim(j));
    }
  }

  std::vector<Index> frame_shape;
  for (int i = 1; i < 4; i++) {
    frame_shape.push_back(input.dim(i));
  }

  return frame_shape;
}

DALI_REGISTER_OPERATOR(SequenceCrop, SequenceCrop, CPU);

}  // namespace dali
