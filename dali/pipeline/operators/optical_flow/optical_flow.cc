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

#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operators/optical_flow/optical_flow.h"

namespace dali {


DALI_SCHEMA(OpticalFlow)
                .DocStr(R"code(Calculates the Optical Flow for sequence of images given as a input.
Mandatory input for the operator is a sequence of frames.
As an optional input, operator accepts external hints for OF calculation.
The output format of this operator matches the output format of OF driver API.
Dali uses Turing optical flow hardware implementation: https://developer.nvidia.com/opticalflow-sdk
)code")
                .NumInput(1, 2)
                .NumOutput(1)
                .AddOptionalArg(detail::kPresetArgName, R"code(Setting quality level of OF calculation.
 0.0f ... 1.0f, where 1.0f is best quality, lowest speed)code", .0f, false)
                .AddOptionalArg(detail::kOutputFormatArgName,
                                R"code(Setting grid size for output vector.
Value defines width of grid square (e.g. if value == 4, 4x4 grid is used).
For values <=0, grid size is undefined. Currently only grid_size=4 is supported.)code", -1, false)
                .AddOptionalArg(detail::kEnableTemporalHintsArgName,
                                R"code(enabling/disabling temporal hints for sequences longer than 2 images.
They are used to speed up calculation: previous OF result in sequence is used to calculate current flow. You might
want to use temporal hints for sequences, that don't have much changes in the scene (e.g. only moving objects))code",
                                false, false)
                .AddOptionalArg(detail::kEnableExternalHintsArgName,
                                R"code(enabling/disabling external hints for OF calculation. External hints
are analogous to temporal hints, only they come from external source. When this option is enabled,
Operator requires 2 inputs.)code",
                                false, false)
                .AddOptionalArg(detail::kImageTypeArgName,
                                R"code(Type of input images (RGB, BGR, GRAY))code", DALI_RGB,
                                false);


DALI_REGISTER_OPERATOR(OpticalFlow, OpticalFlow<GPUBackend>, GPU);

constexpr int kNOutputDims = 2;
constexpr int kNInputDims = 4;


template<>
void OpticalFlow<GPUBackend>::RunImpl(Workspace<GPUBackend> *ws, const int) {
  if (enable_external_hints_) {
    // Fetch data
    // Input is a TensorList, where every Tensor is a sequence
    const auto &input = ws->Input<GPUBackend>(0);
    const auto &hints = ws->Input<GPUBackend>(1);
    auto &output = ws->Output<GPUBackend>(0);

    of_lazy_init(frames_width_, frames_height_, depth_, image_type_, ws->stream());

    // Extract calculation params
    ExtractParams(input, hints);
    std::vector<Dims> new_sizes;
    auto out_shape = optical_flow_->GetOutputShape();
    for (int i = 0; i < nsequences_; i++) {
      Dims shape = {sequence_sizes_[i]};
      shape.insert(shape.end(), out_shape.begin(), out_shape.end());
      new_sizes.emplace_back(shape);
    }
    output.Resize(new_sizes);


    // Prepare input and output TensorViews
    auto tvlin = view<const uint8_t, kNInputDims>(input);
    auto tvlout = view<float, kNInputDims>(output);
    auto tvlhints = view<const float, kNInputDims>(hints);
    DALI_ENFORCE(tvlhints.size() == nsequences_,
                 "Number of tensors for hints and inputs doesn't match");

    for (int sequence_idx = 0; sequence_idx < nsequences_; sequence_idx++) {
      auto sequence_tv = tvlin[sequence_idx];
      auto output_tv = tvlout[sequence_idx];
      auto hints_tv = tvlhints[sequence_idx];

      for (int i = 1; i < sequence_tv.shape[0]; i++) {
        auto ref = kernels::subtensor(sequence_tv, i - 1);
        auto in = kernels::subtensor(sequence_tv, i);
        auto h = kernels::subtensor(hints_tv, i);
        auto out = kernels::subtensor(output_tv, i - 1);

        optical_flow_->CalcOpticalFlow(ref, in, out, h);
      }
    }
  } else {
    // Fetch data
    // Input is a TensorList, where every Tensor is a sequence
    const auto &input = ws->Input<GPUBackend>(0);
    auto &output = ws->Output<GPUBackend>(0);

    of_lazy_init(frames_width_, frames_height_, depth_, image_type_, ws->stream());

    // Extract calculation params
    ExtractParams(input);
    std::vector<Dims> new_sizes;
    auto out_shape = optical_flow_->GetOutputShape();
    for (int i = 0; i < nsequences_; i++) {
      Dims shape = {sequence_sizes_[i]};
      shape.insert(shape.end(), out_shape.begin(), out_shape.end());
      new_sizes.emplace_back(shape);
    }
    output.Resize(new_sizes);


    // Prepare input and output TensorViews
    auto tvlin = view<const uint8_t, kNInputDims>(input);
    auto tvlout = view<float, kNInputDims>(output);

    for (int sequence_idx = 0; sequence_idx < nsequences_; sequence_idx++) {
      auto sequence_tv = tvlin[sequence_idx];
      auto output_tv = tvlout[sequence_idx];

      for (int i = 1; i < sequence_tv.shape[0]; i++) {
        auto ref = kernels::subtensor(sequence_tv, i - 1);
        auto in = kernels::subtensor(sequence_tv, i);
        auto out = kernels::subtensor(output_tv, i - 1);

        optical_flow_->CalcOpticalFlow(ref, in, out);
      }
    }
  }
}

}  // namespace dali
