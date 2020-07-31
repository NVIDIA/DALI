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
#include "dali/operators/sequence/optical_flow/optical_flow.h"

namespace dali {


DALI_SCHEMA(OpticalFlow)
                .DocStr(R"code(Calculates the optical flow for a sequence of images that were
provided as the input.

The mandatory input for the operator is a sequence of frames. As an Optionally, the operator
also accepts external hints for the optical flow calculation. The output format of this operator
matches the output format of the optical flow driver API.
Refer to https://developer.nvidia.com/opticalflow-sdk for more information about the
Turing and Ampere optical flow hardware that is used by DALI. This operator allows sequence inputs.
)code")
                .NumInput(1, 2)
                .NumOutput(1)
                .AddOptionalArg(detail::kPresetArgName, R"code(Speed and quality level of the
optical flow calculation.

Here are the supported values:

- 0.0: The lowest speed and the highest quality.
- 0.5: The medium speed and quality.
- 1.0: The fastest speed and the lowest quality.

The lower the speed, the more additional pre- and postprocessing is used to enhance the quality of the optical flow result.
)code", .0f, false)
                .AddOptionalArg(detail::kOutputFormatArgName,
                                R"code(Sets the grid size for the output vector.

The value defines the width of the grid square. For example,if vthe value == 4, a 4x4 grid is used.
For values that are less than or equal to 0, the grid size is undefined.

.. note::
  Currently, only a grid_size=4 is supported.
)code", -1, false)
                .AddOptionalArg(detail::kEnableTemporalHintsArgName,
                                R"code(Enables or disables temporal hints for sequences that are
longer than two images.

The hints are used to speed up the calculation, where the previous optical flow result
in the sequence is used to calculate current flow. We recommend that you use temporal
hints for sequences that do not have many changes in the scene (for example, only moving objects).
))code",
                                false, false)
                .AddOptionalArg(detail::kEnableExternalHintsArgName,
                                R"code(Enables or disables the external hints for optical flow
calculations.

External hints are analogous to temporal hints, but the only difference is that external hints
come from an external source. When this option is enabled, the operator requires two inputs.
)code",
                                false, false)
                .AddOptionalArg(detail::kImageTypeArgName,
                                R"code(Type of input images, including RGB, BGR, and GRAY.)code", DALI_RGB,
                                false)
                .AllowSequences();


DALI_REGISTER_OPERATOR(OpticalFlow, OpticalFlow<GPUBackend>, GPU);

constexpr int kNOutputDims = 2;
constexpr int kNInputDims = 4;


template<>
void OpticalFlow<GPUBackend>::RunImpl(Workspace<GPUBackend> &ws) {
  if (enable_external_hints_) {
    // Fetch data
    // Input is a TensorList, where every Tensor is a sequence
    const auto &input = ws.Input<GPUBackend>(0);
    const auto &hints = ws.Input<GPUBackend>(1);
    auto &output = ws.Output<GPUBackend>(0);
    output.SetLayout("HWC");  // Channels represent the two flow vector components (x and y)
    // Extract calculation params
    ExtractParams(input, hints);

    of_lazy_init(frames_width_, frames_height_, depth_, image_type_, device_id_, ws.stream());

    auto out_shape = optical_flow_->GetOutputShape();
    TensorListShape<> new_sizes(nsequences_, 1 + out_shape.sample_dim());
    for (int i = 0; i < nsequences_; i++) {
      auto shape = shape_cat(sequence_sizes_[i] - 1, out_shape);
      new_sizes.set_tensor_shape(i, shape);
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
        auto ref = subtensor(sequence_tv, i - 1);
        auto in = subtensor(sequence_tv, i);
        auto h = subtensor(hints_tv, i);
        auto out = subtensor(output_tv, i - 1);

        optical_flow_->CalcOpticalFlow(ref, in, out, h);
      }
    }
  } else {
    // Fetch data
    // Input is a TensorList, where every Tensor is a sequence
    const auto &input = ws.Input<GPUBackend>(0);
    auto &output = ws.Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());

    // Extract calculation params
    ExtractParams(input);

    of_lazy_init(frames_width_, frames_height_, depth_, image_type_, device_id_, ws.stream());

    auto out_shape = optical_flow_->GetOutputShape();
    TensorListShape<> new_sizes(nsequences_, 1 + out_shape.sample_dim());
    for (int i = 0; i < nsequences_; i++) {
      auto shape = shape_cat(sequence_sizes_[i] - 1, out_shape);
      new_sizes.set_tensor_shape(i, shape);
    }
    output.Resize(new_sizes);

    // Prepare input and output TensorViews
    auto tvlin = view<const uint8_t, kNInputDims>(input);
    auto tvlout = view<float, kNInputDims>(output);

    for (int sequence_idx = 0; sequence_idx < nsequences_; sequence_idx++) {
      auto sequence_tv = tvlin[sequence_idx];
      auto output_tv = tvlout[sequence_idx];

      for (int i = 1; i < sequence_tv.shape[0]; i++) {
        auto ref = subtensor(sequence_tv, i - 1);
        auto in = subtensor(sequence_tv, i);
        auto out = subtensor(output_tv, i - 1);

        optical_flow_->CalcOpticalFlow(ref, in, out);
      }
    }
  }
}

}  // namespace dali
