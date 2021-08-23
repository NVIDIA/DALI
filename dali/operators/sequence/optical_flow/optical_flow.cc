// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/util/nvml.h"
#include "dali/pipeline/data/views.h"
#include "dali/operators/sequence/optical_flow/optical_flow.h"

namespace dali {


DALI_SCHEMA(OpticalFlow)
                .DocStr(R"code(Calculates the optical flow between images in the input.

The main input for this operator is a sequence of frames. Optionally, the operator
can be provided with external hints for the optical flow calculation. The output format of this operator
matches the output format of the optical flow driver API.
Refer to https://developer.nvidia.com/opticalflow-sdk for more information about the
Turing and Ampere optical flow hardware that is used by DALI.
)code")
                .NumInput(1, 2)
                .NumOutput(1)
                .AddOptionalArg(detail::kPresetArgName, R"code(Speed and quality level of the
optical flow calculation.

Allowed values are:

- ``0.0`` is the lowest speed and the best quality.
- ``0.5`` is the medium speed and quality.
- ``1.0`` is the fastest speed and the lowest quality.

The lower the speed, the more additional pre- and postprocessing is used to enhance the quality of the optical flow result.
)code", .0f, false)
                .AddOptionalArg(detail::kOutputFormatArgName,
                                R"code(Sets the grid size for the output vector field.

This operator produces the motion vector field at a coarser resolution than the input pixels.
This parameter specifies the size of the pixel grid cell corresponding to one motion vector.
For example, a value of 4 will produce one motion vector for each 4x4 pixel block.

.. note::
  Currently, only a grid_size=4 is supported.
)code", -1, false)
                .AddOptionalArg(detail::kEnableTemporalHintsArgName,
                                R"code(Enables or disables temporal hints for sequences that are
longer than two images.

The hints are used to improve the quality of the output motion field as well as to speed up
the calculations. The hints are especially useful in presence of large displacements or
periodic patterns which might confuse the optical flow algorithms.
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
                                R"code(Input color space (RGB, BGR or GRAY).)code", DALI_RGB,
                                false)
                .AllowSequences();


DALI_REGISTER_OPERATOR(OpticalFlow, OpticalFlow<GPUBackend>, GPU);

constexpr int kNOutputDims = 2;
constexpr int kNInputDims = 4;


template<>
void OpticalFlow<GPUBackend>::RunImpl(Workspace<GPUBackend> &ws) {
  // This is a workaround for an issue with nvcuvid in drivers >460 where concurrent
  // use on default context and non-default streams may lead to memory corruption.
  // TODO(michalz): add an upper bound when the problem is fixed
  cudaStream_t of_stream = ws.stream();
#if NVML_ENABLED
  {
    static float driver_version = nvml::GetDriverVersion();
    if (driver_version > 460)
      of_stream = 0;
  }
#else
  {
    int driver_cuda_version = 0;
    CUDA_CALL(cuDriverGetVersion(&driver_cuda_version));
    if (driver_cuda_version >= 11030)
      of_stream = 0;
  }
#endif
  if (of_stream != ws.stream()) {
    DALI_WARN_ONCE("Warning: Running optical flow on a default stream.\n"
                   "Performance may be affected.");
  }

  if (of_stream != ws.stream()) {
    CUDA_CALL(cudaEventRecord(sync_, ws.stream()));
    CUDA_CALL(cudaStreamWaitEvent(of_stream, sync_, 0));
  }

  if (enable_external_hints_) {
    // Fetch data
    // Input is a TensorList, where every Tensor is a sequence
    const auto &input = ws.Input<GPUBackend>(0);
    const auto &hints = ws.Input<GPUBackend>(1);
    auto &output = ws.Output<GPUBackend>(0);
    output.SetLayout("HWC");  // Channels represent the two flow vector components (x and y)
    // Extract calculation params
    ExtractParams(input, hints);

    of_lazy_init(frames_width_, frames_height_, depth_, image_type_, device_id_, of_stream);

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

    of_lazy_init(frames_width_, frames_height_, depth_, image_type_, device_id_, of_stream);

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
  if (of_stream != ws.stream()) {
    CUDA_CALL(cudaEventRecord(sync_, of_stream));
    CUDA_CALL(cudaStreamWaitEvent(ws.stream(), sync_, 0));
  }
}

template <>
OpticalFlow<GPUBackend>::~OpticalFlow() {
#if NVML_ENABLED
  nvml::Shutdown();
#endif
}

}  // namespace dali
