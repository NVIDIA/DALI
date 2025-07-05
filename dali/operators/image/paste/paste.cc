// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/paste/paste.h"

namespace dali {

DALI_SCHEMA(Paste)
    .DocStr(R"code(Pastes the input images on a larger canvas, where the canvas size is equal to
``input size * ratio``. Only uint8 images of up to 1024 channels are supported.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("ratio", R"code(Ratio of canvas size to input size. Must be >= 1.)code", DALI_FLOAT,
            true)
    .AddOptionalArg("n_channels", R"code(Number of channels in the image.)code", 3)
    .AddArg("fill_value",
            R"code(Tuple of the values of the color that is used to fill the canvas.

The length of the tuple must be equal to `n_channels`.)code",
            DALI_INT_VEC)
    .AddOptionalArg("paste_x",
                    R"code(Horizontal position of the paste in (0.0 - 1.0) image coordinates.)code",
                    0.5f, true)
    .AddOptionalArg("paste_y",
                    R"code(Vertical position of the paste in (0.0 - 1.0) image coordinates.)code",
                    0.5f, true)
    .AddOptionalArg("min_canvas_size",
                    R"code(Enforces the minimum paste canvas dimension after scaling the input size
by the ratio.)code",
                    0.0f, true)
    .InputLayout("HWC");

struct PasteParameters {
  int in_H;
  int in_W;
  int out_H;
  int out_W;
  int paste_y;
  int paste_x;
};

void PasteKernel(ConstSampleView<CPUBackend> input, SampleView<CPUBackend> output,
                 const Tensor<CPUBackend> &fill_value, const PasteParameters &params) {
  const auto img_C = input.shape()[2];
  const auto fill_C = fill_value.size();

  // Fill output row with the fill value
  auto fill_range = [&](std::size_t offset, std::size_t nelem) {
    if (fill_C == 1) {
      std::memset(output.mutable_data<uint8_t>() + offset, fill_value.data<uint8_t>()[0],
                  nelem * img_C);
    }
    for (std::size_t w = 0; w < nelem; ++w) {
      std::copy(fill_value.data<uint8_t>(), fill_value.data<uint8_t>() + fill_C,
                output.mutable_data<uint8_t>() + offset + w * fill_C);
    }
  };

  for (int h = 0; h < params.out_H; ++h) {
    const auto out_row_offset = h * params.out_W * img_C;
    if (h < params.paste_y || h >= params.paste_y + params.in_H) {
      fill_range(out_row_offset, params.out_W);
    } else {
      fill_range(out_row_offset, params.paste_x);
      std::memcpy(output.mutable_data<uint8_t>() + out_row_offset + params.paste_x * img_C,
                  input.data<uint8_t>() + (h - params.paste_y) * params.in_W * img_C,
                  params.in_W * img_C);
      fill_range(out_row_offset + (params.paste_x + params.in_W) * img_C,
                 params.out_W - params.paste_x - params.in_W);
    }
  }
}

template <>
void Paste<CPUBackend>::RunHelper(Workspace &ws) {
  auto &tp = ws.GetThreadPool();
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);

  for (int sampleIdx = 0; sampleIdx < input.num_samples(); sampleIdx++) {
    tp.AddWork([&, sampleIdx](int) {
      const auto *pasteParams =
          static_cast<const PasteParameters *>(in_out_dims_paste_yx_.raw_data()) + sampleIdx;
      PasteKernel(input[sampleIdx], output[sampleIdx], fill_value_, *pasteParams);
    });
  }
  tp.RunAll();
}

template <>
void Paste<CPUBackend>::SetupGPUPointers(Workspace &ws) {}

DALI_REGISTER_OPERATOR(Paste, Paste<CPUBackend>, CPU);

}  // namespace dali
