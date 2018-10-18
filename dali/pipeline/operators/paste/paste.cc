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

#include "dali/pipeline/operators/paste/paste.h"

namespace dali {

DALI_SCHEMA(Paste)
  .DocStr(R"code(Paste the input image on a larger canvas.
The canvas size is equal to `input size * ratio`.)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddArg("ratio",
      R"code(Ratio of canvas size to input size, must be > 1.)code",
      DALI_FLOAT, true)
  .AddOptionalArg("n_channels",
      R"code(Number of channels in the image.)code",
      3)
  .AddArg("fill_value",
      R"code(Tuple of values of the color to fill the canvas.
  Length of the tuple needs to be equal to `n_channels`.)code",
      DALI_INT_VEC)
  .AddOptionalArg("paste_x",
      R"code(Horizontal position of the paste in image coordinates (0.0 - 1.0))code",
      0.5f, true)
  .AddOptionalArg("paste_y",
      R"code(Vertical position of the paste in image coordinates (0.0 - 1.0))code",
      0.5f, true)
  .EnforceInputLayout(DALI_NHWC);


void PasteKernel(
  const int C,
  const uint8* fill_value,
  const uint8* input_ptr,
  uint8* output_ptr,
  const int* in_out_dims_paste_yx) {

  const int in_H = in_out_dims_paste_yx[0];
  const int in_W = in_out_dims_paste_yx[1];
  const int out_H = in_out_dims_paste_yx[2];
  const int out_W = in_out_dims_paste_yx[3];
  const int paste_y = in_out_dims_paste_yx[4];
  const int paste_x = in_out_dims_paste_yx[5];

  const auto len = C * out_W;
  auto rightBound = paste_x + in_W;
  uint8* output = NULL;
  if (out_H > in_H) {
    // Define which row should be filled
    output = output_ptr + (!paste_y? C * out_H * (out_W - 1) : 0);
    uint8 *outputTmp = output - C;
    // Fill one full row
    for (int w = 0; w < out_W; ++w)
      memcpy(outputTmp += C, fill_value, C);

    // Copy filled row on top and on bottom of the image
    int h = output_ptr == output? 1 : 0;
    int hMax = paste_y;

    // Loop for top and bottom parts of the image
    int i = 0;
    while (true) {
      outputTmp = output + (h - 1) * len;
      for (; h < hMax; ++h)
        memcpy(outputTmp += len, output, len);

      if (++i > 1)
        break;   // Done with bottom part

      h = hMax + in_H;
      hMax = out_H - (output_ptr == output? 0 : 1);
    }
  } else {
    if (out_W == in_W) {
      memcpy(output_ptr, input_ptr, len * out_H);
      return;
    }

    // Choose longest (left or right) part of the image to be fill out
    const bool flag = paste_x <= out_W - rightBound;
    uint8 *outputTmp = (output = output_ptr + (flag? rightBound : 0)) - C;

    const auto wMax = flag? out_W - rightBound : paste_x;
    for (int w = 0; w < wMax; ++w)
      memcpy(outputTmp += C, fill_value, C);
  }

  // Creating the middle part of the output image
  const auto leftSize = C * paste_x;
  const auto middleSize = C * in_W;
  const auto rightSize = C * (out_W - rightBound);
  input_ptr -= middleSize;
  rightBound *= C;
  output_ptr += (paste_y - 1) * len;
  for (int h = 0; h < in_H; h++) {
    memcpy(output_ptr += len, output, leftSize);
    memcpy(output_ptr + leftSize, input_ptr += middleSize, middleSize);
    memcpy(output_ptr + rightBound, output, rightSize);
  }
}

template<>
void Paste<CPUBackend>::SetupSampleParams(SampleWorkspace *ws, const int idx) {
}

template<>
void Paste<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto output = ws->Output<CPUBackend>(idx);

  std::vector<int>sample_dims_paste_yx;
  output->set_type(input.type());
  output->Resize(Prepare(input.shape(), spec_, ws, idx, sample_dims_paste_yx));
  output->SetLayout(DALI_NHWC);

  PasteKernel(C_,
              fill_value_.template data<uint8>(),
              input.template data<uint8>(),
              output->template mutable_data<uint8>(),
              sample_dims_paste_yx.data());
}

DALI_REGISTER_OPERATOR(Paste, Paste<CPUBackend>, CPU);

}  // namespace dali
