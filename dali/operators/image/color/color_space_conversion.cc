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

#include <map>
#include "dali/operators/image/color/color_space_conversion.h"
#include "dali/util/ocv.h"

namespace dali {

DALI_SCHEMA(ColorSpaceConversion)
    .DocStr(R"code(Converts between various image color models.)code")
    .NumInput(1)
    .NumOutput(1)
    .InputLayout({"FDHWC", "FHWC", "DHWC", "HWC"})
    .AddArg("image_type", R"code(The color space of the input image.)code", DALI_IMAGE_TYPE)
    .AddArg("output_type", R"code(The color space of the output image.)code", DALI_IMAGE_TYPE)
    .AllowSequences()
    .SupportVolumetric();

template <>
void ColorSpaceConversion<CPUBackend>::RunImpl(HostWorkspace &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  output.SetLayout(input.GetLayout());
  auto in_view = view<const uint8_t>(input);
  auto out_view = view<uint8_t>(output);
  const auto &in_sh = in_view.shape;
  const auto &out_sh = out_view.shape;
  int nsamples = in_sh.num_samples();
  int ndim = in_sh.sample_dim();
  auto& thread_pool = ws.GetThreadPool();
  for (int i = 0; i < nsamples; i++) {
    thread_pool.AddWork(
      [&, i](int thread_id) {
        auto in_sample_sh = in_sh.tensor_shape_span(i);
        // flatten any leading dimensions together with the height
        int height = volume(in_sample_sh.begin(), in_sample_sh.end() - 2);
        int width  = in_sample_sh[ndim - 2];
        auto cv_in =
          CreateMatFromPtr(height, width, GetOpenCvChannelType(in_nchannels_), in_view[i].data);
        auto cv_out =
          CreateMatFromPtr(height, width, GetOpenCvChannelType(out_nchannels_), out_view[i].data);
        OpenCvColorConversion(input_type_, cv_in, output_type_, cv_out);
      }, in_sh.tensor_size(i));
  }
  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(ColorSpaceConversion, ColorSpaceConversion<CPUBackend>, CPU);

}  // namespace dali
