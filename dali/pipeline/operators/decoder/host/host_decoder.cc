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

#include <opencv2/opencv.hpp>
#include <tuple>
#include <memory>
#include "dali/image/image_factory.h"
#include "dali/pipeline/operators/decoder/host/host_decoder.h"

namespace dali {

void HostDecoder::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);
  auto file_name = input.GetSourceInfo();

  // Verify input
  DALI_ENFORCE(input.ndim() == 1,
                "Input must be 1D encoded jpeg string.");
  DALI_ENFORCE(IsType<uint8>(input.type()),
                "Input must be stored as uint8 data.");

  std::unique_ptr<Image> img;
  try {
    img = ImageFactory::CreateImage(input.data<uint8>(), input.size(), output_type_);
    img->SetCropWindowGenerator(GetCropWindowGenerator(ws->data_idx()));
    img->Decode();
  } catch (std::exception &e) {
    DALI_FAIL(e.what() + "File: " + file_name);
  }
  const auto decoded = img->GetImage();
  const auto hwc = img->GetImageDims();
  const auto h = std::get<0>(hwc);
  const auto w = std::get<1>(hwc);
  const auto c = std::get<2>(hwc);

  output.Resize({static_cast<int>(h), static_cast<int>(w), static_cast<int>(c)});
  unsigned char *out_data = output.mutable_data<unsigned char>();
  std::memcpy(out_data, decoded.get(), h * w * c);
}

DALI_SCHEMA(HostDecoder)
  .DocStr(R"code(Specific implementation of `ImageDecoder` for `cpu` backend)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoder")
  .Deprecate("ImageDecoder");

DALI_REGISTER_OPERATOR(HostDecoder, HostDecoder, CPU);
DALI_REGISTER_OPERATOR(ImageDecoder, HostDecoder, CPU);

}  // namespace dali
