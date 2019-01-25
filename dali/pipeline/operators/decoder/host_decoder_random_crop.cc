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

#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include <memory>
#include "dali/error_handling.h"
#include "dali/image/image_factory.h"
#include "dali/pipeline/operators/decoder/host_decoder_random_crop.h"
#include "dali/pipeline/operators/common.h"
#include "dali/util/random_crop_generator.h"

namespace dali {

HostDecoderRandomCrop::HostDecoderRandomCrop(const OpSpec &spec)
  : Operator<CPUBackend>(spec)
  , output_type_(spec.GetArgument<DALIImageType>("output_type"))
  , c_(IsColor(output_type_) ? 3 : 1)
  , seed_(spec.GetArgument<int64_t>("seed")) {
  std::vector<float> aspect_ratio;
  GetSingleOrRepeatedArg(spec, &aspect_ratio, "random_aspect_ratio", 2);

  std::vector<float> area;
  GetSingleOrRepeatedArg(spec, &area, "random_area", 2);

  random_crop_generator_.reset(
    new RandomCropGenerator(
      {aspect_ratio[0], aspect_ratio[1]},
      {area[0], area[1]},
      seed_));
}

void HostDecoderRandomCrop::RunImpl(SampleWorkspace *ws, const int idx) {
  auto &input = ws->Input<CPUBackend>(idx);
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
    img->SetRandomCropGenerator(random_crop_generator_);
    img->Decode();
  } catch (std::runtime_error &e) {
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

DALI_REGISTER_OPERATOR(HostDecoderRandomCrop, HostDecoderRandomCrop, CPU);

DALI_SCHEMA(HostDecoderRandomCrop)
  .DocStr(R"code(Decode images on the host with a random cropping anchor/window.
When possible, will make use of partial decoding (e.g. libjpeg-turbo).
When not supported, will decode the whole image and then crop.
Output of the decoder is in `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type",
      R"code(The color space of output image.)code",
      DALI_RGB)
  .AddOptionalArg("random_aspect_ratio",
      R"code(Range from which to choose random aspect ratio.)code",
      std::vector<float>{3./4., 4./3.})
  .AddOptionalArg("random_area",
      R"code(Range from which to choose random area factor `A`.
The cropped image's area will be equal to `A` * original image's area.)code",
      std::vector<float>{0.08, 1.0});

}  // namespace dali
