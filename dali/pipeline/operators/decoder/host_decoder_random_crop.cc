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
  : HostDecoder(spec) {
    int64_t seed = spec.GetArgument<int64_t>("seed");
    int num_attempts = spec.GetArgument<int>("num_attempts");

    std::vector<float> aspect_ratio;
    GetSingleOrRepeatedArg(spec, &aspect_ratio, "random_aspect_ratio", 2);

    std::vector<float> area;
    GetSingleOrRepeatedArg(spec, &area, "random_area", 2);

    std::shared_ptr<RandomCropGenerator> random_crop_generator(
      new RandomCropGenerator(
        {aspect_ratio[0], aspect_ratio[1]},
        {area[0], area[1]},
        seed,
        num_attempts));

    crop_window_generator_ = std::bind(
      &RandomCropGenerator::GenerateCropWindow, random_crop_generator,
      std::placeholders::_1, std::placeholders::_2);
}

DALI_REGISTER_OPERATOR(HostDecoderRandomCrop, HostDecoderRandomCrop, CPU);

DALI_SCHEMA(HostDecoderRandomCrop)
  .DocStr(R"code(Decode images on the host with a random cropping anchor/window.
When possible, will make use of partial decoding (e.g. libjpeg-turbo).
When not supported, will decode the whole image and then crop.
Output of the decoder is in `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("random_aspect_ratio",
      R"code(Range from which to choose random aspect ratio.)code",
      std::vector<float>{3./4., 4./3.})
  .AddOptionalArg("random_area",
      R"code(Range from which to choose random area factor `A`.
The cropped image's area will be equal to `A` * original image's area.)code",
      std::vector<float>{0.08, 1.0})
  .AddOptionalArg("num_attempts",
      R"code(Maximum number of attempts used to choose random area and aspect ratio.)code",
      10)
  .AddParent("HostDecoder");

}  // namespace dali
