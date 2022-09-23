// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_REGISTRY_H_
#define DALI_IMGCODEC_REGISTRY_H_

#include <memory>
#include "dali/imgcodec/image_format.h"

namespace dali {
namespace imgcodec {

template <typename Factory>
void RegisterDecoder(const char *format_name, float priority) {
  if (format_name[0] == '*') {
    auto factory = std::make_shared<Factory>();
    for (ImageFormat *format : ImageFormatRegistry::instance().Formats())
          format->RegisterDecoder(factory, priority);
  } else {
    auto *format = ImageFormatRegistry::instance().GetImageFormat(format_name);
    assert(format != nullptr);
    format->RegisterDecoder(std::make_shared<Factory>(), priority);
  }
}


template <typename Factory>
struct ImageDecoderRegisterer {
  ImageDecoderRegisterer(const char *format_name, float priority) {
    RegisterDecoder<Factory>(format_name, priority);
  }
};

#define REGISTER_DECODER(FormatName, Factory, Priority) \
static auto &RegisterDecoder_##Factory() { \
  static ImageDecoderRegisterer<Factory> reg(FormatName, Priority); \
  return reg; \
} \
static auto &_##Factory##_registerer = RegisterDecoder_##Factory();

static constexpr float HighestDecoderPriority = 0;
/// @brief The decoder has a dedicated hardware unit and thus, is preferred
static constexpr float HWDecoderPriority = 10;
/// @brief The decoder runs on a GPU, utilizing normal CUDA cores
static constexpr float CUDADecoderPriority = 100;
/// @brief The decoder runs on the CPU
static constexpr float HostDecoderPriority = 1000;
/// @brief Use if nothing better is found
static constexpr float FallbackDecoderPriority = 1e+30;

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_REGISTRY_H_
