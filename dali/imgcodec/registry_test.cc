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

#include <gtest/gtest.h>
#include <typeinfo>
#include "dali/imgcodec/registry.h"
#include "dali/imgcodec/image_decoder_interfaces.h"

namespace dali {
namespace imgcodec {

TEST(DecoderRegistryTest, BasicTest) {
  for (const char *name : { "JPEG", "JPEG2000", "TIFF", "PNG", "BMP", "WebP" }) {
    const ImageFormat *f = ImageFormatRegistry::instance().GetImageFormat(name);
    EXPECT_NE(f, nullptr) << name << " format not found.";
    if (!f)
      continue;
    EXPECT_GT(f->Decoders().size(), 0) << "No decoder found format: " << name;
  }
}

TEST(DecoderRegistryTest, Fallback) {
  for (auto *format : ImageFormatRegistry::instance().Formats()) {
    bool found = false;
    for (auto *factory : format->Decoders()) {
      if (strstr(typeid(*factory).name(), "OpenCV")) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found) <<  "Fallback decoder not registered for a format";
  }
}

}  // namespace imgcodec
}  // namespace dali
