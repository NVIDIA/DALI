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

#include <atomic>
#include <memory>
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/image_decoder_interfaces.h"

#include "dali/imgcodec/parsers/bmp.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/imgcodec/parsers/jpeg2000.h"
#include "dali/imgcodec/parsers/png.h"
#include "dali/imgcodec/parsers/pnm.h"
#include "dali/imgcodec/parsers/tiff.h"
#include "dali/imgcodec/parsers/webp.h"

using std::shared_ptr;
using std::make_shared;

namespace dali {
namespace imgcodec {

template <typename Parser>
std::shared_ptr<ImageFormat> make_format(const char *name) {
  return make_shared<ImageFormat>(name, make_shared<Parser>());
}

void InitFormats(ImageFormatRegistry &reg) {
  shared_ptr<ImageFormat> format;

  reg.RegisterFormat(make_format<BmpParser>("BMP"));
  reg.RegisterFormat(make_format<JpegParser>("JPEG"));
  reg.RegisterFormat(make_format<Jpeg2000Parser>("JPEG2000"));
  reg.RegisterFormat(make_format<PngParser>("PNG"));
  reg.RegisterFormat(make_format<PnmParser>("PNM"));
  reg.RegisterFormat(make_format<TiffParser>("TIFF"));
  reg.RegisterFormat(make_format<WebpParser>("WebP"));
}

}  // namespace imgcodec
}  // namespace dali
