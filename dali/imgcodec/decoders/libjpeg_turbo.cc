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

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "dali/imgcodec/decoders/opencv_fallback.h"
#include "dali/imgcodec/decoders/libjpeg_turbo.h"
#include "dali/imgcodec/util/convert.h"
#include "dali/image/jpeg_mem.h"

namespace dali {
namespace imgcodec {

DecodeResult LibJpegTurboDecoderInstance::Decode(SampleView<CPUBackend> out,
                                                 ImageSource *in,
                                                 DecodeParams opts) {
  jpeg::UncompressFlags flags;

  if (false) {  // ???
    flags.dct_method = JDCT_FASTEST;
  }

  flags.components = 0;  // autodetect -- ???

  const auto &type = opts.format;
  DALI_ENFORCE(type == DALI_RGB || type == DALI_BGR || type == DALI_GRAY,
               "Color space not supported by libjpeg-turbo");
  flags.color_space = type;

  if ((flags.crop = opts.use_roi)) {
    flags.crop_y = opts.roi.begin[0];
    flags.crop_x = opts.roi.begin[1];
    flags.crop_height = opts.roi.shape()[0];
    flags.crop_width = opts.roi.shape()[1];
  }

  const uint8_t *encoded_data;
  size_t data_size;
  if (in->Kind() == InputKind::HostMemory) {
    encoded_data = static_cast<const uint8_t*>(in->RawData());
    data_size = in->Size();
  } else {
    // ???
  }

  std::shared_ptr<uint8_t> decoded_image;
  int cropped_h = 0;
  int cropped_w = 0;
  uint8_t* result = jpeg::Uncompress(
    encoded_data, data_size, flags, nullptr /* nwarn */, 
    [&decoded_image, &cropped_h, &cropped_w](int width, int height, int channels) {
      decoded_image.reset(
        new uint8_t[height * width * channels],
        [](uint8_t* data) { delete [] data; });
      cropped_h = height;
      cropped_w = width;
      return decoded_image.get();
    }
  );

  DecodeResult res;
  res.success = result != nullptr;
  res.exception = nullptr;
  return res;
}


}  // namespace imgcodec
}  // namespace dali
