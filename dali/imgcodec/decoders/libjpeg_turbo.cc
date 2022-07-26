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

#include "dali/imgcodec/decoders/libjpeg_turbo.h"
#include "dali/imgcodec/decoders/jpeg/jpeg_mem.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/imgcodec/util/convert.h"

namespace dali {
namespace imgcodec {

DecodeResult LibJpegTurboDecoderInstance::Decode(SampleView<CPUBackend> out,
                                                 ImageSource *in,
                                                 DecodeParams opts) {
  jpeg::UncompressFlags flags;

  auto &type = opts.format;
  auto info = JpegParser().GetInfo(in);

  flags.components = info.shape[2];
  if (type == DALI_ANY_DATA)
    type = info.shape[2] == 3 ? DALI_RGB : DALI_GRAY;

  DALI_ENFORCE(type == DALI_RGB || type == DALI_BGR || type == DALI_GRAY,
               "Color space not supported by libjpeg-turbo");
  flags.color_space = type;

  if (any_cast<bool>(GetParam("fast_idct"))) {
    flags.dct_method = JDCT_FASTEST;
  }

  if ((flags.crop = opts.use_roi)) {
    flags.crop_y = opts.roi.begin[0];
    flags.crop_x = opts.roi.begin[1];
    flags.crop_height = opts.roi.shape()[0];
    flags.crop_width = opts.roi.shape()[1];
  }

  const uint8_t *encoded_data;
  size_t data_size;
  if (in->Kind() == InputKind::HostMemory) {
    encoded_data = in->RawData<uint8_t>();
    data_size = in->Size();
  } else {
    DALI_FAIL("InputKind not supported");
  }

  DecodeResult res;
  try {
    std::shared_ptr<uint8_t> decoded_image;
    uint8_t* result = jpeg::Uncompress(
      encoded_data, data_size, flags, nullptr /* nwarn */,
      [&decoded_image, &info](int width, int height, int channels) {
        decoded_image.reset(
          new uint8_t[height * width * channels],
          [](uint8_t* data) { delete [] data; });
        info.shape[0] = height;
        info.shape[1] = width;
        return decoded_image.get();
      });

    if ((res.success = result != nullptr)) {
      SampleView<CPUBackend> in(decoded_image.get(), info.shape, opts.dtype);
      TensorLayout layout = "HWC";
      Convert(out, layout, type, in, layout, type, {}, {});
    }
  } catch (...) {
    res.exception = std::current_exception();
    res.success = false;
  }

  return res;
}

}  // namespace imgcodec
}  // namespace dali
