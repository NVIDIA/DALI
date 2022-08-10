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
#include "dali/core/common.h"

namespace dali {
namespace imgcodec {

DecodeResult LibJpegTurboDecoderInstance::Decode(SampleView<CPUBackend> out,
                                                 ImageSource *in,
                                                 DecodeParams opts,
                                                 const ROI &roi) {
  jpeg::UncompressFlags flags;

  auto &out_type = opts.format;
  auto info = JpegParser{}.GetInfo(in);
  auto target_shape = info.shape;

  if (out_type == DALI_ANY_DATA) {
    flags.color_space = out_type = info.shape[2] == 3 ? DALI_RGB : DALI_GRAY;
  } else if (out_type == DALI_YCbCr) {
    flags.color_space = DALI_RGB;
  } else {
    assert(out_type == DALI_RGB || out_type == DALI_BGR || out_type == DALI_GRAY);
    flags.color_space = out_type;
  }

  flags.components = info.shape[2];
  target_shape[2] = NumberOfChannels(out_type);

  if (any_cast<bool>(GetParam("fast_idct"))) {
    flags.dct_method = JDCT_FASTEST;
  }

  if (roi) {
    flags.crop = true;
    flags.crop_y = roi.begin[0];
    flags.crop_x = roi.begin[1];
    flags.crop_height = target_shape[0] = roi.shape()[0];
    flags.crop_width  = target_shape[1] = roi.shape()[1];
  }

  const uint8_t *encoded_data;
  size_t data_size;
  if (in->Kind() == InputKind::HostMemory) {
    encoded_data = in->RawData<uint8_t>();
    data_size = in->Size();
  } else {
    DALI_FAIL(make_string("InputKind not supported: ", static_cast<int>(in->Kind())));
  }

  DecodeResult res;
  try {
    auto decoded_image = jpeg::Uncompress(encoded_data, data_size, flags);
    if ((res.success = decoded_image != nullptr)) {
      // JPEG images are always 8-bit, in HWC format
      SampleView<CPUBackend> in(decoded_image.get(), target_shape, DALI_UINT8);
      TensorLayout layout = "HWC";
      Convert(out, layout, out_type, in, layout, flags.color_space, {}, {});
    }
  } catch (...) {
    res.exception = std::current_exception();
    res.success = false;
  }

  return res;
}

}  // namespace imgcodec
}  // namespace dali
