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

#include "dali/image/png.h"

#include <opencv2/opencv.hpp>

namespace dali {

bool CheckIsPNG(const uint8_t *png, int size) {
  // first bytes should be: 89 50 4E 47 0D 0A 1A 0A (hex)
  //                        137 80 78 71 13 10 26 10 (decimal)
  return (png[0] == 137 && png[1] == 80 && png[2] == 78 && png[3] == 71 &&
          png[4] == 13 && png[5] == 10 && png[6] == 26 && png[7] == 10);
}

// Assume chunk points to a 4-byte value
int ReadIntFromPNG(const uint8 *chunk) {
  // reverse the bytes, cast
  return (unsigned int)(chunk[0] << 24 | chunk[1] << 16 | chunk[2] << 8 | chunk[3]);
}

DALIError_t GetPNGImageDims(const uint8 *png, int size, int *h, int *w) {
  DALI_ASSERT(png);

  // IHDR needs to be the first chunk
  const uint8 *IHDR = png + 8;

  // Layout:
  // 4 bytes: chunk size (should be 13 bytes for IHDR)
  // 4 bytes: Chunk Identifier (should be "IHDR")
  // 4 bytes: Width
  // 4 bytes: Height
  // 1 byte : Bit Depth
  // 1 byte : Color Type
  // 1 byte : Compression method
  // 1 byte : Filter method
  // 1 byte : Interlace method
  *w = ReadIntFromPNG(IHDR + 8);
  *h = ReadIntFromPNG(IHDR + 12);

  return DALISuccess;
}

DALIError_t DecodePNGHost(const uint8_t *png, int size, DALIImageType image_type,
                          Tensor<CPUBackend>* output) {
  int h, w;
  int c = (image_type == DALI_GRAY) ? 1 : 3;
  // Get image size so we can pre-allocate output
  DALI_CALL(GetPNGImageDims(png, size, &h, &w));

#ifndef NDEBUG
  DALI_ASSERT(png != nullptr);
  DALI_ASSERT(size > 0);
  DALI_ASSERT(h > 0);
  DALI_ASSERT(w > 0);
  DALI_ASSERT(CheckIsPNG(png, size));
#endif

  // resize the output tensor
  output->Resize({h, w, c});
  // force allocation
  output->mutable_data<uint8>();

  cv::Mat dst(h, w, (c == 1) ? CV_8UC1: CV_8UC3,
              reinterpret_cast<char*>(output->raw_mutable_data()));

  // TODO(slayton): Work out why this doesn't work and dst isn't populated
  // with the result.
  cv::Mat tmp = cv::imdecode(
      cv::Mat(1, size, CV_8UC1, const_cast<char*>(reinterpret_cast<const char*>(png))),
      (c == 1) ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR,
      &dst);

  // if RGB needed, permute from BGR
  if (image_type == DALI_RGB) {
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
  }

  assert(tmp.data != nullptr);

  // See above. Shouldn't need this step
  std::memcpy(output->raw_mutable_data(), tmp.ptr(), tmp.rows * tmp.cols * c);

  return DALISuccess;
}

}  // namespace dali
