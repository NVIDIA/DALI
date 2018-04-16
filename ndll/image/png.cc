// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/image/png.h"

#include <opencv2/opencv.hpp>

namespace ndll {

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

NDLLError_t GetPNGImageDims(const uint8 *png, int size, int *h, int *w) {
  NDLL_ASSERT(png);

  // IHDR needs to be the first chunk
  const uint8 *IHDR = &png[8];

  // Layout:
  // 4 bytes: chunk size (should be 13 bytes for IHDR)
  // 4 bytes: Chunk Identifier (should be "IHDR")
  // 4 bytes: Height
  // 4 bytes: Width
  // 1 byte : Bit Depth
  // 1 byte : Color Type
  // 1 byte : Compression method
  // 1 byte : Filter method
  // 1 byte : Interlace method
  *h = ReadIntFromPNG(&IHDR[8]);
  *w = ReadIntFromPNG(&IHDR[12]);

  return NDLLSuccess;
}

NDLLError_t DecodePNGHost(const uint8_t *png, int size, NDLLImageType image_type,
                          Tensor<CPUBackend>* output) {
  int h, w;
  int c = (image_type == NDLL_GRAY) ? 1 : 3;
  // Get image size so we can pre-allocate output
  NDLL_CALL(GetPNGImageDims(png, size, &h, &w));

#ifndef NDEBUG
  NDLL_ASSERT(png != nullptr);
  NDLL_ASSERT(size > 0);
  NDLL_ASSERT(h > 0);
  NDLL_ASSERT(w > 0);
  NDLL_ASSERT(CheckIsPNG(png, size));
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
  if (image_type == NDLL_RGB) {
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
  }

  assert(tmp.data != nullptr);

  // See above. Shouldn't need this step
  std::memcpy(output->raw_mutable_data(), tmp.ptr(), tmp.rows * tmp.cols * c);

  return NDLLSuccess;
}

}  // namespace ndll
