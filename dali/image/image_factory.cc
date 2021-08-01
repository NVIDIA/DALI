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

#include "dali/image/image_factory.h"
#include "dali/image/generic_image.h"
#include "dali/image/png.h"
#include "dali/image/bmp.h"
#include "dali/image/jpeg.h"
#include "dali/image/jpeg2k.h"
#if LIBTIFF_ENABLED
#include "dali/image/tiff_libtiff.h"
#else
#include "dali/image/tiff.h"
#endif
#include "dali/image/pnm.h"
#include "dali/image/webp.h"

namespace dali {

namespace {

bool CheckIsJPEG(const uint8 *jpeg, int) {
  DALI_ENFORCE(jpeg);
  return (jpeg[0] == 255) && (jpeg[1] == 216);
}

bool CheckIsPNG(const uint8_t *png, int size) {
  DALI_ENFORCE(png);
  // first bytes should be: 89 50 4E 47 0D 0A 1A 0A (hex)
  //                        137 80 78 71 13 10 26 10 (decimal)
  return (size >= 8 && png[0] == 137 && png[1] == 80 && png[2] == 78 && png[3] == 71 &&
          png[4] == 13 && png[5] == 10 && png[6] == 26 && png[7] == 10);
}

bool CheckIsGIF(const uint8_t *gif, int size) {
  DALI_ENFORCE(gif);
  return (size >= 10 && gif[0] == 'G' && gif[1] == 'I' && gif[2] == 'F' && gif[3] == '8' &&
          (gif[4] == '7' || gif[4] == '9') && gif[5] == 'a');
}

bool CheckIsBMP(const uint8_t *bmp, int size) {
  return (size > 2 && bmp[0] == 'B' && bmp[1] == 'M');
}

bool CheckIsPNM(const uint8_t *pnm, int size) {
  return (size > 2 && pnm[0] == 'P' && pnm[1] >= '1' && pnm[1] <= '6');
}

bool CheckIsWebP(const uint8_t *webp, int size) {
  DALI_ENFORCE(webp);
  return (size > 12 && webp[0] == 'R' && webp[1] == 'I' && webp[2] == 'F' && webp[3] == 'F' &&
          webp[8] == 'W' && webp[9] == 'E' && webp[10] == 'B' && webp[11] == 'P');
}

constexpr std::array<int, 4> header_intel = {77, 77, 0, 42};
constexpr std::array<int, 4> header_motorola = {73, 73, 42, 0};

bool CheckIsTiff(const uint8_t *tiff, int size) {
  DALI_ENFORCE(tiff);
  auto check_header = [](const unsigned char *tiff_buf, const std::array<int, 4> &header) -> bool {
      DALI_ENFORCE(tiff_buf);
      for (unsigned int i = 0; i < header.size(); i++) {
        if (tiff_buf[i] != header[i]) {
          return false;
        }
      }
      return true;
  };
  return check_header(tiff, header_intel) || check_header(tiff, header_motorola);
}

}  // namespace

std::unique_ptr<Image>
ImageFactory::CreateImage(const uint8_t *encoded_image, size_t length, DALIImageType image_type) {
  DALI_ENFORCE(CheckIsPNG(encoded_image, length) + CheckIsBMP(encoded_image, length) +
               CheckIsGIF(encoded_image, length) + CheckIsJPEG(encoded_image, length) +
               CheckIsTiff(encoded_image, length) + CheckIsPNM(encoded_image, length) +
               CheckIsJPEG2k(encoded_image, length) + CheckIsWebP(encoded_image, length) == 1,
               "Encoded image has ambiguous format");
  if (CheckIsPNG(encoded_image, length)) {
    return std::make_unique<PngImage>(encoded_image, length, image_type);
  } else if (CheckIsJPEG2k(encoded_image, length)) {
    return std::make_unique<Jpeg2kImage>(encoded_image, length, image_type);
  } else if (CheckIsJPEG(encoded_image, length)) {
    return std::make_unique<JpegImage>(encoded_image, length, image_type);
  } else if (CheckIsBMP(encoded_image, length)) {
    return std::make_unique<BmpImage>(encoded_image, length, image_type);
  } else if (CheckIsPNM(encoded_image, length)) {
    return std::make_unique<PnmImage>(encoded_image, length, image_type);
  } else if (CheckIsGIF(encoded_image, length)) {
    DALI_FAIL("GIF format is not supported");
  } else if (CheckIsTiff(encoded_image, length)) {
#if LIBTIFF_ENABLED
    return std::make_unique<TiffImage_Libtiff>(encoded_image, length, image_type);
#else
    return std::make_unique<TiffImage>(encoded_image, length, image_type);
#endif
  } else if (CheckIsWebP(encoded_image, length)) {
    return std::make_unique<WebpImage>(encoded_image, length, image_type);
  }
  return std::make_unique<GenericImage>(encoded_image, length, image_type);
}

}  // namespace dali
