// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/decoder/image/image_factory.h"
#include "dali/operators/decoder/image/generic_image.h"
#include "dali/operators/decoder/image/png.h"
#include "dali/operators/decoder/image/bmp.h"
#include "dali/operators/decoder/image/jpeg.h"
#include "dali/operators/decoder/image/jpeg2k.h"
#if LIBTIFF_ENABLED
#include "dali/operators/decoder/image/tiff_libtiff.h"
#else
#include "dali/operators/decoder/image/tiff.h"
#endif
#include "dali/operators/decoder/image/pnm.h"
#include "dali/operators/decoder/image/webp.h"

namespace dali {

namespace {

bool CheckIsJPEG(const uint8_t *jpeg, int) {
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
  bool is_png    = CheckIsPNG(encoded_image, length);
  bool is_bmp    = CheckIsBMP(encoded_image, length);
  bool is_jpeg   = CheckIsJPEG(encoded_image, length);
  bool is_tiff   = CheckIsTiff(encoded_image, length);
  bool is_pnm    = CheckIsPNM(encoded_image, length);
  bool is_jpeg2k = CheckIsJPEG2k(encoded_image, length);
  bool is_webp   = CheckIsWebP(encoded_image, length);

  int matches = is_png + is_bmp + is_jpeg + is_tiff + is_pnm + is_jpeg2k + is_webp;
  if (matches == 0) {
    if (CheckIsGIF(encoded_image, length)) {
      DALI_FAIL("GIF images are not supported.");
    } else {
      DALI_FAIL(
          "Unrecognized image format. Supported formats are: JPEG, PNG, BMP, TIFF, PNM, JPEG2000 "
          "and WebP.");
    }
  } else if (matches > 1) {
    DALI_FAIL("Ambiguous image format. The header matches more than one image format.");
  }

  if (is_png) {
    return std::make_unique<PngImage>(encoded_image, length, image_type);
  } else if (is_jpeg2k) {
    return std::make_unique<Jpeg2kImage>(encoded_image, length, image_type);
  } else if (is_jpeg) {
    return std::make_unique<JpegImage>(encoded_image, length, image_type);
  } else if (is_bmp) {
    return std::make_unique<BmpImage>(encoded_image, length, image_type);
  } else if (is_pnm) {
    return std::make_unique<PnmImage>(encoded_image, length, image_type);
  } else if (is_tiff) {
#if LIBTIFF_ENABLED
    return std::make_unique<TiffImage_Libtiff>(encoded_image, length, image_type);
#else
    return std::make_unique<TiffImage>(encoded_image, length, image_type);
#endif
  } else if (is_webp) {
    return std::make_unique<WebpImage>(encoded_image, length, image_type);
  }
  return std::make_unique<GenericImage>(encoded_image, length, image_type);
}

}  // namespace dali
