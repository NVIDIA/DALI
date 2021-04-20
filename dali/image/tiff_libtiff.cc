// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

// ****************************************************************************************
//  A part of the file implements was extracted from OpenCV's
//  `modules/imgcodecs/src/grfmt_tiff.cpp`

//  OpenCV copyright notice:
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// ****************************************************************************************

#include "dali/image/tiff_libtiff.h"
#include <tiffio.h>
#include <cstring>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"
#include "dali/core/convert.h"
#include "dali/core/span.h"

#define LIBTIFF_CALL_SUCCESS 1
#define LIBTIFF_CALL(call)                                \
  do {                                                    \
    int retcode = (call);                                 \
    DALI_ENFORCE(LIBTIFF_CALL_SUCCESS == retcode,         \
      "libtiff call failed with code "                    \
      + std::to_string(retcode) + ": " #call);            \
  } while (0)

namespace dali {

namespace detail {

// Extracted and adjusted from OpenCV's modules/imgcodecs/src/grfmt_tiff.cpp
class BufDecoderHelper {
 private:
  span<const uint8_t> &buf_;
  size_t &buf_pos_;

 public:
  BufDecoderHelper(span<const uint8_t> &buf, size_t &buf_pos)
      : buf_(buf), buf_pos_(buf_pos) {}

  static tmsize_t read(thandle_t handle, void *buffer, tmsize_t n) {
    BufDecoderHelper *helper = reinterpret_cast<BufDecoderHelper *>(handle);
    auto &buf = helper->buf_;
    const tmsize_t size = buf.size();
    tmsize_t pos = helper->buf_pos_;
    if (n > (size - pos)) {
      n = size - pos;
    }
    memcpy(buffer, buf.data() + pos, n);
    helper->buf_pos_ += n;
    return n;
  }

  static tmsize_t write(thandle_t /*handle*/, void * /*buffer*/, tmsize_t /*n*/) {
    // Not used for decoding.
    return 0;
  }

  static toff_t seek(thandle_t handle, toff_t offset, int whence) {
    BufDecoderHelper *helper = reinterpret_cast<BufDecoderHelper *>(handle);
    auto &buf = helper->buf_;
    const toff_t size = buf.size();
    toff_t new_pos = helper->buf_pos_;
    switch (whence) {
    case SEEK_SET:
      new_pos = offset;
      break;
    case SEEK_CUR:
      new_pos += offset;
      break;
    case SEEK_END:
      new_pos = size + offset;
      break;
    }
    new_pos = std::min(new_pos, size);
    helper->buf_pos_ = static_cast<size_t>(new_pos);
    return new_pos;
  }

  static int map(thandle_t handle, void **base, toff_t *size) {
    BufDecoderHelper *helper = reinterpret_cast<BufDecoderHelper *>(handle);
    auto &buf = helper->buf_;
    *base = const_cast<uint8_t*>(buf.data());
    *size = buf.size();
    return 0;
  }

  static toff_t size(thandle_t handle) {
    BufDecoderHelper *helper = reinterpret_cast<BufDecoderHelper *>(handle);
    return helper->buf_.size();
  }

  static int close(thandle_t handle) {
    BufDecoderHelper *helper = reinterpret_cast<BufDecoderHelper *>(handle);
    delete helper;
    return 0;
  }
};

template <typename OutType, typename InType>
void ConvertLineFromRGBX(OutType *out_row, int64_t out_C, const InType *in_row, int64_t in_C,
                         int64_t roi_x, int64_t roi_w, DALIImageType out_img_type) {
  DALI_ENFORCE(in_C >= 3 || out_img_type == DALI_ANY_DATA);
  OutType * const out_row_end = out_row + roi_w * out_C;
  const InType *in = in_row + roi_x * in_C;
  OutType *out = out_row;

  if (out_img_type == DALI_ANY_DATA) {
    for (; out < out_row_end; out++, in++) {
      *out = ConvertSatNorm<OutType>(*in);
    }
  } else {
    for (; out < out_row_end; out += out_C, in += in_C) {
      vec<3, InType> rgb{in[0], in[1], in[2]};
      if (out_img_type == DALI_GRAY) {
        out[0] = kernels::color::rgb_to_gray<OutType>(rgb);
      } else if (out_img_type == DALI_YCbCr) {
        out[0] = kernels::color::itu_r_bt_601::rgb_to_y<OutType>(rgb);
        out[1] = kernels::color::itu_r_bt_601::rgb_to_cb<OutType>(rgb);
        out[2] = kernels::color::itu_r_bt_601::rgb_to_cr<OutType>(rgb);
      } else if (out_img_type == DALI_RGB) {
        out[0] = ConvertSatNorm<OutType>(rgb.x);
        out[1] = ConvertSatNorm<OutType>(rgb.y);
        out[2] = ConvertSatNorm<OutType>(rgb.z);
      } else if (out_img_type == DALI_BGR) {
        out[0] = ConvertSatNorm<OutType>(rgb.z);
        out[1] = ConvertSatNorm<OutType>(rgb.y);
        out[2] = ConvertSatNorm<OutType>(rgb.x);
      } else {
        DALI_FAIL("Image type not supported" + std::to_string(out_img_type));
      }
    }
  }
}

template <typename OutType, typename InType>
void ConvertLineFromMonochrome(OutType *out_row, int64_t out_C, const InType *in_row, int64_t in_C,
                               int64_t roi_x, int64_t roi_w, DALIImageType out_img_type) {
  DALI_ENFORCE(in_C == 1);
  OutType * const out_row_end = out_row + roi_w * out_C;
  const InType *in = in_row + roi_x * in_C;
  OutType *out = out_row;
  for (; out < out_row_end; out += out_C, in += in_C) {
    if (out_img_type == DALI_GRAY) {
      out[0] = ConvertSatNorm<OutType>(in[0]);
    } else if (out_img_type == DALI_YCbCr) {
      out[0] = ConvertSatNorm<OutType>(in[0] * 0.859f + 16.0f / 256.0f);
      out[1] = out[2] = ConvertNorm<OutType>(0.5f);
    } else if (out_img_type == DALI_RGB || out_img_type == DALI_BGR) {
      out[0] = out[1] = out[2] = ConvertSatNorm<OutType>(in[0]);
    } else {  // DALI_ANY_DATA
      const auto value = ConvertSatNorm<OutType>(in[0]);
      for (int64_t c = 0; c < out_C; c++) {
        out[c] = value;
      }
    }
  }
}

template <typename OutType, typename InType>
void ConvertLine(OutType *out_row, int64_t out_C, const InType *in_row, int64_t in_C,
                 int64_t roi_x, int64_t roi_w, DALIImageType out_img_type) {
  if (in_C == 1) {
    return ConvertLineFromMonochrome(out_row, out_C, in_row, in_C, roi_x, roi_w, out_img_type);
  } else {
    return ConvertLineFromRGBX(out_row, out_C, in_row, in_C, roi_x, roi_w, out_img_type);
  }
}

}  // namespace detail

TiffImage_Libtiff::TiffImage_Libtiff(const uint8_t *encoded_buffer,
                                     size_t length,
                                     DALIImageType image_type)
    : GenericImage(encoded_buffer, length, image_type),
      buf_({encoded_buffer, static_cast<ptrdiff_t>(length)}),
      buf_pos_(0) {
  tif_.reset(
    TIFFClientOpen("", "r",
                   reinterpret_cast<thandle_t>(
                     new detail::BufDecoderHelper(buf_, buf_pos_)),
                   &detail::BufDecoderHelper::read,
                   &detail::BufDecoderHelper::write,
                   &detail::BufDecoderHelper::seek,
                   &detail::BufDecoderHelper::close,
                   &detail::BufDecoderHelper::size,
                   &detail::BufDecoderHelper::map,
                   /*unmap=*/0));
  DALI_ENFORCE(tif_, "Cannot open TIFF file.");

  LIBTIFF_CALL(
    TIFFGetField(tif_.get(), TIFFTAG_IMAGELENGTH, &shape_[0]));
  LIBTIFF_CALL(
    TIFFGetField(tif_.get(), TIFFTAG_IMAGEWIDTH, &shape_[1]));
  LIBTIFF_CALL(
    TIFFGetFieldDefaulted(tif_.get(), TIFFTAG_SAMPLESPERPIXEL, &shape_[2]));
  is_tiled_ = static_cast<bool>(
    TIFFIsTiled(tif_.get()));
  LIBTIFF_CALL(
    TIFFGetFieldDefaulted(tif_.get(), TIFFTAG_BITSPERSAMPLE, &bit_depth_));
  DALI_ENFORCE(bit_depth_ <= 64,
    "Unexpected bit depth: " + std::to_string(bit_depth_));
  LIBTIFF_CALL(
    TIFFGetFieldDefaulted(tif_.get(), TIFFTAG_ORIENTATION, &orientation_));
  LIBTIFF_CALL(
    TIFFGetFieldDefaulted(tif_.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip_));
  LIBTIFF_CALL(
    TIFFGetFieldDefaulted(tif_.get(), TIFFTAG_COMPRESSION, &compression_));
}

Image::Shape TiffImage_Libtiff::PeekShapeImpl(const uint8_t *encoded_buffer,
                                              size_t length) const {
  DALI_ENFORCE(encoded_buffer != nullptr);
  assert(encoded_buffer == buf_.data());
  return shape_;
}

std::pair<std::shared_ptr<uint8_t>, Image::Shape>
TiffImage_Libtiff::DecodeImpl(DALIImageType image_type,
                              const uint8 *encoded_buffer,
                              size_t length) const {
  // This decoder only handles bitdepth=8, non-tiled and top-left orientation
  // Other cases go to OpenCV's based decoder
  if (!CanDecode(image_type)) {
    return GenericImage::DecodeImpl(image_type, encoded_buffer, length);
  }
  const int64_t H = shape_[0], W = shape_[1], C = shape_[2];
  auto roi_generator = GetCropWindowGenerator();

  int64_t roi_x = 0, roi_y = 0;
  int64_t roi_h = H, roi_w = W;
  if (roi_generator) {
    auto roi = roi_generator({H, W}, "HW");
    roi_y = roi.anchor[0];
    roi_x = roi.anchor[1];
    roi_h = roi.shape[0];
    roi_w = roi.shape[1];
    DALI_ENFORCE(roi_w > 0 && roi_w <= W);
    DALI_ENFORCE(roi_h > 0 && roi_h <= H);
  }

  int64_t out_C = C;
  switch (image_type) {
    case DALI_GRAY:
      out_C = 1;
      break;
    case DALI_RGB:
    case DALI_BGR:
    case DALI_YCbCr:
      out_C = 3;
      break;
    case DALI_ANY_DATA:
    default:
      out_C = C;
  }

  TensorShape<3> decoded_shape = {roi_h, roi_w, out_C};
  const size_t decoded_size = volume(decoded_shape);
  std::shared_ptr<uint8_t> decoded_img_ptr{
    new uint8_t[decoded_size],
    [](uint8_t* ptr){ delete [] ptr; }
  };

  // TODO(janton): support different types in ImageDecoder
  using InType = uint8_t;
  using OutType = uint8_t;

  // allocate memory for reading tif image
  auto row_nbytes = TIFFScanlineSize(tif_.get());
  DALI_ENFORCE(row_nbytes > 0);

  std::unique_ptr<InType, void(*)(void*)> row_buf{
    static_cast<InType *>(_TIFFmalloc(row_nbytes)), _TIFFfree};
  DALI_ENFORCE(row_buf.get() != nullptr, "Could not allocate memory");
  memset(row_buf.get(), 0, row_nbytes);


  const int64_t out_row_stride = roi_w * out_C;
  InType * const row_in  = row_buf.get();
  OutType * const img_out = decoded_img_ptr.get();

  // Need to read sequentially since not all the images support random access

  // From: http://www.libtiff.org/man/TIFFReadScanline.3t.html
  // Compression algorithm does not support random access. Data was requested in a non-sequential
  // order from a file that uses a compression algorithm and that has RowsPerStrip greater than one.
  // That is, data in the image is stored in a compressed form, and with multiple rows packed into a
  // strip. In this case, the library does not support random access to the data. The data should
  // either be accessed sequentially, or the file should be converted so that each strip is made up
  // of one row of data.
  const bool allow_random_row_access =
    (compression_ == COMPRESSION_NONE || rows_per_strip_ == 1);

  // If random access is not allowed, need to read sequentially all previous rows
  if (!allow_random_row_access) {
    for (int64_t y = 0; y < roi_y; y++) {
      LIBTIFF_CALL(
        TIFFReadScanline(tif_.get(), row_in, y, 0));
    }
  }

  for (int64_t y = 0; y < roi_h; y++) {
    LIBTIFF_CALL(
      TIFFReadScanline(tif_.get(), row_in, roi_y + y, 0));
    OutType * const row_out = img_out + (y * out_row_stride);
    detail::ConvertLine(row_out, out_C, row_in, C, roi_x, roi_w, image_type);
  }

  return {decoded_img_ptr, decoded_shape};
}

bool TiffImage_Libtiff::CanDecode(DALIImageType image_type) const {
  return !is_tiled_
      && bit_depth_ == 8
      && orientation_ == ORIENTATION_TOPLEFT;
}

}  // namespace dali
