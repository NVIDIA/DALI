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

#include "dali/image/tiff_libtiff_impl.h"
#include <vector>
#include <string>
#include <cstring>
#include <utility>
#include <memory>
#include <tiffio.h>

#define LIBTIFF_CALL(call)                                \
  do {                                                    \
    int retcode = (call);                                 \
    if (0 == retcode) {                                   \
      DALI_FAIL("libtiff call failed with code "          \
        + std::to_string(retcode) + ": " #call);          \
    }                                                     \
  } while (0)

namespace dali {

namespace detail {

// Extracted and adjusted from OpenCV's modules/imgcodecs/src/grfmt_tiff.cpp
class BufDecoderHelper {
 private:
  span<uint8_t> &buf_;
  size_t &buf_pos_;

 public:
  BufDecoderHelper(span<uint8_t> &buf, size_t &buf_pos)
      : buf_(buf), buf_pos_(buf_pos) {}

  static tmsize_t read(thandle_t handle, void *buffer, tmsize_t n) {
    BufDecoderHelper *helper = reinterpret_cast<BufDecoderHelper *>(handle);
    span<uint8_t> &buf = helper->buf_;
    const tmsize_t size = buf.size();
    tmsize_t pos = helper->buf_pos_;
    if (n > (size - pos))
    {
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
    span<uint8_t> &buf = helper->buf_;
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
    span<uint8_t> &buf = helper->buf_;
    *base = buf.data();
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

} // namespace detail

LibtiffImpl::LibtiffImpl(span<uint8_t> buf)
    : buf_(std::move(buf)), buf_pos_(0) {
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

  LIBTIFF_CALL(
    TIFFGetField(tif_.get(), TIFFTAG_IMAGELENGTH, &shape_[0]));
  LIBTIFF_CALL(
    TIFFGetField(tif_.get(), TIFFTAG_IMAGEWIDTH, &shape_[1]));
  LIBTIFF_CALL(
    TIFFGetField(tif_.get(), TIFFTAG_SAMPLESPERPIXEL, &shape_[2]));
  is_tiled_ = static_cast<bool>(
    TIFFIsTiled(tif_.get()));
  LIBTIFF_CALL(
    TIFFGetField(tif_.get(), TIFFTAG_BITSPERSAMPLE, &bit_depth_));
  DALI_ENFORCE(bit_depth_ <= 64,
    "Unexpected bit depth: " + std::to_string(bit_depth_));

  // optional
  TIFFGetField(tif_.get(), TIFFTAG_ORIENTATION, &orientation_);
}

kernels::TensorShape<3> LibtiffImpl::Dims() const {
  return shape_;
}

bool LibtiffImpl::CanDecode() const {
  // TODO(janton): Implement to other variants
  std::cout << "is_tiled[" << is_tiled_ << "] bit_depth[" << bit_depth_ << "] orientation[" << orientation_ << "]" << std::endl;
  return !is_tiled_
      && bit_depth_ == 8
      && orientation_ == ORIENTATION_TOPLEFT;
}

std::pair<std::shared_ptr<uint8_t>, kernels::TensorShape<3>>
LibtiffImpl::Decode() const {
  DALI_ENFORCE(CanDecode(), "The image can't be decoded");

  // TODO(janton): Implement partial decoding
  auto crop_generator = false; // GetCropWindowGenerator();
  DALI_ENFORCE(!static_cast<bool>(crop_generator),
    "Partial decoding not implemented for TIFF images with more than 3 channels");

  // Will decode image dimensions if they were not yet read
  const size_t H = shape_[0], W = shape_[1], C = shape_[2];
  const size_t decoded_size = volume(shape_);

  //allocate memory for reading tif image
  std::unique_ptr<uint8_t, void(*)(void*)> row_buf{
    static_cast<uint8_t *>(_TIFFmalloc(decoded_size)), _TIFFfree};
  DALI_ENFORCE(row_buf.get() != nullptr, "Could not allocate memory");

  std::shared_ptr<uint8_t> decoded_img_ptr{
    new uint8_t[decoded_size],
    [](uint8_t* ptr){ delete [] ptr; }
  };

  const size_t row_stride = W * C;
  for (size_t y = 0; y < H; y++) {
    LIBTIFF_CALL(
      TIFFReadScanline(tif_.get(), row_buf.get(), y, 0));
    uint8_t *ptr = decoded_img_ptr.get() + y * row_stride;
    for (size_t x = 0; x < W; x++) {
      for (size_t c = 0; c < C; c++) {
        ptr[x*C + c] = row_buf.get()[x*C + c];
      }
    }
  }

  return std::make_pair(decoded_img_ptr, shape_);
}

}  // namespace dali
