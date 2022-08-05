/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Copyright 2019, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file defines functions to compress and uncompress JPEG files
// to and from memory.  It provides interfaces for raw images
// (data array and size fields).
// Direct manipulation of JPEG strings are supplied: Flip, Rotate, Crop..

#ifndef DALI_IMGCODEC_DECODERS_JPEG_JPEG_MEM_H_
#define DALI_IMGCODEC_DECODERS_JPEG_JPEG_MEM_H_

#include <functional>
#include <memory>
#include <string>
#include "dali/core/common.h"
#include "dali/imgcodec/decoders/jpeg/jpeg_utils.h"

namespace dali {
namespace imgcodec {
namespace jpeg {

// Flags for Uncompress
struct UncompressFlags {
  // ratio can be 1, 2, 4, or 8 and represent the denominator for the scaling
  // factor (eg ratio = 4 means that the resulting image will be at 1/4 original
  // size in both directions).
  int ratio = 1;

  // The number of bytes per pixel (1, 3 or 4), or 0 for autodetect.
  int components = 0;

  // If true, decoder will use a slower but nicer upscaling of the chroma
  // planes (yuv420/422 only).
  bool fancy_upscaling = true;

  // If true, will attempt to fill in missing lines of truncated files
  bool try_recover_truncated_jpeg = false;

  // The minimum required fraction of lines read before the image is accepted.
  float min_acceptable_fraction = 1.0;

  // The distance in bytes from one scanline to the other.  Should be at least
  // equal to width*components*sizeof(JSAMPLE).  If 0 is passed, the stride
  // used will be this minimal value.
  int stride = 0;

  // Setting of J_DCT_METHOD enum in jpeglib.h, for choosing which
  // algorithm to use for DCT/IDCT.
  //
  // Setting this has a quality/speed trade-off implication.
  J_DCT_METHOD dct_method = JDCT_DEFAULT;

  // Output color space
  DALIImageType color_space = DALI_RGB;

  // Settings of crop window before decompression.
  bool crop = false;
  // Vertical coordinate of the top-left corner of the result in the input.
  int crop_x = 0;
  // Horizontal coordinate of the top-left corner of the result in the input.
  int crop_y = 0;
  // Width of the output image.
  int crop_width = 0;
  // Height of the output image.
  int crop_height = 0;
};

// Uncompress some raw JPEG data given by the pointer srcdata and the length
// datasize.
// The function returns a shared pointer to the uncompressed data or a null pointer if
// there was an error.
std::unique_ptr<uint8[]> Uncompress(const void* srcdata, int datasize,
                                    const UncompressFlags& flags);

// Read jpeg header and get image information.  Returns true on success.
// The width, height, and components points may be null.
bool GetImageInfo(const void* srcdata, int datasize, int* width, int* height,
                  int* components);

// Note: (format & 0xff) = number of components (<=> bytes per pixels)
enum Format {
  FORMAT_GRAYSCALE = 0x001,  // 1 byte/pixel
  FORMAT_RGB = 0x003,        // 3 bytes/pixel RGBRGBRGBRGB...
  FORMAT_RGBA = 0x004,       // 4 bytes/pixel RGBARGBARGBARGBA...
  FORMAT_ABGR = 0x104        // 4 bytes/pixel ABGRABGRABGR...
};

// Flags for compression
struct CompressFlags {
  // Encoding of the input data for compression
  Format format;

  // Quality of the compression from 0-100
  int quality = 95;

  // If true, create a jpeg image that loads progressively
  bool progressive = false;

  // If true, reduce jpeg size without changing quality (at the cost of CPU/RAM)
  bool optimize_jpeg_size = false;

  // See http://en.wikipedia.org/wiki/Chroma_subsampling
  bool chroma_downsampling = true;

  // Resolution
  int density_unit = 1;  // 1 = in, 2 = cm
  int x_density = 300;
  int y_density = 300;

  // If not empty, embed this XMP metadata in the image header
  std::string xmp_metadata;

  // The distance in bytes from one scanline to the other.  Should be at least
  // equal to width*components*sizeof(JSAMPLE).  If 0 is passed, the stride
  // used will be this minimal value.
  int stride = 0;
};

}  // namespace jpeg
}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_JPEG_JPEG_MEM_H_
