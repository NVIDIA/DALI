/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Copyright 2019, 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

// This file defines functions to compress and uncompress JPEG data
// to and from memory, as well as some direct manipulations of JPEG string

#include "dali/operators/decoder/jpeg/jpeg_mem.h"
#include <setjmp.h>
#include <cstring>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "dali/operators/decoder/jpeg/jpeg_handle.h"
#include "dali/core/error_handling.h"
#include "dali/core/call_at_exit.h"

namespace dali {
namespace jpeg {

// -----------------------------------------------------------------------------
// Decompression

namespace {

enum JPEGErrors {
  JPEGERRORS_OK,
  JPEGERRORS_UNEXPECTED_END_OF_DATA,
  JPEGERRORS_BAD_PARAM
};

// Prevent bad compiler behavior in ASAN mode by wrapping most of the
// arguments in a struct struct.
class FewerArgsForCompiler {
 public:
  FewerArgsForCompiler(int datasize, const UncompressFlags& flags)
      : datasize_(datasize),
        flags_(flags),
        height_read_(0),
        height_(0),
        stride_(0) {
  }

  const int datasize_;
  const UncompressFlags flags_;
  int height_read_;  // number of scanline lines successfully read
  int height_;
  int stride_;
};

// Check whether the crop window is valid, assuming crop is true.
bool IsCropWindowValid(const UncompressFlags& flags, int input_image_width,
                       int input_image_height) {
  // Crop window is valid only if it is non zero and all the window region is
  // within the original image.
  return flags.crop_width > 0 && flags.crop_height > 0 && flags.crop_x >= 0 &&
         flags.crop_y >= 0 &&
         flags.crop_y + flags.crop_height <= input_image_height &&
         flags.crop_x + flags.crop_width <= input_image_width;
}

std::unique_ptr<uint8_t[]> UncompressLow(const void* srcdata, FewerArgsForCompiler* argball) {
  // unpack the argball
  const int datasize = argball->datasize_;
  const auto& flags = argball->flags_;
  const int ratio = flags.ratio;
  int components = flags.components;
  int stride = flags.stride;              // may be 0
  auto color_space = flags.color_space;

  // Can't decode if the ratio is not recognized by libjpeg
  if ((ratio != 1) && (ratio != 2) && (ratio != 4) && (ratio != 8)) {
    return nullptr;
  }

  // Channels must be autodetect, grayscale, or rgb.
  if (!(components == 0 || components == 1 || components == 3)) {
    return nullptr;
  }

  // if empty image, return
  if (datasize == 0 || srcdata == nullptr) return nullptr;

  // Declare buffers here so that we can free on error paths
  std::unique_ptr<JSAMPLE[]> temp;
  std::unique_ptr<uint8_t[]> dstdata;
  JSAMPLE *tempdata = nullptr;

  // Initialize libjpeg structures to have a memory source
  // Modify the usual jpeg error manager to catch fatal errors.
  JPEGErrors error = JPEGERRORS_OK;
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  struct ProgressMgr progress{};
  cinfo.err = jpeg_std_error(&jerr);
  jmp_buf jpeg_jmpbuf;
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = CatchError;
  if (setjmp(jpeg_jmpbuf)) {
    // For progressive scan failure, error out with hard error not to let the
    // opencv fallback to kick-in
    DALI_ENFORCE(
        progress.scans_exceeded == 0,
        make_string(
            "The number of scans (", progress.scans_exceeded,
            ") during progressive decoding of the image exceeded ",
            "the currently supported value of ", progress.max_scans,
            ". If that's intentional, you can increase the limit ",
            "of progressive scans by setting the environmental variable "
            "`DALI_MAX_JPEG_SCANS` to the desired limit."));
    return nullptr;
  }

  jpeg_create_decompress(&cinfo);
  auto destroy_cinfo = AtScopeExit([&cinfo]() {
    jpeg_destroy_decompress(&cinfo);
  });
  // The progress mgr must be set after the call to the
  // `jpeg_create_decompress` which overwrites/initializes
  // the `cinfo.progress` to NULL.
  SetupProgressMgr(&cinfo, &progress);

  SetSrc(&cinfo, srcdata, datasize, flags.try_recover_truncated_jpeg);
  jpeg_read_header(&cinfo, TRUE);

  // Set components automatically if desired, autoconverting cmyk to rgb.
  if (components == 0) components = std::min(cinfo.num_components, 3);

  // set grayscale and ratio parameters
  switch (components) {
    case 1:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
    case 3:
      cinfo.out_color_space = color_space == DALI_BGR ?
        JCS_EXT_BGR : JCS_RGB;
      break;
    default:
      ERROR_LOG << " Invalid components value " << components << std::endl;
      return nullptr;
  }

  if (cinfo.jpeg_color_space == JCS_CMYK ||
      cinfo.jpeg_color_space == JCS_YCCK) {
    // Always use cmyk for output in a 4 channel jpeg. libjpeg has a builtin
    // decoder.  We will further convert to rgb/bgr/gray below.
    cinfo.out_color_space = JCS_CMYK;
  }

  cinfo.do_fancy_upsampling = boolean(flags.fancy_upscaling);
  cinfo.scale_num = 1;
  cinfo.scale_denom = ratio;
  cinfo.dct_method = flags.dct_method;

  // Determine the output image size before attempting decompress to prevent
  // OOM'ing doing the decompress
  jpeg_calc_output_dimensions(&cinfo);

  int64_t total_size = static_cast<int64_t>(cinfo.output_height) *
                     static_cast<int64_t>(cinfo.output_width) *
                     static_cast<int64_t>(cinfo.num_components);
  // Some of the internal routines do not gracefully handle ridiculously
  // large images, so fail fast.
  if (cinfo.output_width <= 0 || cinfo.output_height <= 0) {
    ERROR_LOG << "Invalid image size: " << cinfo.output_width << " x "
              << cinfo.output_height << std::endl;
    return nullptr;
  }
  if (total_size >= (1LL << 29)) {
    ERROR_LOG << "Image too large: " << total_size << std::endl;
    return nullptr;
  }

  DALI_ENFORCE(jpeg_start_decompress(&cinfo));

  JDIMENSION target_output_width = cinfo.output_width;
  JDIMENSION target_output_height = cinfo.output_height;
  JDIMENSION skipped_scanlines = 0;
  int left_cor = 0;
  int right_cor = 0;
#if defined(LIBJPEG_TURBO_VERSION)
  if (flags.crop) {
    // Update target output height and width based on crop window.
    target_output_height = flags.crop_height;
    target_output_width = flags.crop_width;

    // So far, cinfo holds the original input image information.
    if (!IsCropWindowValid(flags, cinfo.output_width, cinfo.output_height)) {
      ERROR_LOG << "Invalid crop window: x=" << flags.crop_x
                << ", y=" << flags.crop_y << ", w=" << target_output_width
                << ", h=" << target_output_height
                << " for image_width: " << cinfo.output_width
                << " and image_height: " << cinfo.output_height
                << std::endl;
      return nullptr;
    }

    // We are croping one pixel more left and right so pixel on the requested edge
    // won't be intepolated by will use value of spare pixel we asked for
    left_cor = flags.crop_x == 0 ? 0 : 1;
    int r_bound = flags.crop_x + flags.crop_width;
    right_cor = std::max(0, std::min(1, static_cast<int>(cinfo.output_width - r_bound)));

    // Update cinfo.output_width. It is tricky that cinfo.output_width must
    // fall on an Minimum Coded Unit (MCU) boundary; if it doesn't, then it will
    // be moved left to the nearest MCU boundary, and width will be increased
    // accordingly. Therefore, the final cinfo.crop_width might differ from the
    // given flags.crop_width. Please see libjpeg library for details.
    JDIMENSION crop_width = flags.crop_width + left_cor + right_cor;
    JDIMENSION crop_x = flags.crop_x - left_cor;

    jpeg_crop_scanline(&cinfo, &crop_x, &crop_width);

    // Update cinfo.output_scanline.
    skipped_scanlines = jpeg_skip_scanlines(&cinfo, flags.crop_y);
    DALI_ENFORCE(skipped_scanlines == static_cast<JDIMENSION>(flags.crop_y));
  }
#endif

  // check for compatible stride
  const int min_stride = target_output_width * components * sizeof(JSAMPLE);
  if (stride == 0) {
    stride = min_stride;
  } else if (stride < min_stride) {
    ERROR_LOG << "Incompatible stride: " << stride
              << " < " << min_stride << std::endl;
    return nullptr;
  }

  // Remember stride and height for use in Uncompress
  argball->height_ = target_output_height;
  argball->stride_ = stride;

#if !defined(LIBJPEG_TURBO_VERSION)
  if (flags.crop)
    dstdata.reset(new JSAMPLE[stride * target_output_height]);
  else
    dstdata.reset(new JSAMPLE[target_output_width * target_output_height * components]);
#else
  dstdata.reset(new JSAMPLE[target_output_width * target_output_height * components]);
#endif

  if (dstdata == nullptr) {
    return nullptr;
  }

  JSAMPLE* output_line = static_cast<JSAMPLE*>(dstdata.get());

  // jpeg_read_scanlines requires the buffers to be allocated based on
  // cinfo.output_width, but the target image width might be different if crop
  // is enabled and crop_width is not MCU aligned. In this case, we need to
  // realign the scanline output to achieve the exact cropping.  Notably, only
  // cinfo.output_width needs to fall on MCU boundary, while cinfo.output_height
  // has no such constraint.
  const bool need_realign_cropped_scanline =
      (target_output_width != cinfo.output_width);
  const bool use_cmyk = (cinfo.out_color_space == JCS_CMYK);

  if (use_cmyk) {
    // Temporary buffer used for CMYK -> RGB conversion.
    temp.reset(new JSAMPLE[cinfo.output_width * 4]);
  } else if (need_realign_cropped_scanline) {
    // Temporary buffer used for MCU-aligned scanline data.
    temp.reset(new JSAMPLE[cinfo.output_width * components]);
  }
  tempdata = temp.get();

  // If there is an error reading a line, this aborts the reading.
  // Save the fraction of the image that has been read.
  argball->height_read_ = target_output_height;

  // These variables are just to avoid repeated computation in the loop.
  const int max_scanlines_to_read = skipped_scanlines + target_output_height;
  const int mcu_align_offset =
      (cinfo.output_width - target_output_width - right_cor) * (use_cmyk ? 4 : components);
  while (cinfo.output_scanline < static_cast<JDIMENSION>(max_scanlines_to_read)) {
    int num_lines_read = 0;
    if (use_cmyk) {
      num_lines_read = jpeg_read_scanlines(&cinfo, &tempdata, 1);
      if (num_lines_read > 0) {
        // Convert CMYK to RGB if scanline read succeeded.
        for (size_t i = 0; i < target_output_width; ++i) {
          int offset = 4 * i;
          if (need_realign_cropped_scanline) {
            // Align the offset for MCU boundary.
            offset += mcu_align_offset;
          }
          const int c = tempdata[offset + 0];
          const int m = tempdata[offset + 1];
          const int y = tempdata[offset + 2];
          const int k = tempdata[offset + 3];
          int r, g, b;
          if (cinfo.saw_Adobe_marker) {
            r = (k * c) / 255;
            g = (k * m) / 255;
            b = (k * y) / 255;
          } else {
            r = (255 - k) * (255 - c) / 255;
            g = (255 - k) * (255 - m) / 255;
            b = (255 - k) * (255 - y) / 255;
          }

          switch (color_space) {
            case DALI_RGB:
              output_line[3 * i + 0] = r;
              output_line[3 * i + 1] = g;
              output_line[3 * i + 2] = b;
              break;
            case DALI_BGR:
              output_line[3 * i + 0] = b;
              output_line[3 * i + 1] = g;
              output_line[3 * i + 2] = r;
              break;
            case DALI_GRAY:
              output_line[i] = 0.299f * r + 0.587f * g + 0.114f * b;
              break;
            default:
              DALI_FAIL("color_space " + std::to_string(color_space) + " not supported");
          }
        }
      }
    } else if (need_realign_cropped_scanline) {
      num_lines_read = jpeg_read_scanlines(&cinfo, &tempdata, 1);
      if (num_lines_read > 0) {
        memcpy(output_line, tempdata + mcu_align_offset, min_stride);
      }
    } else {
      num_lines_read = jpeg_read_scanlines(&cinfo, &output_line, 1);
    }
    // Handle error cases
    if (num_lines_read == 0) {
      ERROR_LOG << "Premature end of JPEG data. Stopped at line "
                << cinfo.output_scanline - skipped_scanlines << "/"
                << target_output_height << std::endl;
      if (!flags.try_recover_truncated_jpeg) {
        argball->height_read_ = cinfo.output_scanline - skipped_scanlines;
        error = JPEGERRORS_UNEXPECTED_END_OF_DATA;
      } else {
        for (size_t line = cinfo.output_scanline; line < static_cast<size_t>(max_scanlines_to_read);
             ++line) {
          if (line == 0) {
            // If even the first line is missing, fill with black color
            memset(output_line, 0, min_stride);
          } else {
            // else, just replicate the line above.
            memcpy(output_line, output_line - stride, min_stride);
          }
          output_line += stride;
        }
        argball->height_read_ =
            target_output_height;  // consider all lines as read
        // prevent error-on-exit in libjpeg:
        cinfo.output_scanline = max_scanlines_to_read;
      }
      break;
    }
    DALI_ENFORCE(num_lines_read == 1);
    output_line += stride;
  }
  temp.reset();
  tempdata = nullptr;

#if defined(LIBJPEG_TURBO_VERSION)
  if (flags.crop && cinfo.output_scanline < cinfo.output_height) {
    // Skip the rest of scanlines, required by jpeg_destroy_decompress.
    JDIMENSION skip_count = cinfo.output_height - flags.crop_y - flags.crop_height;
    jpeg_skip_scanlines(&cinfo, skip_count);
    // After this, cinfo.output_height must be equal to cinfo.output_scanline;
    // otherwise, jpeg_destroy_decompress would fail.
    DALI_ENFORCE(cinfo.output_height == cinfo.output_scanline);
  }
#endif

  // Convert the RGB data to RGBA, with alpha set to 0xFF to indicate
  // opacity.
  // RGBRGBRGB... --> RGBARGBARGBA...
  if (components == 4) {
    // Start on the last line.
    JSAMPLE* scanlineptr = static_cast<JSAMPLE*>(
        dstdata.get() + static_cast<int64_t>(target_output_height - 1) * stride);
    const JSAMPLE kOpaque = -1;  // All ones appropriate for JSAMPLE.
    const int right_rgb = (target_output_width - 1) * 3;
    const int right_rgba = (target_output_width - 1) * 4;

    for (int y = target_output_height; y-- > 0;) {
      // We do all the transformations in place, going backwards for each row.
      const JSAMPLE* rgb_pixel = scanlineptr + right_rgb;
      JSAMPLE* rgba_pixel = scanlineptr + right_rgba;
      scanlineptr -= stride;
      for (int x = target_output_width; x-- > 0;
           rgba_pixel -= 4, rgb_pixel -= 3) {
        // We copy the 3 bytes at rgb_pixel into the 4 bytes at rgba_pixel
        // The "a" channel is set to be opaque.
        rgba_pixel[3] = kOpaque;
        rgba_pixel[2] = rgb_pixel[2];
        rgba_pixel[1] = rgb_pixel[1];
        rgba_pixel[0] = rgb_pixel[0];
      }
    }
  }

  switch (components) {
    case 1:
      if (cinfo.output_components != 1) {
        error = JPEGERRORS_BAD_PARAM;
      }
      break;
    case 3:
    case 4:
      if (cinfo.out_color_space == JCS_CMYK) {
        if (cinfo.output_components != 4) {
          error = JPEGERRORS_BAD_PARAM;
        }
      } else {
        if (cinfo.output_components != 3) {
          error = JPEGERRORS_BAD_PARAM;
        }
      }
      break;
    default:
      // will never happen, should be catched by the previous switch
      ERROR_LOG << "Invalid components value " << components << std::endl;
      return nullptr;
  }

  // Handle errors in JPEG
  switch (error) {
    case JPEGERRORS_OK:
      DALI_ENFORCE(jpeg_finish_decompress(&cinfo));
      break;
    case JPEGERRORS_UNEXPECTED_END_OF_DATA:
    case JPEGERRORS_BAD_PARAM:
      jpeg_abort(reinterpret_cast<j_common_ptr>(&cinfo));
      break;
    default:
      ERROR_LOG << "Unhandled case " << error << std::endl;
      break;
  }

#if !defined(LIBJPEG_TURBO_VERSION)
  // TODO(tanmingxing): delete all these code after migrating to libjpeg_turbo
  // for Windows.
  if (flags.crop) {
    // Update target output height and width based on crop window.
    target_output_height = flags.crop_height;
    target_output_width = flags.crop_width;

    // cinfo holds the original input image information.
    if (!IsCropWindowValid(flags, cinfo.output_width, cinfo.output_height)) {
      ERROR_LOG << "Invalid crop window: x=" << flags.crop_x
                << ", y=" << flags.crop_y << ", w=" << target_output_width
                << ", h=" << target_output_height
                << " for image_width: " << cinfo.output_width
                << " and image_height: " << cinfo.output_height
                << std::endl;
      return nullptr;
    }

    const auto full_image = std::move(dstdata);
    dstdata = std::unique_ptr<uint8_t[]>(
        new JSAMPLE[target_output_width, target_output_height, components]);
    if (dstdata == nullptr) {
      return nullptr;
    }

    const int full_image_stride = stride;
    // Update stride and hight for crop window.
    const int min_stride = target_output_width * components * sizeof(JSAMPLE);
    if (flags.stride == 0) {
      stride = min_stride;
    }
    argball->height_ = target_output_height;
    argball->stride_ = stride;

    if (argball->height_read_ > target_output_height) {
      argball->height_read_ = target_output_height;
    }
    const int crop_offset = flags.crop_x * components * sizeof(JSAMPLE);
    const uint8_t* full_image_ptr = full_image.get() + flags.crop_y * full_image_stride;
    uint8_t* crop_image_ptr = dstdata.get();
    for (int i = 0; i < argball->height_read_; i++) {
      memcpy(crop_image_ptr, full_image_ptr + crop_offset, min_stride);
      crop_image_ptr += stride;
      full_image_ptr += full_image_stride;
    }
  }
#endif

  return dstdata;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
//  We do the apparently silly thing of packing 5 of the arguments
//  into a structure that is then passed to another routine
//  that does all the work.  The reason is that we want to catch
//  fatal JPEG library errors with setjmp/longjmp, and g++ and
//  associated libraries aren't good enough to guarantee that 7
//  parameters won't get clobbered by the longjmp.  So we help
//  it out a little.
std::unique_ptr<uint8_t[]> Uncompress(const void* srcdata, int datasize,
                                    const UncompressFlags& flags) {
  FewerArgsForCompiler argball(datasize, flags);
  auto dstdata = UncompressLow(srcdata, &argball);

  const float fraction_read =
      argball.height_ == 0
          ? 1.0
          : (static_cast<float>(argball.height_read_) / argball.height_);
  if (dstdata == nullptr ||
      fraction_read < std::min(1.0f, flags.min_acceptable_fraction)) {
    // Major failure, none or too-partial read returned; get out
    return nullptr;
  }

  // If there was an error in reading the jpeg data,
  // set the unread pixels to black
  if (argball.height_read_ != argball.height_) {
    const int first_bad_line = argball.height_read_;
    uint8_t* start = dstdata.get() + first_bad_line * argball.stride_;
    const int nbytes = (argball.height_ - first_bad_line) * argball.stride_;
    memset(static_cast<void*>(start), 0, nbytes);
  }

  return dstdata;
}

// ----------------------------------------------------------------------------
// Computes image information from jpeg header.
// Returns true on success; false on failure.
bool GetImageInfo(const void* srcdata, int datasize, int* width, int* height,
                  int* components) {
  // Init in case of failure
  if (width) *width = 0;
  if (height) *height = 0;
  if (components) *components = 0;

  // If empty image, return
  if (datasize == 0 || srcdata == nullptr) return false;

  // Initialize libjpeg structures to have a memory source
  // Modify the usual jpeg error manager to catch fatal errors.
  struct jpeg_decompress_struct cinfo{};
  struct jpeg_error_mgr jerr;
  jmp_buf jpeg_jmpbuf;
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = CatchError;

  // set up, read header, set image parameters, save size
  jpeg_create_decompress(&cinfo);

  if (setjmp(jpeg_jmpbuf)) {
    jpeg_destroy_decompress(&cinfo);
    return false;
  }
  SetSrc(&cinfo, srcdata, datasize, false);

  if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    jpeg_destroy_decompress(&cinfo);
    return false;
  }
  if (width) *width = cinfo.image_width;
  if (height) *height = cinfo.image_height;
  if (components) *components = cinfo.num_components;

  jpeg_destroy_decompress(&cinfo);
  return true;
}

}  // namespace jpeg
}  // namespace dali
