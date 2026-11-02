// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>
#include "dali/core/cuda_error.h"
#include "dali/core/format.h"
#include "dali/core/mm/memory.h"
#include "dali/core/util.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"
#include "dali/operators/video/color_space.h"

namespace dali {
namespace test {

namespace {

constexpr int kStrideAlignment = 8;

// Runs the conversion on NV12 planes of one color and returns the RGB (or YUV) output
template <typename Out = uint8_t>
std::vector<Out> Convert(int width, int height, u8vec3 yuv, VideoColorSpaceConversionType type,
                         bool normalized = false) {
  int pitch = div_ceil(width, kStrideAlignment) * kStrideAlignment;
  std::vector<uint8_t> nv12(static_cast<size_t>(pitch) * height * 3 / 2);
  for (int y = 0; y < height; y++)
    std::fill_n(&nv12[static_cast<size_t>(y) * pitch], width, yuv[0]);
  for (int y = 0; y < height / 2; y++) {
    for (int x = 0; x < width / 2; x++) {
      nv12[static_cast<size_t>(height + y) * pitch + 2 * x] = yuv[1];
      nv12[static_cast<size_t>(height + y) * pitch + 2 * x + 1] = yuv[2];
    }
  }

  // A pixel that is not written keeps the value the output was filled with, so running with two
  // different fills gives two different results
  size_t out_size = static_cast<size_t>(width) * height * 3;
  auto d_nv12 = mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(nv12.size());
  auto d_out = mm::alloc_raw_unique<Out, mm::memory_kind::device>(out_size);
  CUDA_CALL(cudaMemcpy(d_nv12.get(), nv12.data(), nv12.size(), cudaMemcpyHostToDevice));
  std::vector<Out> out(out_size);
  std::vector<std::vector<Out>> results;
  for (uint8_t fill : {0x00, 0xff}) {
    CUDA_CALL(cudaMemset(d_out.get(), fill, out_size * sizeof(Out)));
    ConvertVideoColorSpace(d_out.get(), width * 3, d_nv12.get(), pitch, height, width, type,
                           normalized, 0);
    CUDA_CALL(cudaStreamSynchronize(0));
    CUDA_CALL(cudaMemcpy(out.data(), d_out.get(), out_size * sizeof(Out),
                         cudaMemcpyDeviceToHost));
    results.push_back(out);
  }
  EXPECT_EQ(results[0], results[1])
      << "Some pixels are not written (" << width << "x" << height << ")";
  return results[0];
}

/**
 * @brief The RGB of an 8-bit YCbCr pixel (BT.601), in [0, 1] and not clamped, from the
 *        quantization formulas of ITU-R BT.601 and H.273
 *
 * Limited range: Y = 219 E'Y + 16, Cb = 224 E'Cb + 128. Full range: Y = 255 E'Y,
 * Cb = 255 E'Cb + 128. The same for Cr.
 */
dvec3 ReferenceRGB(u8vec3 yuv, bool full_range) {
  double y = full_range ? yuv[0] / 255.0 : (yuv[0] - 16) / 219.0;
  double cb = (yuv[1] - 128) / (full_range ? 255.0 : 224.0);
  double cr = (yuv[2] - 128) / (full_range ? 255.0 : 224.0);
  constexpr double kr = 0.299, kb = 0.114, kg = 1 - kr - kb;
  double r = y + 2 * (1 - kr) * cr;
  double b = y + 2 * (1 - kb) * cb;
  double g = (y - kr * r - kb * b) / kg;
  return {r, g, b};
}

}  // namespace

// The conversion launches one thread per 2x2 pixels. The sizes are not multiples of the block
// (64x4 pixels), so that every pixel, also at the right and bottom edges, has to be covered by
// a grid that has no spare blocks.
TEST(ConvertVideoColorSpaceTest, WritesEveryPixel) {
  const std::pair<int, int> sizes[] = {{2, 2},   {4, 2},    {62, 6},    {64, 4},     {66, 34},
                                       {130, 70}, {638, 478}, {1280, 720}, {1922, 1082}};
  const u8vec3 colors[] = {u8vec3(128, 128, 128), u8vec3(81, 90, 240), u8vec3(173, 42, 26)};
  using namespace kernels::color;  // NOLINT(build/namespaces)
  for (auto [width, height] : sizes) {
    for (auto yuv : colors) {
      SCOPED_TRACE(make_string(width, "x", height, " YUV ", int(yuv[0]), " ", int(yuv[1]), " ",
                               int(yuv[2])));
      struct {
        VideoColorSpaceConversionType type;
        u8vec3 expected;
      } cases[] = {
          {VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB, itu_r_bt_601::ycbcr_to_rgb<uint8_t>(yuv)},
          {VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_FULL_RANGE,
           jpeg::ycbcr_to_rgb<uint8_t>(yuv)},
          {VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_UPSAMPLE, yuv}};
      for (const auto &c : cases) {
        auto out = Convert(width, height, yuv, c.type);
        for (int i = 0; i < width * height; i++) {
          for (int ch = 0; ch < 3; ch++) {
            // the conversion is done in float, the reference in the type of the output
            ASSERT_NEAR(out[3 * i + ch], c.expected[ch], 1)
                << "type " << c.type << ", pixel " << i << ", channel " << ch;
          }
        }
      }
    }
  }
}

// The conversion to RGB follows the quantization of ITU-R BT.601 and H.273: Y of the limited range
// at 16 and the chroma at 128 (for a long time they were 15.9375 and 127.5, as the samples were
// normalized by 255 instead of 256, which made e.g. black (1, 0, 1)). The 8-bit output is the
// reference rounded to the nearest level, and the floating point outputs are the reference
// itself, in levels or in [0, 1] (white is 1.0).
TEST(ConvertVideoColorSpaceTest, ItuLevels) {
  const u8vec3 colors[] = {
      // black, gray and white of the limited range, and beyond them
      u8vec3(16, 128, 128), u8vec3(126, 128, 128), u8vec3(235, 128, 128), u8vec3(0, 128, 128),
      u8vec3(255, 128, 128),
      // the 75% and 100% color bars (BT.601, limited range): yellow, cyan, green, magenta, red
      // and blue
      u8vec3(162, 44, 142), u8vec3(131, 156, 44), u8vec3(112, 72, 58), u8vec3(84, 184, 198),
      u8vec3(65, 100, 212), u8vec3(35, 212, 114), u8vec3(210, 16, 146), u8vec3(170, 166, 16),
      u8vec3(145, 54, 34), u8vec3(106, 202, 222), u8vec3(81, 90, 240), u8vec3(41, 240, 110),
      // arbitrary colors, and colors at the ends of the chroma range
      u8vec3(173, 42, 26), u8vec3(1, 2, 254), u8vec3(254, 253, 3)};
  const int width = 6, height = 4;
  for (bool full_range : {false, true}) {
    auto type = full_range ? VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_FULL_RANGE
                           : VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB;
    for (auto yuv : colors) {
      SCOPED_TRACE(make_string(full_range ? "full" : "limited", " range, YUV ", int(yuv[0]), " ",
                               int(yuv[1]), " ", int(yuv[2])));
      dvec3 ref = ReferenceRGB(yuv, full_range);
      auto u8 = Convert(width, height, yuv, type);
      auto u8_norm = Convert(width, height, yuv, type, true);
      auto f = Convert<float>(width, height, yuv, type);
      auto f_norm = Convert<float>(width, height, yuv, type, true);
      for (int i = 0; i < width * height; i++) {
        for (int ch = 0; ch < 3; ch++) {
          // the nearest level, up to the float rounding of a tie
          double level = std::clamp(ref[ch] * 255, 0.0, 255.0);
          ASSERT_NEAR(u8[3 * i + ch], level, 0.5 + 1e-3) << "pixel " << i << ", channel " << ch;
          ASSERT_NEAR(u8_norm[3 * i + ch], level, 0.5 + 1e-3)
              << "pixel " << i << ", channel " << ch;
          ASSERT_NEAR(f[3 * i + ch], ref[ch] * 255, 1e-3) << "pixel " << i << ", channel " << ch;
          ASSERT_NEAR(f_norm[3 * i + ch], ref[ch], 1e-5) << "pixel " << i << ", channel " << ch;
        }
      }
    }
  }
}

// The upsampled YCbCr is not converted: the samples as they are (exactly, also in floating point,
// as the chroma of a constant plane is interpolated exactly), or normalized by 255
TEST(ConvertVideoColorSpaceTest, UpsampleLevels) {
  const u8vec3 colors[] = {u8vec3(16, 128, 128), u8vec3(235, 16, 240), u8vec3(0, 255, 1),
                           u8vec3(255, 0, 254), u8vec3(1, 127, 129), u8vec3(254, 3, 252)};
  const int width = 6, height = 4;
  auto type = VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_UPSAMPLE;
  for (auto yuv : colors) {
    SCOPED_TRACE(make_string("YUV ", int(yuv[0]), " ", int(yuv[1]), " ", int(yuv[2])));
    auto u8 = Convert(width, height, yuv, type);
    auto u8_norm = Convert(width, height, yuv, type, true);
    auto f = Convert<float>(width, height, yuv, type);
    auto f_norm = Convert<float>(width, height, yuv, type, true);
    for (int i = 0; i < width * height; i++) {
      for (int ch = 0; ch < 3; ch++) {
        ASSERT_EQ(u8[3 * i + ch], yuv[ch]) << "pixel " << i << ", channel " << ch;
        ASSERT_EQ(u8_norm[3 * i + ch], yuv[ch]) << "pixel " << i << ", channel " << ch;
        ASSERT_EQ(f[3 * i + ch], yuv[ch]) << "pixel " << i << ", channel " << ch;
        ASSERT_NEAR(f_norm[3 * i + ch], yuv[ch] / 255.0, 1e-6)
            << "pixel " << i << ", channel " << ch;
      }
    }
  }
}

}  // namespace test
}  // namespace dali
