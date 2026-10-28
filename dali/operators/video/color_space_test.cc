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
std::vector<uint8_t> Convert(int width, int height, u8vec3 yuv, VideoColorSpaceConversionType type) {
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
  auto d_out = mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(out_size);
  CUDA_CALL(cudaMemcpy(d_nv12.get(), nv12.data(), nv12.size(), cudaMemcpyHostToDevice));
  std::vector<uint8_t> out(out_size);
  std::vector<std::vector<uint8_t>> results;
  for (uint8_t fill : {0x00, 0xff}) {
    CUDA_CALL(cudaMemset(d_out.get(), fill, out_size));
    VideoColorSpaceConversion(d_out.get(), width * 3, d_nv12.get(), pitch, height, width, type,
                              false, 0);
    CUDA_CALL(cudaStreamSynchronize(0));
    CUDA_CALL(cudaMemcpy(out.data(), d_out.get(), out_size, cudaMemcpyDeviceToHost));
    results.push_back(out);
  }
  EXPECT_EQ(results[0], results[1])
      << "Some pixels are not written (" << width << "x" << height << ")";
  return results[0];
}

}  // namespace

// The conversion launches one thread per 2x2 pixels. The sizes are not multiples of the block
// (64x4 pixels), so that every pixel, also at the right and bottom edges, has to be covered by
// a grid that has no spare blocks.
TEST(VideoColorSpaceConversionTest, WritesEveryPixel) {
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

}  // namespace test
}  // namespace dali
