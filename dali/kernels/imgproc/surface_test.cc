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

#include <gtest/gtest.h>
#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/backend_tags.h"

namespace dali {
namespace kernels {

TEST(Surface, HWC) {
  const int H = 4;
  const int W = 10;
  const int C = 3;
  float data[W*H*C];
  auto tensor = make_tensor_cpu<3>(data, { H, W, C });
  Surface2D<float> s = as_surface_HWC(tensor);
  EXPECT_EQ(s.width, W);
  EXPECT_EQ(s.height, H);
  EXPECT_EQ(s.channels, C);

  EXPECT_EQ(s.channel_stride, 1);
  EXPECT_EQ(s.pixel_stride, C);
  EXPECT_EQ(s.row_stride, W*C);

  EXPECT_EQ(&s(0, 0, 0), data);
  EXPECT_EQ(&s(0, 0, 1), &data[1]);
  EXPECT_EQ(&s(1, 0, 0), &data[C]);
  EXPECT_EQ(&s(0, 1, 0), &data[W*C]);
  EXPECT_EQ(&s(W-1, H-1, C-1), &data[H*W*C-1]);
}

TEST(Surface, CHW) {
  const int H = 4;
  const int W = 10;
  const int C = 3;
  float data[W*H*C];
  auto tensor = make_tensor_cpu<3>(data, { C, H, W, });
  Surface2D<float> s = as_surface_CHW(tensor);
  EXPECT_EQ(s.width, W);
  EXPECT_EQ(s.height, H);
  EXPECT_EQ(s.channels, C);

  EXPECT_EQ(s.channel_stride, W*H);
  EXPECT_EQ(s.pixel_stride, 1);
  EXPECT_EQ(s.row_stride, W);

  EXPECT_EQ(&s(0, 0, 0), data);
  EXPECT_EQ(&s(0, 0, 1), &data[W*H]);
  EXPECT_EQ(&s(1, 0, 0), &data[1]);
  EXPECT_EQ(&s(0, 1, 0), &data[W]);
  EXPECT_EQ(&s(W-1, H-1, C-1), &data[H*W*C-1]);
}

TEST(Surface, HW) {
  const int H = 4;
  const int W = 10;
  float data[W*H];
  auto tensor = make_tensor_cpu<2>(data, { H, W, });
  Surface2D<float> s = as_surface(tensor);
  EXPECT_EQ(s.width, W);
  EXPECT_EQ(s.height, H);
  EXPECT_EQ(s.channels, 1);

  EXPECT_EQ(s.pixel_stride, 1);
  EXPECT_EQ(s.row_stride, W);

  EXPECT_EQ(&s(0, 0), data);
  EXPECT_EQ(&s(1, 0), &data[1]);
  EXPECT_EQ(&s(0, 1), &data[W]);
  EXPECT_EQ(&s(W-1, H-1, 0), &data[H*W-1]);
}

}  // namespace kernels
}  // namespace dali
