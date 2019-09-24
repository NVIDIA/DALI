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
#include <vector>
#include "dali/kernels/imgproc/roi.h"

namespace dali {
namespace kernels {

namespace test {

using Roi = ::dali::kernels::Roi<2>;

TEST(RoiTest, roi_to_TensorShape) {
  {
    Roi roi{0, 3};
    auto sh = ::dali::kernels::ShapeFromRoi(roi, 3);
    TensorShape<3> ref_sh = {3, 3, 3};
    EXPECT_EQ(ref_sh, sh);
  }
  {
    Roi roi{{0, 2},
            {5, 6}};
    auto sh = ::dali::kernels::ShapeFromRoi(roi, 666);
    TensorShape<3> ref_sh = {4, 5, 666};
    EXPECT_EQ(ref_sh, sh);
  }
  {
    Roi roi{{0, 0},
            {0, 0}};
    auto sh = ::dali::kernels::ShapeFromRoi(roi, 666);
    TensorShape<3> ref_sh = {0, 0, 666};
    EXPECT_EQ(ref_sh, sh);
  }
}


TEST(RoiTest, roi_to_TensorListShape) {
  using Rois = std::vector<Roi>;
  constexpr int nchannels = 3;
  Roi roi1{0, 3};
  Roi roi2{{0, 2},
           {5, 6}};
  Roi roi3{{0, 0},
           {0, 0}};
  Rois rois = {roi1, roi2, roi3};

  std::vector<TensorShape<-1>> ref = {{3, 3, nchannels},
                                      {4, 5, nchannels},
                                      {0, 0, nchannels}};
  auto shs = ShapeFromRoi(make_cspan(rois), nchannels);
  EXPECT_EQ(shs, TensorListShape<-1>(ref));
}


TEST(RoiTest, whole_image) {
  TensorShape<3> sh1 = {3, 3, 5};
  TensorShape<3> sh2 = {4, 5, 2};
  TensorShape<3> sh3 = {0, 0, 89};

  Roi ref1 = {0, 3};
  Roi ref2 = {{0, 0},
              {5, 4}};
  Roi ref3 = {0, 0};

  EXPECT_EQ(detail::WholeImage(sh1), ref1);
  EXPECT_EQ(detail::WholeImage(sh2), ref2);
  EXPECT_EQ(detail::WholeImage(sh3), ref3);
}


TEST(RoiTest, adjust_empty_roi) {
  constexpr int ndims = 3;
  std::vector<Roi> rois;

  TensorShape<ndims> ts{5, 6, 7};
  Roi *roi = nullptr;
  Roi ref = {{0, 0},
             {6, 5}};
  auto res = AdjustRoi(roi, ts);
  EXPECT_EQ(ref, res);
}


TEST(RoiTest, adjust_roi) {
  constexpr int ndims = 3;

  Roi roi = {{5, 6},
           {7, 8}};
  TensorShape<ndims> ts = {12, 13, 14};
  Roi ref = {{5, 6},
             {7, 8}};
  auto res = AdjustRoi(&roi, ts);
  EXPECT_EQ(ref, res);
}


TEST(RoiTest, adjust_empty_rois) {
  constexpr int ndims = 3;
  std::vector<Roi> rois;

  TensorListShape<ndims> tls({{2, 3, 4},
                              {5, 6, 7}});
  std::vector<Roi> ref = {
          {{0, 0}, {3, 2}},
          {{0, 0}, {6, 5}},
  };
  auto res = AdjustRoi(make_cspan(rois), tls);
  ASSERT_EQ(ref.size(), res.size());
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(ref[i], res[i]);
  }
}


TEST(RoiTest, adjust_rois) {
  constexpr int ndims = 3;

  std::vector<Roi> rois = {
          {{1, 2}, {3, 4}},
          {{5, 6}, {7, 8}},
          {0,      20},
          {{0, 0}, {640, 480}}
  };
  std::vector<TensorShape<ndims>> ts = {{9,   10,  11},
                                        {12,  13,  14},
                                        {1,   1,   1},
                                        {480, 640, 3}};
  TensorListShape<ndims> tls = ts;
  std::vector<Roi> ref = {
          {{1, 2}, {3, 4}},
          {{5, 6}, {7, 8}},
          {0,      1},
          {0,      {640, 480}},
  };
  auto res = AdjustRoi(make_cspan(rois), tls);
  ASSERT_EQ(ref.size(), res.size());
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(ref[i], res[i]);
  }
}

}  // namespace test
}  // namespace kernels
}  // namespace dali
