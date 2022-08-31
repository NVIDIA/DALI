// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/imgcodec/util/output_shape.h"

namespace dali {
namespace imgcodec {

TEST(OutputShapeTest, Trivial) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  OutputShape(out, ii, {}, {});
  EXPECT_EQ(ii.shape, out);
}

TEST(OutputShapeTest, Planar) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  DecodeParams params;
  params.planar = true;
  OutputShape(out, ii, params, {});
  ASSERT_EQ(out.sample_dim(), 3);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], 480);
  EXPECT_EQ(out[2], 640);
}

TEST(OutputShapeTest, RoiBegin) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  ROI roi;
  roi.begin = { 7, 5 };
  DecodeParams params;
  params.planar = true;
  OutputShape(out, ii, params, roi);
  ASSERT_EQ(out.sample_dim(), 3);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], 473);
  EXPECT_EQ(out[2], 635);
}

TEST(OutputShapeTest, RoiEnd) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  ROI roi;
  roi.end = { 70, 50 };
  DecodeParams params;
  params.planar = true;
  OutputShape(out, ii, params, roi);
  ASSERT_EQ(out.sample_dim(), 3);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], 70);
  EXPECT_EQ(out[2], 50);
}

TEST(OutputShapeTest, RoiBoth) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  ROI roi;
  roi.begin = { 7, 5 };
  roi.end = { 77, 55 };
  OutputShape(out, ii, {}, roi);
  ASSERT_EQ(out.sample_dim(), 3);
  EXPECT_EQ(out[0], 70);
  EXPECT_EQ(out[1], 50);
  EXPECT_EQ(out[2], 3);
}


TEST(OutputShapeTest, Rotate) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  DecodeParams params;
  params.use_orientation = true;
  for (int a = 0; a < 360; a += 90) {
    ii.orientation.rotate = a;
    bool swapped = a % 180 == 90;
    OutputShape(out, ii, params, {});
    ASSERT_EQ(out.sample_dim(), 3);
    if (swapped) {
      EXPECT_EQ(out[0], 640);
      EXPECT_EQ(out[1], 480);
    } else {
      EXPECT_EQ(out[0], 480);
      EXPECT_EQ(out[1], 640);
    }
    EXPECT_EQ(out[2], 3);
  }
}


TEST(OutputShapeTest, RoiRotate) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  ROI roi;
  roi.begin = { 7, 5 };
  roi.end = { 77, 55 };
  for (int a = 0; a < 360; a += 90) {
    ii.orientation.rotate = a;
    OutputShape(out, ii, {}, roi);
    ASSERT_EQ(out.sample_dim(), 3);
    // ROI is always in the output space
    EXPECT_EQ(out[0], 70);
    EXPECT_EQ(out[1], 50);
    EXPECT_EQ(out[2], 3);
  }
}

}  // namespace imgcodec
}  // namespace dali
