// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/imgcodec/util/output_shape.h"

namespace dali {
namespace imgcodec {

TEST(OutputShapeTest, Trivial) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  OutputShape(out, ii, DALI_RGB, false, true, ROI{});
  EXPECT_EQ(ii.shape, out);
}

TEST(OutputShapeTest, Grayscale) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  OutputShape(out, ii, DALI_GRAY, false, true, ROI{});
  EXPECT_EQ(out[0], 480);
  EXPECT_EQ(out[1], 640);
  EXPECT_EQ(out[2], 1);
}

TEST(OutputShapeTest, Planar) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  OutputShape(out, ii, DALI_RGB, true, true, ROI{});
  ASSERT_EQ(out.sample_dim(), 3);
  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], 480);
  EXPECT_EQ(out[2], 640);
}

TEST(OutputShapeTest, PlanarGrayscale) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  OutputShape(out, ii, DALI_GRAY, true, true, ROI{});
  ASSERT_EQ(out.sample_dim(), 3);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 480);
  EXPECT_EQ(out[2], 640);
}

TEST(OutputShapeTest, RoiBegin) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  ROI roi;
  roi.begin = { 7, 5 };
  OutputShape(out, ii, DALI_RGB, true, true, roi);
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
  OutputShape(out, ii, DALI_RGB, true, true, roi);
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
  OutputShape(out, ii, DALI_RGB, false, true, roi);
  ASSERT_EQ(out.sample_dim(), 3);
  EXPECT_EQ(out[0], 70);
  EXPECT_EQ(out[1], 50);
  EXPECT_EQ(out[2], 3);
}

TEST(OutputShapeTest, NegativeRoi) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  ROI roi;
  roi.begin = { -1, 5 };
  roi.end = { 77, 55 };
  for (int a = 0; a < 360; a += 90) {
    ii.orientation.rotated = a;
    ASSERT_THROW(OutputShape(out, ii, DALI_RGB, false, true, roi), DALIException);
  }
}

TEST(OutputShapeTest, RoiExceedingShape) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  ROI roi;
  roi.begin = { 7, 5 };
  roi.end = { 481, 641 };
  for (int a = 0; a < 360; a += 90) {
    ii.orientation.rotated = a;
    ASSERT_THROW(OutputShape(out, ii, DALI_RGB, false, true, roi), DALIException);
  }
}

TEST(OutputShapeTest, Rotate) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  for (int a = 0; a < 360; a += 90) {
    ii.orientation.rotated = a;
    bool swapped = a % 180 == 90;
    OutputShape(out, ii, DALI_RGB, false, true, ROI{});
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
    ii.orientation.rotated = a;
    OutputShape(out, ii, DALI_RGB, false, true, roi);
    ASSERT_EQ(out.sample_dim(), 3);
    // ROI is always in the output space
    EXPECT_EQ(out[0], 70);
    EXPECT_EQ(out[1], 50);
    EXPECT_EQ(out[2], 3);
  }
}

TEST(OutputShapeTest, InvalidRoiRotate) {
  ImageInfo ii;
  ii.shape = { 480, 640, 3 };
  TensorShape<> out;
  ROI roi;
  roi.begin = { 7, 5 };
  roi.end = { 432, 543 };
  for (int a = 0; a < 360; a += 90) {
    ii.orientation.rotated = a;
    if (a % 180 == 90) {
      ASSERT_THROW(OutputShape(out, ii, DALI_RGB, false, true, roi), DALIException);
    } else {
      OutputShape(out, ii, DALI_RGB, false, true, roi);
      ASSERT_EQ(out.sample_dim(), 3);
      EXPECT_EQ(out[0], 425);
      EXPECT_EQ(out[1], 538);
      EXPECT_EQ(out[2], 3);
    }
  }
}

}  // namespace imgcodec
}  // namespace dali
