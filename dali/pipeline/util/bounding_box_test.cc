// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/util/bounding_box.h"

namespace dali {
namespace test {

static int kBboxSize = 2;

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptNegativeCoordinates) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(-0.5, 0.5, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBWithNoBoundsAcceptNegativeCoordinates) {
  BoundingBox::FromLtrb(-0.5, 0.5, 0.5, 0.5, BoundingBox::NoBounds());
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptNegativeCoordinates2) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, -0.5, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBWithNoBoundsAcceptNegativeCoordinates2) {
  BoundingBox::FromLtrb(0.5, -0.5, 0.5, 0.5, BoundingBox::NoBounds());
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptNegativeCoordinates3) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 0.5, -0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBWithNoBoundsAcceptNegativeCoordinates3) {
  BoundingBox::FromLtrb(-1.5, 0.5, -0.5, 0.5, BoundingBox::NoBounds());
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptNegativeCoordinates4) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 0.5, 0.5, -0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBWithNoBoundsAcceptNegativeCoordinates4) {
  BoundingBox::FromLtrb(0.5, -1.5, 0.5, -0.5, BoundingBox::NoBounds());
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptCoordinatesLargerThanOne) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(1.1, 0.5, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBWithNoBoundsAcceptCoordinatesLargerThanOne) {
  BoundingBox::FromLtrb(1.1, 0.5, 1.5, 0.5, BoundingBox::NoBounds());
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptCoordinatesLargerThanOne2) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 1.1, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBWithNoBoundsAcceptCoordinatesLargerThanOne2) {
  BoundingBox::FromLtrb(0.5, 1.1, 0.5, 1.5, BoundingBox::NoBounds());
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptCoordinatesLargerThanOne3) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 0.5, 1.1, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBWithNoBoundsAcceptCoordinatesLargerThanOne3) {
  BoundingBox::FromLtrb(0.5, 0.5, 1.1, 0.5, BoundingBox::NoBounds());
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptCoordinatesLargerThanOne4) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 0.5, 0.5, 1.1));
}

TEST(BoundingBoxTest, BoundingBoxLTRBWithNoBoundsAcceptCoordinatesLargerThanOne4) {
  BoundingBox::FromLtrb(0.5, 0.5, 0.5, 1.1, BoundingBox::NoBounds());
}

TEST(BoundingBoxTest, BoundingBoxRightMustBeGreaterOrEqualToLeft) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 0.0, 0.0, 0.0));
}

TEST(BoundingBoxTest, BoundingBoxTopMustBeGreaterOrEqualToBottom) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.0, 0.5, 0.0, 0.0));
}

TEST(BoundingBoxTest, BoundingBoxXYWHDoesNotAcceptNegativeCoordinates) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(-0.5, 0.5, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxXYWHDoesNotAcceptNegativeCoordinates2) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(0.5, -0.5, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxXYWHDoesNotAcceptNegativeCoordinates3) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(0.5, 0.0, -0.6, 0.0));
}

TEST(BoundingBoxTest, BoundingBoxXYWHDoesNotAcceptNegativeCoordinates4) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(0.0, 0.5, 0.0, -0.6));
}

TEST(BoundingBoxTest, BoundingBoxXYWHDoesNotAcceptCoordinatesLargerThanOne) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(1.1, 0.5, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxXYWHDoesNotAcceptCoordinatesLargerThanOne2) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(0.5, 1.1, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxXYWHDoesNotAcceptCoordinatesLargerThanOne3) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(0.5, 0.0, 1.1, 0.0));
}

TEST(BoundingBoxTest, BoundingBoxXYWHDoesNotAcceptCoordinatesLargerThanOne4) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(0.0, 0.5, 0.0, 1.1));
}

TEST(BoundingBoxTest, BoundingBoxXYWHMustNotExceedFrameHorizontally) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(0.5, 0.0, 0.6, 0.0));
}

TEST(BoundingBoxTest, BoundingBoxXYWHMustNotExceedFrameVertically) {
  EXPECT_ANY_THROW(BoundingBox::FromXywh(0.5, 0.0, 0.6, 0.0));
}

TEST(BoundingBoxTest, ContainsReturnsTrueIfPointIsContained) {
  auto box = BoundingBox::FromLtrb(0.25, 0.25, 0.75, 0.75);

  EXPECT_TRUE(box.Contains(0.5, 0.5));
  EXPECT_TRUE(box.Contains(0.25, 0.25));
  EXPECT_TRUE(box.Contains(0.25, 0.75));
  EXPECT_TRUE(box.Contains(0.75, 0.75));

  auto box2 = BoundingBox::FromLtrb(0.0, 0.0, 0.0, 0.0);
  EXPECT_TRUE(box2.Contains(0.0, 0.0));
}

TEST(BoundingBoxTest, ContainsReturnsTrueIfPointIsContainedXYWH) {
  auto box = BoundingBox::FromXywh(0.25, 0.25, 0.5, 0.5);

  EXPECT_TRUE(box.Contains(0.5, 0.5));
  EXPECT_TRUE(box.Contains(0.25, 0.25));
  EXPECT_TRUE(box.Contains(0.25, 0.75));
  EXPECT_TRUE(box.Contains(0.75, 0.75));

  auto box2 = BoundingBox::FromXywh(0.0, 0.0, 0.0, 0.0);
  EXPECT_TRUE(box2.Contains(0.0, 0.0));
}

TEST(BoundingBoxTest, ContainsReturnsFalseIfPointIsContained) {
  auto box = BoundingBox::FromLtrb(0.0, 0.0, 0.0, 0.0);

  EXPECT_FALSE(box.Contains(0.5, 0.5));
  EXPECT_FALSE(box.Contains(0.25, 0.25));
  EXPECT_FALSE(box.Contains(0.25, 0.75));
  EXPECT_FALSE(box.Contains(0.75, 0.75));
}

TEST(BoundingBoxTest, ContainsReturnsFalseIfPointIsContainedXYWH) {
  auto box = BoundingBox::FromXywh(0.0, 0.0, 0.0, 0.0);

  EXPECT_FALSE(box.Contains(0.5, 0.5));
  EXPECT_FALSE(box.Contains(0.25, 0.25));
  EXPECT_FALSE(box.Contains(0.25, 0.75));
  EXPECT_FALSE(box.Contains(0.75, 0.75));
}

TEST(BoundingBoxTest, CanBeIntersectedToSmaller) {
  auto bigger_box = BoundingBox::FromXywh(0.0, 0.0, 1.0, 1.0);
  auto smaller_box = BoundingBox::FromXywh(0.0, 0.0, 0.1, 0.1);

  auto intersected = bigger_box.Intersect(smaller_box);

  EXPECT_EQ(intersected .Area(), smaller_box.Area());
}

TEST(BoundingBoxTest, IntersectBiggerLeavesTheSame) {
  auto bigger_box = BoundingBox::FromXywh(0.0, 0.0, 1.0, 1.0);
  auto smaller_box = BoundingBox::FromXywh(0.0, 0.0, 0.1, 0.1);

  auto intersected = smaller_box.Intersect(bigger_box);

  EXPECT_EQ(intersected .Area(), smaller_box.Area());
}

TEST(BoundingBoxTest, CanFlipHorizontallyXYWH) {
  auto box_initial = BoundingBox::FromXywh(0.25, 0.25, 0.25, 0.25);
  auto coords_initial_ltrb = box_initial.AsStartAndEnd();
  auto coords_initial_xywh = box_initial.AsStartAndShape();

  auto box = BoundingBox::FromXywh(0.25, 0.25, 0.25, 0.25);

  box = box.HorizontalFlip();
  box = box.HorizontalFlip();

  auto coords_flipped_ltrb = box.AsStartAndEnd();
  auto coords_flipped_xywh = box.AsStartAndShape();

  for (int i = 0; i < kBboxSize; i++) {
    EXPECT_EQ(coords_initial_ltrb[i], coords_flipped_ltrb[i]);
    EXPECT_EQ(coords_initial_xywh[i], coords_flipped_xywh[i]);
  }
}

TEST(BoundingBoxTest, CanFlipHorizontallyLTRB) {
  auto box_initial = BoundingBox::FromLtrb(0.25, 0.25, 0.5, 0.5);
  auto coords_initial_ltrb = box_initial.AsStartAndEnd();
  auto coords_initial_xywh = box_initial.AsStartAndShape();

  auto box = BoundingBox::FromLtrb(0.25, 0.25, 0.5, 0.5);

  box = box.HorizontalFlip();
  box = box.HorizontalFlip();

  auto coords_flipped_ltrb = box.AsStartAndEnd();
  auto coords_flipped_xywh = box.AsStartAndShape();

  for (int i = 0; i < kBboxSize; i++) {
    EXPECT_EQ(coords_initial_ltrb[i], coords_flipped_ltrb[i]);
    EXPECT_EQ(coords_initial_xywh[i], coords_flipped_xywh[i]);
  }
}

TEST(BoundingBoxTest, CanFlipVerticallyXYWH) {
  auto box_initial = BoundingBox::FromXywh(0.25, 0.25, 0.25, 0.25);
  auto coords_initial_ltrb = box_initial.AsStartAndEnd();
  auto coords_initial_xywh = box_initial.AsStartAndShape();

  auto box = BoundingBox::FromXywh(0.25, 0.25, 0.25, 0.25);

  box = box.VerticalFlip();
  box = box.VerticalFlip();

  auto coords_flipped_ltrb = box.AsStartAndEnd();
  auto coords_flipped_xywh = box.AsStartAndShape();

  for (int i = 0; i < kBboxSize; i++) {
    EXPECT_EQ(coords_initial_ltrb[i], coords_flipped_ltrb[i]);
    EXPECT_EQ(coords_initial_xywh[i], coords_flipped_xywh[i]);
  }
}

TEST(BoundingBoxTest, CanFlipVerticallyLTRB) {
  auto box_initial = BoundingBox::FromLtrb(0.25, 0.25, 0.5, 0.5);
  auto coords_initial_ltrb = box_initial.AsStartAndEnd();
  auto coords_initial_xywh = box_initial.AsStartAndShape();

  auto box = BoundingBox::FromLtrb(0.25, 0.25, 0.5, 0.5);

  box = box.VerticalFlip();
  box = box.VerticalFlip();

  auto coords_flipped_ltrb = box.AsStartAndEnd();
  auto coords_flipped_xywh = box.AsStartAndShape();

  for (int i = 0; i < kBboxSize; i++) {
    EXPECT_EQ(coords_initial_ltrb[i], coords_flipped_ltrb[i]);
    EXPECT_EQ(coords_initial_xywh[i], coords_flipped_xywh[i]);
  }
}

TEST(BoundingBoxTest, CanCalculateArea) {
  auto ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 0.0, 0.0);
  auto xywh_box = BoundingBox::FromXywh(0.0, 0.0, 0.0, 0.0);

  EXPECT_EQ(ltrb_box.Area(), 0.0f);
  EXPECT_EQ(xywh_box.Area(), 0.0f);

  ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 1.0, 1.0);
  xywh_box = BoundingBox::FromXywh(0.0, 0.0, 1.0, 1.0);

  EXPECT_EQ(ltrb_box.Area(), 1.0f);
  EXPECT_EQ(xywh_box.Area(), 1.0f);

  ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 1.0, 0.5);
  xywh_box = BoundingBox::FromXywh(0.0, 0.0, 1.0, 0.5);

  EXPECT_EQ(ltrb_box.Area(), 0.5f);
  EXPECT_EQ(xywh_box.Area(), 0.5f);
}

TEST(BoundingBoxTest, CanFindContained) {
  auto ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 0.0, 0.0);
  auto xywh_box = BoundingBox::FromXywh(0.0, 0.0, 0.0, 0.0);

  EXPECT_TRUE(ltrb_box.Contains(0, 0));
  EXPECT_TRUE(xywh_box.Contains(0, 0));

  EXPECT_FALSE(ltrb_box.Contains(0.1, 0));
  EXPECT_FALSE(xywh_box.Contains(0.1, 0));

  ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 1.0, 1.0);
  xywh_box = BoundingBox::FromXywh(0.0, 0.0, 1.0, 1.0);

  EXPECT_TRUE(ltrb_box.Contains(1, 1));
  EXPECT_TRUE(xywh_box.Contains(1, 1));

  EXPECT_TRUE(ltrb_box.Contains(1, 0.9));
  EXPECT_TRUE(xywh_box.Contains(1, 0.9));

  ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 1.0, 0.5);
  xywh_box = BoundingBox::FromXywh(0.0, 0.0, 1.0, 0.5);

  EXPECT_TRUE(ltrb_box.Contains(1, 0.5));
  EXPECT_TRUE(xywh_box.Contains(1, 0.5));
}

TEST(BoundingBoxTest, CanFindIfOverlap) {
  auto ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 0.0, 0.0);
  auto xywh_box = BoundingBox::FromXywh(0.0, 0.0, 0.0, 0.0);

  EXPECT_FALSE(ltrb_box.Overlaps(ltrb_box));
  EXPECT_FALSE(xywh_box.Overlaps(ltrb_box));
  EXPECT_FALSE(xywh_box.Overlaps(xywh_box));
  EXPECT_FALSE(ltrb_box.Overlaps(xywh_box));

  ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 1.0, 1.0);
  xywh_box = BoundingBox::FromXywh(0.0, 0.0, 1.0, 1.0);

  EXPECT_TRUE(ltrb_box.Overlaps(ltrb_box));
  EXPECT_TRUE(xywh_box.Overlaps(ltrb_box));
  EXPECT_TRUE(xywh_box.Overlaps(xywh_box));
  EXPECT_TRUE(ltrb_box.Overlaps(xywh_box));

  ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 0.5, 0.5);
  xywh_box = BoundingBox::FromXywh(0.5, 0.5, 0.5, 0.5);

  EXPECT_TRUE(ltrb_box.Overlaps(ltrb_box));
  EXPECT_FALSE(xywh_box.Overlaps(ltrb_box));
  EXPECT_TRUE(xywh_box.Overlaps(xywh_box));
  EXPECT_FALSE(ltrb_box.Overlaps(xywh_box));
}

TEST(BoundingBoxTest, CalculateIOUIsZeroIfNoOverlap) {
  auto ltrb_box = BoundingBox::FromLtrb(0.0, 0.0, 0.0, 0.0);
  auto xywh_box = BoundingBox::FromXywh(0.0, 0.0, 0.0, 0.0);

  EXPECT_EQ(ltrb_box.IntersectionOverUnion(ltrb_box), 0.0f);
  EXPECT_EQ(xywh_box.IntersectionOverUnion(ltrb_box), 0.0f);
  EXPECT_EQ(xywh_box.IntersectionOverUnion(xywh_box), 0.0f);
  EXPECT_EQ(ltrb_box.IntersectionOverUnion(xywh_box), 0.0f);
}

TEST(BoundingBoxTest, Bbox3dIoUNoOverlap) {
  auto bbox1 = BoundingBox::FromStartAndEnd({0.0, 0.0, 0.0, 0.25, 0.25, 0.25});
  auto bbox2 = BoundingBox::FromStartAndEnd({0.25, 0.25, 0.25, 0.5, 0.5, 0.5});

  EXPECT_FALSE(bbox1.Overlaps(bbox2));
  EXPECT_EQ(bbox1.IntersectionOverUnion(bbox2), 0.0f);
}

TEST(BoundingBoxTest, Bbox3dIoUOverlap) {
  auto bbox1 = BoundingBox::FromStartAndEnd({0.0, 0.0, 0.0, 0.3, 0.3, 0.3});
  auto bbox2 = BoundingBox::FromStartAndEnd({0.2, 0.2, 0.2, 0.5, 0.5, 0.5});

  auto intersection_vol = 0.1 * 0.1 * 0.1;
  auto union_vol = 2 * (0.3 * 0.3 * 0.3) - intersection_vol;

  EXPECT_TRUE(bbox1.Overlaps(bbox2));
  EXPECT_FLOAT_EQ(bbox1.IntersectionOverUnion(bbox2), intersection_vol / union_vol);
}

TEST(BoundingBoxTest, Bbox3dIntersect) {
  auto bigger_box = BoundingBox::FromStartAndEnd({0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
  auto smaller_box = BoundingBox::FromStartAndEnd({0.1, 0.1, 0.1, 0.9, 0.9, 0.9});

  EXPECT_EQ(bigger_box.Intersect(smaller_box), smaller_box);
}

TEST(BoundingBoxTest, Bbox3dBboxFormatConversion) {
  RelBounds start_and_end = {0.2, 0.2, 0.2, 0.7, 0.7, 0.7};
  auto bbox1 = BoundingBox::FromStartAndEnd(start_and_end);

  RelBounds start_and_shape = {0.2, 0.2, 0.2, 0.5, 0.5, 0.5};
  auto bbox2 = BoundingBox::FromStartAndShape(start_and_shape);

  EXPECT_EQ(bbox1, bbox2);
  EXPECT_EQ(bbox1.AsStartAndShape(), start_and_shape);
  EXPECT_EQ(bbox2.AsStartAndEnd(), start_and_end);
}

TEST(BoundingBoxTest, LayoutConversionStartAndEnd) {
  RelBounds start_and_end_yzx = {0.2, 0.3, 0.1, 0.44, 0.66, 0.22};
  RelBounds start_and_end_xyz = {0.1, 0.2, 0.3, 0.22, 0.44, 0.66};
  RelBounds start_and_end_zyx = {0.3, 0.2, 0.1, 0.66, 0.44, 0.22};

  ASSERT_EQ(BoundingBox::FromStartAndEnd(start_and_end_yzx, {}, "yzxYZX"),
            BoundingBox::FromStartAndEnd(start_and_end_zyx, {}, "zyxZYX"));

  ASSERT_EQ(BoundingBox::FromStartAndEnd(start_and_end_yzx, {}, "yzxYZX"),
            BoundingBox::FromStartAndEnd(start_and_end_xyz, {}, "xyzXYZ"));

  auto box = BoundingBox::FromStartAndEnd(start_and_end_xyz, {}, "xyzXYZ");
  auto yzx_box = box.AsStartAndEnd("yzxYZX");
  for (int i = 0; i < 6; i++)
    ASSERT_EQ(yzx_box[i], start_and_end_yzx[i]);

  auto zyx_box = box.AsStartAndEnd("zyxZYX");
  for (int i = 0; i < 6; i++)
    ASSERT_EQ(zyx_box[i], start_and_end_zyx[i]);
}

TEST(BoundingBoxTest, LayoutConversionStartAndShape) {
  RelBounds start_and_shape_yzx = {0.2, 0.3, 0.1, 0.44, 0.66, 0.22};
  RelBounds start_and_shape_xyz = {0.1, 0.2, 0.3, 0.22, 0.44, 0.66};
  RelBounds start_and_shape_zyx = {0.3, 0.2, 0.1, 0.66, 0.44, 0.22};

  ASSERT_EQ(BoundingBox::FromStartAndShape(start_and_shape_yzx, {}, "yzxHDW"),
            BoundingBox::FromStartAndShape(start_and_shape_zyx, {}, "zyxDHW"));

  ASSERT_EQ(BoundingBox::FromStartAndShape(start_and_shape_yzx, {}, "yzxHDW"),
            BoundingBox::FromStartAndShape(start_and_shape_xyz, {}, "xyzWHD"));

  auto box = BoundingBox::FromStartAndShape(start_and_shape_xyz, {}, "xyzWHD");
  auto yzx_box = box.AsStartAndShape("yzxHDW");
  for (int i = 0; i < 6; i++)
    ASSERT_EQ(yzx_box[i], start_and_shape_yzx[i]);

  auto zyx_box = box.AsStartAndShape("zyxDHW");
  for (int i = 0; i < 6; i++) {
    ASSERT_EQ(zyx_box[i], start_and_shape_zyx[i]);
  }
}

TEST(BoundingBoxTest, From) {
  RelBounds start_and_shape_yzx = {0.2, 0.3, 0.1, 0.44, 0.66, 0.22};
  RelBounds start_and_end_yzx = {0.2, 0.3, 0.1, 0.64, 0.96, 0.32};

  ASSERT_EQ(BoundingBox::FromStartAndShape(start_and_shape_yzx, {}, "yzxHDW"),
            BoundingBox::From(start_and_shape_yzx, {}, "yzxHDW"));

  ASSERT_EQ(BoundingBox::FromStartAndEnd(start_and_end_yzx, {}, "yzxXYZ"),
            BoundingBox::From(start_and_end_yzx, {}, "yzxXYZ"));

  ASSERT_EQ(BoundingBox::FromStartAndShape(start_and_shape_yzx, {}, "yzxHDW"),
            BoundingBox::FromStartAndEnd(start_and_end_yzx, {}, "yzxYZX"));
}

}  // namespace test
}  // namespace dali
