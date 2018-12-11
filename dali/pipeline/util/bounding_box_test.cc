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

namespace {

using dali::BoundingBox;

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

TEST(BoundingBoxTest, CanBeClampedToSmaller) {
  auto bigger_box = BoundingBox::FromXywh(0.0, 0.0, 1.0, 1.0);
  auto smaller_box = BoundingBox::FromXywh(0.0, 0.0, 0.1, 0.1);

  auto clamped = bigger_box.ClampTo(smaller_box);

  EXPECT_EQ(clamped.Area(), smaller_box.Area());
}

TEST(BoundingBoxTest, ClampToBiggerLeavesTheSame) {
  auto bigger_box = BoundingBox::FromXywh(0.0, 0.0, 1.0, 1.0);
  auto smaller_box = BoundingBox::FromXywh(0.0, 0.0, 0.1, 0.1);

  auto clamped = smaller_box.ClampTo(bigger_box);

  EXPECT_EQ(clamped.Area(), smaller_box.Area());
}

TEST(BoundingBoxTest, CanFlipHorizontallyXYWH) {
  auto box_initial = BoundingBox::FromXywh(0.25, 0.25, 0.25, 0.25);
  auto coords_initial_ltrb = box_initial.AsLtrb();
  auto coords_initial_xywh = box_initial.AsXywh();

  auto box = BoundingBox::FromXywh(0.25, 0.25, 0.25, 0.25);

  box = box.HorizontalFlip();
  box = box.HorizontalFlip();

  auto coords_flipped_ltrb = box.AsLtrb();
  auto coords_flipped_xywh = box.AsXywh();

  for (size_t i = 0; i < BoundingBox::kSize; i++) {
    EXPECT_EQ(coords_initial_ltrb[i], coords_flipped_ltrb[i]);
    EXPECT_EQ(coords_initial_xywh[i], coords_flipped_xywh[i]);
  }
}

TEST(BoundingBoxTest, CanFlipHorizontallyLTRB) {
  auto box_initial = BoundingBox::FromLtrb(0.25, 0.25, 0.5, 0.5);
  auto coords_initial_ltrb = box_initial.AsLtrb();
  auto coords_initial_xywh = box_initial.AsXywh();

  auto box = BoundingBox::FromLtrb(0.25, 0.25, 0.5, 0.5);

  box = box.HorizontalFlip();
  box = box.HorizontalFlip();

  auto coords_flipped_ltrb = box.AsLtrb();
  auto coords_flipped_xywh = box.AsXywh();

  for (size_t i = 0; i < BoundingBox::kSize; i++) {
    EXPECT_EQ(coords_initial_ltrb[i], coords_flipped_ltrb[i]);
    EXPECT_EQ(coords_initial_xywh[i], coords_flipped_xywh[i]);
  }
}

TEST(BoundingBoxTest, CanFlipVerticallyXYWH) {
  auto box_initial = BoundingBox::FromXywh(0.25, 0.25, 0.25, 0.25);
  auto coords_initial_ltrb = box_initial.AsLtrb();
  auto coords_initial_xywh = box_initial.AsXywh();

  auto box = BoundingBox::FromXywh(0.25, 0.25, 0.25, 0.25);

  box = box.VerticalFlip();
  box = box.VerticalFlip();

  auto coords_flipped_ltrb = box.AsLtrb();
  auto coords_flipped_xywh = box.AsXywh();

  for (size_t i = 0; i < BoundingBox::kSize; i++) {
    EXPECT_EQ(coords_initial_ltrb[i], coords_flipped_ltrb[i]);
    EXPECT_EQ(coords_initial_xywh[i], coords_flipped_xywh[i]);
  }
}

TEST(BoundingBoxTest, CanFlipVerticallyLTRB) {
  auto box_initial = BoundingBox::FromLtrb(0.25, 0.25, 0.5, 0.5);
  auto coords_initial_ltrb = box_initial.AsLtrb();
  auto coords_initial_xywh = box_initial.AsXywh();

  auto box = BoundingBox::FromLtrb(0.25, 0.25, 0.5, 0.5);

  box = box.VerticalFlip();
  box = box.VerticalFlip();

  auto coords_flipped_ltrb = box.AsLtrb();
  auto coords_flipped_xywh = box.AsXywh();

  for (size_t i = 0; i < BoundingBox::kSize; i++) {
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

}  // namespace
