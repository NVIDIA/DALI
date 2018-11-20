//
// Created by pribalta on 19.11.18.
//

#include <gtest/gtest.h>

#include "bounding_box.h"

namespace {
using namespace dali;

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptNegativeCoordinates) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(-0.5, 0.5, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptNegativeCoordinates2) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, -0.5, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptNegativeCoordinates3) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 0.5, -0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptNegativeCoordinates4) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 0.5, 0.5, -0.5));
}
TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptCoordinatesLargerThanOne) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(1.1, 0.5, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptCoordinatesLargerThanOne2) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 1.1, 0.5, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptCoordinatesLargerThanOne3) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 0.5, 1.1, 0.5));
}

TEST(BoundingBoxTest, BoundingBoxLTRBDoesNotAcceptCoordinatesLargerThanOne4) {
  EXPECT_ANY_THROW(BoundingBox::FromLtrb(0.5, 0.5, 0.5, 1.1));
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

}  // namespace