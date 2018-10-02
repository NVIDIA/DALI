#include <gtest/gtest.h>

#include "dali/pipeline/operators/crop/bbox_crop.h"

namespace {

using namespace dali;

class BBoxCropInspector : public BBoxCrop {
 public:
  BBoxCropInspector(OpSpec &spec) : BBoxCrop(spec) {}

  std::vector<float> Thresholds() const { return BBoxCrop::thresholds_; }

  BBoxCrop::Bounds AspectRatio() const {
    return BBoxCrop::aspect_ratio_bounds_;
  }

  BBoxCrop::Bounds Scaling() const { return BBoxCrop::scaling_bounds_; }
};

OpSpec DefaultSpec() {
  OpSpec spec("BBoxCrop");
  spec.AddArg("num_threads", 1);
  spec.AddArg("batch_size", 1);

  return spec;
}

TEST(BBoxCropTest, test_that_default_number_of_thresholds_is_one) {
  auto spec = DefaultSpec();
  BBoxCropInspector inspector(spec);

  EXPECT_TRUE(inspector.Thresholds().size() == 1);
}

TEST(BBoxCropTest, test_that_default_threshold_is_zero) {
  auto spec = DefaultSpec();
  BBoxCropInspector inspector(spec);

  EXPECT_TRUE(inspector.Thresholds().at(0) == 0.0);
}

TEST(BBoxCropTest, test_that_provided_thresholds_cannot_be_empty) {
  auto spec = DefaultSpec();
  spec.AddArg("thresholds", std::vector<float>{});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_thresholds_have_lower_bound) {
  auto spec = DefaultSpec();
  spec.AddArg("thresholds", std::vector<float>{-1.0});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_thresholds_have_upper_bound) {
  auto spec = DefaultSpec();
  spec.AddArg("thresholds", std::vector<float>{1.1});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_thresholds_can_be_supplied) {
  auto spec = DefaultSpec();
  spec.AddArg("thresholds", std::vector<float>{0.0, 0.5, 1.0});

  BBoxCropInspector inspector(spec);

  EXPECT_TRUE(inspector.Thresholds().size() == 3);
  EXPECT_TRUE(inspector.Thresholds()[0] == 0.0);
  EXPECT_TRUE(inspector.Thresholds()[1] == 0.5);
  EXPECT_TRUE(inspector.Thresholds()[2] == 1.0);
}

TEST(BBoxCropTest, test_that_thresholds_can_be_supplied2) {
  auto spec = DefaultSpec();
  spec.AddArg("thresholds", std::vector<float>{0.0, 0.25, 0.5, 0.75, 1.0});

  BBoxCropInspector inspector(spec);

  EXPECT_TRUE(inspector.Thresholds().size() == 5);
  EXPECT_TRUE(inspector.Thresholds()[0] == 0.0);
  EXPECT_TRUE(inspector.Thresholds()[1] == 0.25);
  EXPECT_TRUE(inspector.Thresholds()[2] == 0.5);
  EXPECT_TRUE(inspector.Thresholds()[3] == 0.75);
  EXPECT_TRUE(inspector.Thresholds()[4] == 1.0);
}

TEST(BBoxCropTest, test_that_default_bounds_for_aspect_ratio_is_1) {
  auto spec = DefaultSpec();
  BBoxCropInspector inspector(spec);

  EXPECT_TRUE(inspector.AspectRatio().max_ == 1.0);
  EXPECT_TRUE(inspector.AspectRatio().min_ == 1.0);
}

TEST(BBoxCropTest, test_that_provided_aspect_ratio_bounds_are_two_values) {
  auto spec = DefaultSpec();
  spec.AddArg("aspect_ratio", std::vector<float>{1.1});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_aspect_ratio_bounds_are_two_values2) {
  auto spec = DefaultSpec();
  spec.AddArg("aspect_ratio", std::vector<float>{0.0, 0.5, 1.0});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_aspect_ratio_bounds_are_min_and_max) {
  auto spec = DefaultSpec();
  spec.AddArg("aspect_ratio", std::vector<float>{1.0, 0.5});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_aspect_ratio_bounds_are_min_and_max2) {
  auto spec = DefaultSpec();
  spec.AddArg("aspect_ratio", std::vector<float>{0.1, 2.0});

  EXPECT_NO_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_aspect_ratio_min_must_be_at_least_zero) {
  auto spec = DefaultSpec();
  spec.AddArg("aspect_ratio", std::vector<float>{-0.1, 2.0});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_default_bounds_for_scaling_is_1) {
  auto spec = DefaultSpec();
  BBoxCropInspector inspector(spec);

  EXPECT_TRUE(inspector.Scaling().max_ == 1.0);
  EXPECT_TRUE(inspector.Scaling().min_ == 1.0);
}

TEST(BBoxCropTest, test_that_provided_scaling_bounds_are_two_values) {
  auto spec = DefaultSpec();
  spec.AddArg("scaling", std::vector<float>{1.1});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_scaling_bounds_are_two_values2) {
  auto spec = DefaultSpec();
  spec.AddArg("scaling", std::vector<float>{0.0, 0.5, 1.0});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_scaling_bounds_are_min_and_max) {
  auto spec = DefaultSpec();
  spec.AddArg("scaling", std::vector<float>{1.0, 0.5});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_scaling_bounds_are_min_and_max2) {
  auto spec = DefaultSpec();
  spec.AddArg("scaling", std::vector<float>{0.1, 2.0});

  EXPECT_NO_THROW(BBoxCropInspector inspector(spec));
}

TEST(BBoxCropTest, test_that_provided_scaling_min_must_be_at_least_zero) {
  auto spec = DefaultSpec();
  spec.AddArg("scaling", std::vector<float>{-0.1, 2.0});

  EXPECT_ANY_THROW(BBoxCropInspector inspector(spec));
}
}