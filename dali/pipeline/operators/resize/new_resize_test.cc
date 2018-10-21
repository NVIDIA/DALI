// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/test/dali_test_resize.h"

namespace dali {

// Values of eps for different tests
//      first column:   average deviation is calculated combined for all images
//      second column:  average deviation is calculated separately for each image

static double testEps[] = {
                    0.32, 0.9,
                    0.31, 0.6,
};

template <typename ImgType>
class NewResizeTest : public GenericResizeTest<ImgType> {
 protected:
  double *testEpsValues() const override            { return testEps; }
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(NewResizeTest, Types);

typedef enum {
  t_TestFixedResizeAndCrop,
  t_TestFixedResizeAndCropWarp,
} testTID;

TYPED_TEST(NewResizeTest, TestFixedResizeAndCrop) {
  this->TstBody(this->DefaultSchema("ResizeCropMirror")
                  .AddArg("resize_shorter", 480.f)
                  .AddArg("crop", vector<float>{224, 224}),
                this->getEps(t_TestFixedResizeAndCrop));
}

TYPED_TEST(NewResizeTest, TestFixedResizeAndCropWarp) {
  this->TstBody(this->DefaultSchema("ResizeCropMirror")
                  .AddArg("resize_x", 480.f)
                  .AddArg("resize_y", 480.f)
                  .AddArg("crop", vector<float>{224, 224}),
                this->getEps(t_TestFixedResizeAndCropWarp));
}

}  // namespace dali
