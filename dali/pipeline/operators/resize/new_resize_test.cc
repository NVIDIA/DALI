// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/test/dali_test_resize.h"

namespace dali {

template <typename ImgType>
class NewResizeTest : public GenericResizeTest<ImgType> {
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(NewResizeTest, Types);

TYPED_TEST(NewResizeTest, TestFixedResizeAndCrop) {
  this->TstBody(this->DefaultSchema("ResizeCropMirror")
                  .AddArg("resize_shorter", 480.f)
                  .AddArg("crop", vector<int>{224, 224}), 0.85);
}

TYPED_TEST(NewResizeTest, TestFixedResizeAndCropWarp) {
  this->TstBody(this->DefaultSchema("ResizeCropMirror")
                  .AddArg("resize_x", 480.f)
                  .AddArg("resize_y", 480.f)
                  .AddArg("crop", vector<int>{224, 224}), 0.85);
}

}  // namespace dali
