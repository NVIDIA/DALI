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

#include "dali/test/dali_test_resize.h"

namespace dali {

template <typename ImgType>
class ResizeCropMirrorTest : public GenericResizeTest<ImgType> {
 protected:
  // tell compiler we don't want to hide overloaded DefaultSchema
  using GenericResizeTest<ImgType>::DefaultSchema;

  OpSpec DefaultSchema(bool fast_resize = false) {
    const char *op = (fast_resize) ? "FastResizeCropMirror"
                                   : "ResizeCropMirror";
    return GenericResizeTest<ImgType>::DefaultSchema(op, "cpu");
  }
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(ResizeCropMirrorTest, Types);

// Note: lower accuracy due to TJPG and OCV implementations for BGR/RGB.
// Difference is consistent, deterministic and goes away if I force OCV
// instead of TJPG decoding.

TYPED_TEST(ResizeCropMirrorTest, TestFixedResizeAndCrop) {
  this->TstBody(this->DefaultSchema()
                .AddArg("resize_shorter", 480.f)
                .AddArg("crop", vector<float>{224, 224}), 5e-6);
}

TYPED_TEST(ResizeCropMirrorTest, TestFixedResizeAndCropWarp) {
  this->TstBody(this->DefaultSchema()
                .AddArg("resize_x", 480.f)
                .AddArg("resize_y", 480.f)
                .AddArg("crop", vector<float>{224, 224}), 5e-6);
}

TYPED_TEST(ResizeCropMirrorTest, TestFixedFastResizeAndCrop) {
  this->TstBody(this->DefaultSchema(true)
                .AddArg("resize_shorter", 480.f)
                .AddArg("crop", vector<float>{224, 224}), 1.98);
}

TYPED_TEST(ResizeCropMirrorTest, TestFixedFastResizeAndCropWarp) {
  this->TstBody(this->DefaultSchema(true)
                    .AddArg("resize_x", 480.f)
                    .AddArg("resize_y", 480.f)
                    .AddArg("crop", vector<float>{224, 224}),
                1.80);
}

}  // namespace dali
