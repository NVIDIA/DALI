// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  OpSpec DefaultSchema(bool gpu = false) {
    return GenericResizeTest<ImgType>::DefaultSchema("ResizeCropMirror", gpu ? "gpu" : "cpu");
  }
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(ResizeCropMirrorTest, Types);

// Note: lower accuracy due to TJPG and OCV implementations for BGR/RGB.
// Difference is consistent, deterministic and goes away if I force OCV
// instead of TJPG decoding.

TYPED_TEST(ResizeCropMirrorTest, TestFixedResizeAndCropCPU) {
  this->TstBody(this->DefaultSchema(false)
                .AddArg("resize_shorter", 480.f)
                .AddArg("antialias", false)
                .AddArg("crop", vector<float>{224, 224}), 0.2);
}

TYPED_TEST(ResizeCropMirrorTest, TestFixedResizeAndCropGPU) {
  this->TstBody(this->DefaultSchema(true)
                .AddArg("resize_x", 480.f)
                .AddArg("resize_y", 480.f)
                .AddArg("antialias", false)
                .AddArg("crop", vector<float>{224, 224}), 0.2);
}

}  // namespace dali
