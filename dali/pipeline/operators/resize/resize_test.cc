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
class ResizeTest : public GenericResizeTest<ImgType>  {
 protected:
  uint32_t getResizeOptions() const override          { return 0/*t_externSizes*/; }
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(ResizeTest, Types);

TYPED_TEST(ResizeTest, TestResizeShorter) {
  this->TstBody(this->DefaultSchema("Resize")
                      .AddArg("resize_shorter", 480.f), 14.0);
}

TYPED_TEST(ResizeTest, TestResizeShorter_A) {
  this->TstBody(this->DefaultSchema("Resize")
                      .AddArg("resize_shorter", 224.f), 14.0);
}

TYPED_TEST(ResizeTest, TestResizeXY) {
  this->TstBody(this->DefaultSchema("Resize")
                      .AddArg("resize_x", 224.f)
                      .AddArg("resize_y", 224.f), 14.0);
}

TYPED_TEST(ResizeTest, TestResizeXY_A) {
  this->TstBody(this->DefaultSchema("Resize")
                      .AddArg("resize_x", 240.f)
                      .AddArg("resize_y", 480.f), 14.0);
}

}  // namespace dali
