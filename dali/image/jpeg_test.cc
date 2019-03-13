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

#include "dali/test/dali_test_decoder.h"

namespace dali {

// Fixture for jpeg decode testing. Templated
// to make googletest run our tests grayscale & rgb
template <typename ImgType>
class JpegDecodeTest : public GenericDecoderTest<ImgType> {
};

// Run RGB & grayscale tests
typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(JpegDecodeTest, Types);

TYPED_TEST(JpegDecodeTest, DecodeJPEGHost) {
  this->RunTestDecode(this->jpegs_);
}

}  // namespace dali
