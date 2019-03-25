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

#include "dali/pipeline/operators/decoder/decoder_test.h"

namespace dali {

template <typename ImgType>
class HostDecodeTest : public DecodeTestBase<ImgType> {
 protected:
  OpSpec DecodingOp() const override {
    return this->GetOpSpec("HostDecoder");
  }
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(HostDecodeTest, Types);

TYPED_TEST(HostDecodeTest, JpegDecode) {
  this->Run(t_jpegImgType);
}

TYPED_TEST(HostDecodeTest, PngDecode) {
  this->Run(t_pngImgType);
}

TYPED_TEST(HostDecodeTest, TiffDecode) {
  this->Run(t_tiffImgType);
}

}  // namespace dali
