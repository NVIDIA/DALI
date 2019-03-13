// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include <limits>

#include "dali/pipeline/operators/crop/kernel/coords.h"
#include "dali/pipeline/operators/crop/kernel/crop_kernel.h"

namespace dali {
namespace detail {

template <typename InT_, typename OutT_, bool is_seq_, Index S_, Index H_, Index W_, Index C_,
          Index startH_, Index startW_, Index outH_, Index outW_>
struct info {
  using InT = InT_;
  using OutT = OutT_;
  static const bool is_seq = is_seq_;
  static const Index S = S_;
  static const Index H = H_;
  static const Index W = W_;
  static const Index C = C_;
  static const Index startH = startH_;
  static const Index startW = startW_;
  static const Index outH = outH_;
  static const Index outW = outW_;
};

template <typename T>
class CropTest : public ::testing::Test {
 public:
  void SetUp() override {
    constexpr size_t in_size = T::S * T::H * T::W * T::C;
    constexpr size_t out_size = T::S * T::outH * T::outW * T::C;
    input = new typename T::InT[in_size];
    output = new typename T::OutT[out_size];
    for (size_t i = 0; i < in_size; i++) {
      input[i] = i % std::numeric_limits<typename T::InT>::max();
    }
    for (size_t i = 0; i < out_size; i++) {
      output[i] = 0;
    }
  }

  void TearDown() override {
    for (Index s = 0; s < T::S; s++) {
      auto in_off = s * T::H * T::W * T::C;
      auto out_off = s * T::outH * T::outW * T::C;
      verify_crop(input + in_off, output + out_off, s);
    }
    delete[] input;
    delete[] output;
  }

  typename T::InT *input;
  typename T::OutT *output;

 private:
  void verify_crop(typename T::InT *in_ptr, typename T::OutT *out_ptr, Index s) {
    for (Index h = 0; h < T::outH; h++) {
      for (Index w = 0; w < T::outW; w++) {
        for (Index c = 0; c < T::C; c++) {
          auto in_off = (T::startH + h) * T::W * T::C + (T::startW + w) * T::C + c;
          auto out_off = h * T::outW * T::C + w * T::C + c;
          ASSERT_EQ(in_ptr[in_off], out_ptr[out_off])
              << " seq: " << s << " (" << h << ", " << w << ", " << c << ")";
        }
      }
    }
  }
};

using CropTypes =
    ::testing::Types<info<uint8_t, uint8_t, false, 1, 1080, 1920, 3, 243, 333, 27, 500>,
                     info<int32_t, float, false, 1, 1080, 1920, 3, 243, 333, 27, 500>>;

template <typename T>
using BasicCropTest = CropTest<T>;

TYPED_TEST_SUITE(BasicCropTest, CropTypes);

TYPED_TEST(BasicCropTest, FlatCrop) {
  std::array<Index, 3> in_shape = {TypeParam::H, TypeParam::W, TypeParam::C};
  std::array<Index, 3> out_shape = {TypeParam::outH, TypeParam::outW, TypeParam::C};

  CropKernel<typename TypeParam::InT, typename TypeParam::OutT,
                   dali_index_sequence<0, 1, 2>>::Run(this->input, in_shape,
                                                                  {TypeParam::startH,
                                                                   TypeParam::startW,
                                                                   TypeParam::outH,
                                                                   TypeParam::outW},
                                                                  this->output, out_shape);
}

using CropSequenceTypes =
    ::testing::Types<info<uint8_t, uint8_t, false, 5, 1080, 1920, 3, 243, 333, 27, 500>,
                     info<int32_t, float, false, 5, 1080, 1920, 3, 243, 333, 27, 500>>;

template <typename T>
using SequenceCropTest = CropTest<T>;

TYPED_TEST_SUITE(SequenceCropTest, CropSequenceTypes);

TYPED_TEST(SequenceCropTest, SequenceCrop) {
  std::array<Index, 4> in_shape = {TypeParam::S, TypeParam::H, TypeParam::W, TypeParam::C};
  std::array<Index, 4> out_shape = {TypeParam::S, TypeParam::outH, TypeParam::outW, TypeParam::C};

  SequenceCropKernel<typename TypeParam::InT, typename TypeParam::OutT,
                           dali_index_sequence<0, 1, 2>>::Run(this->input, in_shape,
                                                                          {TypeParam::startH,
                                                                           TypeParam::startW,
                                                                           TypeParam::outH,
                                                                           TypeParam::outW},
                                                                          this->output, out_shape);
}

}  // namespace detail
}  // namespace dali
