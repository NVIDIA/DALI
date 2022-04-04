// Copyright (c) 2019, 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <limits>
#include <vector>
#include <memory>
#include "dali/test/dali_operator_test.h"
#include "dali/test/dali_operator_test_utils.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/convert.h"
#include "dali/operators/image/color/brightness_contrast.h"

namespace dali {

namespace testing {
namespace brightness_contrast {

using dali::brightness_contrast::FullRange;
using dali::brightness_contrast::HalfRange;

using InputDataType = float;


class BrightnessContrastTest : public testing::DaliOperatorTest {
 public:
  BrightnessContrastTest() {
    Init(input_, volume(shape_));
  }


  GraphDescr GenerateOperatorGraph() const override {
    GraphDescr g("BrightnessContrast");
    return g;
  }


  void Init(std::vector<InputDataType> &input, size_t n) {
    std::mt19937_64 rng;
    input.resize(n);
    UniformRandomFill(input, rng, 0.f, 10.f);
  }


  template <typename Backend>
  std::unique_ptr<TensorList<Backend>> ToTensorList(std::vector<InputDataType> data) {
    std::unique_ptr<TensorList<Backend>> tl(new TensorList<Backend>());
    tl->template set_type<InputDataType>();
    tl->Resize(uniform_list_shape(1, shape_));
    auto *ptr = tl->template mutable_tensor<InputDataType>(0);
    assert(data.size() == static_cast<size_t>(volume(shape_)));
    std::memcpy(ptr, data.data(), data.size() * sizeof(InputDataType));
    return tl;
  }


  std::vector<InputDataType> input_;
  TensorShape<3> shape_ = {2, 4, 3};
};

namespace {

static_assert(HalfRange<uint8_t>() == 128.0f, "Half range of uint8_t should be 128");
static_assert(HalfRange<int16_t>() == 16384.0f, "Half range of int16_t should be 2^14");
static_assert(HalfRange<float>() == 0.5f, "Half range of float should be 0.5f");

static_assert(FullRange<uint8_t>() == 255.0f, "Full range of uint8_t should be 255");
static_assert(FullRange<int16_t>() == 32767.0f, "Full range of int16_t should be 2^15-1");
static_assert(FullRange<float>() == 1.0f, "Full range of float should be 1.0f");


template <class OutputType>
void BrightnessContrastVerify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
  static_assert(std::is_fundamental<OutputType>::value, "");
  auto input_tl = input.CopyTo<CPUBackend>();
  auto output_tl = output.CopyTo<CPUBackend>();
  auto brightness = args["brightness"].GetValue<float>();
  auto brightness_shift = args["brightness_shift"].GetValue<float>();
  auto contrast = args["contrast"].GetValue<float>();

  float contrast_offset = HalfRange<InputDataType>();
  float out_range = FullRange<OutputType>();

  ASSERT_EQ(input_tl->num_samples(), output_tl->num_samples());
  for (int t = 0; t < input.cpu().num_samples(); t++) {
    auto out_shape = output_tl->tensor_shape(t);
    auto out_tensor = output_tl->tensor<OutputType>(t);
    auto in_shape = input_tl->tensor_shape(t);
    auto in_tensor = input_tl->tensor<InputDataType>(t);
    ASSERT_EQ(in_shape, out_shape);
    for (int i = 0; i < volume(out_shape); i++) {
      float with_contrast = contrast_offset + contrast*(in_tensor[i] - contrast_offset);
      float with_brighness = brightness * with_contrast;
      float with_shift = out_range * brightness_shift + with_brighness;
      EXPECT_EQ(out_tensor[i], ConvertSat<OutputType>(with_shift));
    }
  }
}


Arguments args1 = {
        {"dtype",             DALI_UINT8},
        {"brightness" ,       1.f},
        {"brightness_shift",  0.f},
        {"contrast",          1.f}
};
Arguments args2 = {
        {"dtype",             DALI_UINT8},
        {"brightness" ,       1.f},
        {"brightness_shift",  0.1f},
        {"contrast",          2.f}
};
Arguments args3 = {
        {"dtype",             DALI_UINT8},
        {"brightness" ,       0.5f},
        {"brightness_shift",  0.051f},
        {"contrast",          1.f}
};

std::vector<Arguments> args_for_types = {args1, args2, args3};

}  // namespace

INSTANTIATE_TEST_SUITE_P(BrightnessContrastTest, BrightnessContrastTest,
                         ::testing::ValuesIn(testing::cartesian(utils::kDevices, args_for_types)));


TEST_P(BrightnessContrastTest, basic_test_float) {
  auto tl = ToTensorList<CPUBackend>(this->input_);
  auto args = GetParam();
  TensorListWrapper tlout;
  this->RunTest(tl.get(), tlout, args, BrightnessContrastVerify<uint8_t>);
}


TEST_P(BrightnessContrastTest, basic_test_int16) {
  auto tl = ToTensorList<CPUBackend>(this->input_);
  auto args = GetParam();
  TensorListWrapper tlout;
  this->RunTest(tl.get(), tlout, args, BrightnessContrastVerify<uint8_t>);
}


TEST_P(BrightnessContrastTest, basic_test_uint8) {
  auto tl = ToTensorList<CPUBackend>(this->input_);
  auto args = GetParam();
  TensorListWrapper tlout;
  this->RunTest(tl.get(), tlout, args, BrightnessContrastVerify<uint8_t>);
}


}  // namespace brightness_contrast
}  // namespace testing
}  // namespace dali
