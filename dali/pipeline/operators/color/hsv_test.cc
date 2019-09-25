// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include <memory>
#include "dali/test/dali_operator_test.h"
#include "dali/test/dali_operator_test_utils.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/core/convert.h"
#include "hsv.h"

namespace dali {

namespace testing {


using InputDataType = float;


class HsvTest : public testing::DaliOperatorTest {
 public:
  HsvTest() {
    Init(input_, volume(shape_));
  }


  GraphDescr GenerateOperatorGraph() const override {
    GraphDescr g("Hsv");
    return g;
  }


  void Init(std::vector<InputDataType> &input, size_t n) {
    std::mt19937_64 rng;
    input.resize(n);
    kernels::UniformRandomFill(input, rng, 0.f, 10.f);
  }


  template <typename Backend>
  std::unique_ptr<TensorList<Backend>> ToTensorList(std::vector<InputDataType> data) {
    std::unique_ptr<TensorList<Backend>> tl(new TensorList<Backend>());
    tl->Resize(kernels::TensorListShape<3>({shape_}));
    auto ptr = tl->template mutable_data<InputDataType>();
    assert(data.size() == static_cast<size_t>(volume(shape_)));
    std::memcpy(ptr, data.data(), data.size() * sizeof(InputDataType));
    return tl;
  }


  std::vector<InputDataType> input_;
  kernels::TensorShape<3> shape_ = {2, 4, 3};
};

namespace {


template <class OutputType>
void HsvVerify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
  static_assert(std::is_fundamental<OutputType>::value, "");
  auto input_tl = input.CopyTo<CPUBackend>();
  auto output_tl = output.CopyTo<CPUBackend>();
  auto hue = args[hsv::kHue.c_str()].GetValue<float>();
  auto saturation = args[hsv::kSaturation.c_str()].GetValue<float>();
  auto value = args[hsv::kValue.c_str()].GetValue<float>();
  ASSERT_EQ(input_tl->ntensor(), output_tl->ntensor());
//  for (size_t t = 0; t < input.cpu().ntensor(); t++) {
//    auto out_shape = output_tl->tensor_shape(t);
//    auto out_tensor = output_tl->tensor<OutputType>(t);
//    auto in_shape = input_tl->tensor_shape(t);
//    auto in_tensor = input_tl->tensor<InputDataType>(t);
//    ASSERT_EQ(in_shape, out_shape);
//    for (int i = 0; i < volume(out_shape); i++) {
//      EXPECT_EQ(out_tensor[i], ConvertSat<OutputType>(in_tensor[i] * contrast + brightness));
//    }
//  }
}


Arguments args1 = {
        {"output_type",      DALI_UINT8},
        {"brightness_delta", 0.f},
        {"contrast_delta",   1.f}
};
Arguments args2 = {
        {"output_type",      DALI_UINT8},
        {"brightness_delta", 1.f},
        {"contrast_delta",   0.f}
};
Arguments args3 = {
        {"output_type",      DALI_UINT8},
        {"brightness_delta", 13.f},
        {"contrast_delta",   0.5f}
};

std::vector<Arguments> args_for_types = {args1, args2, args3};

}  // namespace

INSTANTIATE_TEST_SUITE_P(HsvTest, HsvTest,
                         ::testing::ValuesIn(testing::cartesian(utils::kDevices, args_for_types)));


TEST_P(HsvTest, basic_test_float) {
  auto tl = ToTensorList<CPUBackend>(this->input_);
  auto args = GetParam();
  TensorListWrapper tlout;
  this->RunTest(tl.get(), tlout, args, HsvVerify<uint8_t>);
}


TEST_P(HsvTest, basic_test_int16) {
  auto tl = ToTensorList<CPUBackend>(this->input_);
  auto args = GetParam();
  TensorListWrapper tlout;
  this->RunTest(tl.get(), tlout, args, HsvVerify<uint8_t>);
}


TEST_P(HsvTest, basic_test_uint8) {
  auto tl = ToTensorList<CPUBackend>(this->input_);
  auto args = GetParam();
  TensorListWrapper tlout;
  this->RunTest(tl.get(), tlout, args, HsvVerify<uint8_t>);
}


}  // namespace testing
}  // namespace dali
