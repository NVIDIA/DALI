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

#include "dali/pipeline/data/tensor.h"
#include "dali/test/dali_operator_test.h"
#include "dali/test/dali_operator_test_utils.h"
#include "dali/pipeline/operators/color/brightness_contrast.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/kernels/test/tensor_test_utils.h"

namespace dali {


namespace testing {
namespace brightness_contrast {

template<size_t ndims>
using Shape = ::dali::kernels::TensorListShape<ndims>;

using TestSample = float[2][2][2][3][3];
constexpr size_t kNumberOfThingies = 2*2*2*3*3;

TestSample data_nhwc = {{{{{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}},
                          {{4.1, 4.2, 4.3}, {5.1, 5.2, 5.3}, {6.1, 6.2, 6.3}}},
                         {{{4.1, 4.2, 4.3}, {5.1, 5.2, 5.3}, {6.1, 6.2, 6.3}},
                          {{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}}}},
                        {{{{3.1, 3.2, 3.3}, {2.1, 2.2, 2.3}, {1.1, 1.2, 1.3}},
                          {{6.1, 6.2, 6.3}, {5.1, 5.2, 5.3}, {4.1, 4.2, 4.3}}},
                         {{{6.1, 6.2, 6.3}, {5.1, 5.2, 5.3}, {4.1, 4.2, 4.3}},
                          {{3.1, 3.2, 3.3}, {2.1, 2.2, 2.3}, {1.1, 1.2, 1.3}}}}};

template<class Backend,  size_t ndims>
std::unique_ptr<TensorList<Backend>> ToTensorList(const float *sample,const Shape<ndims>& shape) {
  std::unique_ptr<TensorList<Backend>> tl(new TensorList<Backend>());
  tl->Resize(shape);
  auto ptr =tl->template mutable_data<float>();
  assert(kNumberOfThingies== volume(shape));
  std::memcpy(ptr, sample, volume(shape)*sizeof(float));
}

using Type = float;



class BrightnessContrastTest : public testing::DaliOperatorTest {
 public:
  BrightnessContrastTest() {
      init(input_, volume(shape_));
    }
  GraphDescr GenerateOperatorGraph() const override {
    GraphDescr g("BrightnessContrast");
    return g;
  }

  void init(std::vector<Type>& input, size_t n) {
    std::mt19937_64 rng;
    input.resize(n);
    kernels::UniformRandomFill(input, rng, 0.f, 10.f);
  }

  template<typename Backend>
  std::unique_ptr<TensorList<Backend>> ToTensorList(std::vector<Type> data) {
    std::unique_ptr<TensorList<Backend>> tl(new TensorList<Backend>());
    tl->Resize(kernels::TensorListShape<3>({shape_}));
    auto ptr = tl->template mutable_data<float>();
    assert(data.size() == volume(shape_));
    std::memcpy(ptr, data.data(), data.size()*sizeof(Type));
    return tl;
  }

  std::vector<Type> input_;
  kernels::TensorShape<3> shape_ = {2,4,3};

};

constexpr int ndims = 3;
float brightness=0;

float contrast =1;

std::vector<Type> calc_output(std::vector<Type> input) {
  std::vector<Type> ret;
  ret.reserve(input.size());
for (const auto& i:input) {
    ret.emplace_back(i*contrast+brightness);
  }
  return ret;
}


void BrightnessContrastVerify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
  auto brightness = args["brightness_delta"].GetValue<float>();
  auto contrast = args["contrast_delta"].GetValue<float>();
  assert(input.has_cpu() && output.has_cpu());
  ASSERT_EQ(input.cpu().ntensor(), output.cpu().ntensor());
  for (int t = 0; t < input.cpu().ntensor(); t++) {
    auto in_shape = input.cpu().tensor_shape(t);
    auto out_shape = output.cpu().tensor_shape(t);
    auto in_tensor = input.cpu().tensor<Type>(t);
    auto out_tensor = output.cpu().tensor<Type>(t);
    ASSERT_EQ(volume(in_shape), volume(out_shape));
    for (int i = 0; i < volume(in_shape); i++) {
      EXPECT_EQ(in_tensor[i], out_tensor[i]);
    }

  }
}

Arguments arg = {{"output_type", DALI_FLOAT},{"brightness_delta", 0.f},{"contrast_delta",1.f}};

TEST_F(BrightnessContrastTest, BasicTest) {
  auto tl = ToTensorList<CPUBackend>(this->input_);
  TensorListWrapper tlout;
  this->RunTest(tl.get(), tlout, arg, BrightnessContrastVerify);
}
}

}
}