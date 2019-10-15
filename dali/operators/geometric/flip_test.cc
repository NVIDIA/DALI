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

namespace dali {

namespace testing {

const int kDataWidth = 3;
const int kDataHeight = 2;
const int kDataChannels = 3;

struct TestTlData {
  std::vector<float> data;

  explicit TestTlData(void *data_ptr) noexcept
      : data(2 * kDataWidth * kDataHeight * kDataChannels) {
    auto data_size = kDataWidth * kDataHeight * kDataChannels * sizeof(float);
    std::memcpy(data.data(), data_ptr, data_size);
    std::memcpy(data.data() + data_size / sizeof(float), data_ptr, data_size);
  }

  float *ptr() { return data.data(); }
};

float data_nhwc[2][2][2][3][3] = {{{{{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}},
                                    {{4.1, 4.2, 4.3}, {5.1, 5.2, 5.3}, {6.1, 6.2, 6.3}}},
                                   {{{4.1, 4.2, 4.3}, {5.1, 5.2, 5.3}, {6.1, 6.2, 6.3}},
                                    {{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}}}},
                                  {{{{3.1, 3.2, 3.3}, {2.1, 2.2, 2.3}, {1.1, 1.2, 1.3}},
                                    {{6.1, 6.2, 6.3}, {5.1, 5.2, 5.3}, {4.1, 4.2, 4.3}}},
                                   {{{6.1, 6.2, 6.3}, {5.1, 5.2, 5.3}, {4.1, 4.2, 4.3}},
                                    {{3.1, 3.2, 3.3}, {2.1, 2.2, 2.3}, {1.1, 1.2, 1.3}}}}};

TestTlData nhwc_tensor_list_data(&data_nhwc[0][0]);

float data_nchw[2][2][3][2][3] = {{{{{1.1, 2.1, 3.1}, {4.1, 5.1, 6.1}},
                                    {{1.2, 2.2, 3.2}, {4.2, 5.2, 6.2}},
                                    {{1.3, 2.3, 3.3}, {4.3, 5.3, 6.3}}},
                                   {{{4.1, 5.1, 6.1}, {1.1, 2.1, 3.1}},
                                    {{4.2, 5.2, 6.2}, {1.2, 2.2, 3.2}},
                                    {{4.3, 5.3, 6.3}, {1.3, 2.3, 3.3}}}},
                                  {{{{3.1, 2.1, 1.1}, {6.1, 5.1, 4.1}},
                                    {{3.2, 2.2, 1.2}, {6.2, 5.2, 4.2}},
                                    {{3.3, 2.3, 1.3}, {6.3, 5.3, 4.3}}},
                                   {{{6.1, 5.1, 4.1}, {3.1, 2.1, 1.1}},
                                    {{6.2, 5.2, 4.2}, {3.2, 2.2, 1.2}},
                                    {{6.3, 5.3, 4.3}, {3.3, 2.3, 1.3}}}}};

TestTlData nchw_tensor_list_data(&data_nchw[0][0]);

class FlipTest : public testing::DaliOperatorTest {
  GraphDescr GenerateOperatorGraph() const override {
    GraphDescr graph("Flip");
    return graph;
  }
};

std::vector<Arguments> arguments = {{{"horizontal", 0}, {"vertical", 0}},
                                    {{"horizontal", 1}, {"vertical", 0}},
                                    {{"horizontal", 0}, {"vertical", 1}},
                                    {{"horizontal", 1}, {"vertical", 1}}};

std::vector<Arguments> layout = {{{"hwc", true}}, {{"hwc", false}}};

void FlipVerify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
  int _horizontal = args["horizontal"].GetValue<int>();
  int _vertical = args["vertical"].GetValue<int>();
  auto output_d = output.CopyTo<CPUBackend>();
  auto item_size = output_d->type().size();
  for (size_t i = 0; i < output_d->ntensor(); ++i) {
    auto size =
        output_d->tensor_shape(i)[0] * output_d->tensor_shape(i)[1] * output_d->tensor_shape(i)[2];
    auto out_tensor = output_d->raw_tensor(i);
    if (output_d->GetLayout() == "HWC") {
      ASSERT_EQ(std::memcmp(out_tensor, &data_nhwc[_horizontal][_vertical], size * item_size), 0);
    } else if (output_d->GetLayout() == "CHW") {
      ASSERT_EQ(std::memcmp(out_tensor, &data_nchw[_horizontal][_vertical], size * item_size), 0);
    }
  }
}

TEST_P(FlipTest, BasicTest) {
  auto args = GetParam();
  auto hwc = args["hwc"].GetValue<bool>();
  auto data_size = kDataWidth * kDataHeight * kDataChannels * sizeof(float);
  TensorList<CPUBackend> tl;
  if (hwc) {
    tl.ShareData(nhwc_tensor_list_data.ptr(), 2 * data_size);
    tl.set_type(TypeInfo::Create<float>());
    tl.SetLayout("HWC");
    auto shape = TensorListShape<>{{
        {kDataHeight, kDataWidth, kDataChannels},
        {kDataHeight, kDataWidth, kDataChannels}}};
    tl.Resize(shape);
  } else {
    tl.ShareData(nchw_tensor_list_data.ptr(), 2 * data_size);
    tl.set_type(TypeInfo::Create<float>());
    tl.SetLayout("CHW");
    tl.Resize({{{kDataChannels, kDataHeight, kDataWidth},
              {kDataChannels, kDataHeight, kDataWidth}}});
  }
  TensorListWrapper tlout;
  this->RunTest(&tl, tlout, args, FlipVerify);
}

INSTANTIATE_TEST_SUITE_P(FlipTest, FlipTest,
                         ::testing::ValuesIn(
                             testing::cartesian(testing::utils::kDevices, arguments, layout)));

}  // namespace testing
}  // namespace dali
