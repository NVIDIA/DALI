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

// 1-d elements: ((5),(2),(3)) -> ((5), (5), (5)) - axes = [0]
std::vector<std::vector<float> > batch_axes_0 = {{3, 4, 2, 5, 4},
                                                 {2, 2},
                                                 {3, 199, 5}};
std::vector<std::vector<float> > padded_batch_axes_0 = {{3, 4, 2, 5, 4},
                                                        {2, 2, -1, -1, -1},
                                                        {3, 199, 5, -1, -1}};

// 2-d elements: ((2, 4), (2, 2)) -> ((2, 4), (2, 4)) - axes = [1]
std::vector<std::vector<int> > batch_axes_1 = {{1, 2 , 3, 4,
                                                5, 6, 7, 8},
                                               {1, 2,
                                                4, 5}};
std::vector<std::vector<int> > padded_batch_axes_1 = {{1, 2 , 3, 4,
                                                       5, 6, 7, 8},
                                                      {1, 2, 42, 42,
                                                       4, 5, 42, 42}};

// 2-d elements: ((2, 4), (3, 2)) -> ((3, 4), (3, 4)) - axes = []
std::vector<std::vector<int> > batch_axes_empty = {{1, 2 , 3, 4,
                                                    5, 6, 7, 8},
                                                   {1, 2,
                                                    4, 5,
                                                    7, 8}};
std::vector<std::vector<int> > padded_batch_axes_empty = {{1,  2,  3,  4,
                                                           5,  6,  7,  8,
                                                           0, 0, 0, 0},
                                                          {1,  2,  0, 0,
                                                           4,  5,  0, 0,
                                                           7,  8,  0, 0}};

// 3-d elements: ((1, 2, 2), (2, 1, 1)) -> ((2, 2, 2), (2, 1, 2)) - axes = [0, 2]
std::vector<std::vector<int> > batch_axes_02 = {{1, 2,
                                                 3, 4},
                                                {1,
                                                 2}};
std::vector<std::vector<int> > padded_batch_axes_02 = {{1,  2,
                                                        3,  4,
                                                        0, 0,
                                                        0, 0},
                                                       {1,  0,
                                                        2,  0}};

template <typename T>
std::vector<std::vector<T> > GetPaddedBatchForaxes(std::vector<int> axes) {
  return {};
}

template <>
std::vector<std::vector<float> > GetPaddedBatchForaxes(std::vector<int> axes) {
  if (axes[0] == 0) {
    return padded_batch_axes_0;
  }
  return {};
}

template <>
std::vector<std::vector<int> > GetPaddedBatchForaxes(std::vector<int> axes) {
  if (axes.empty()) {
    return padded_batch_axes_empty;
  } else if (axes.size() == 2) {
    return padded_batch_axes_02;
  } else if (axes[0] == 1) {
    return padded_batch_axes_1;
  }
  return {};
}

template <typename T>
void PadVerify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
  auto output_d = output.CopyTo<CPUBackend>();
  std::vector<int> axes = args["axes"].GetValue<std::vector<int>>();

  auto padded_batch = GetPaddedBatchForaxes<T>(axes);

  for (size_t i = 0; i < output_d->ntensor(); ++i) {
    auto out_tensor = output_d->tensor<T>(i);
    for (size_t j = 0; j < padded_batch[i].size(); ++j) {
      ASSERT_EQ(out_tensor[j], padded_batch[i][j]);
    }
  }
}

class PadTest : public testing::DaliOperatorTest {
  GraphDescr GenerateOperatorGraph() const override {
    GraphDescr graph("Pad");
    return graph;
  }

 protected:
  template <typename T>
  void RunPadTest(std::vector<std::vector<T> >& batch,
                  const TensorListShape<> tl_shape) {
    auto args = GetParam();
    testing::TensorListWrapper tl_out;
    TensorList<CPUBackend> tl_in;
    tl_in.Resize(tl_shape);
    tl_in.set_type(TypeInfo::Create<T>());
    for (size_t i = 0; i < tl_in.ntensor(); ++i) {
      auto t = tl_in.mutable_tensor<T>(i);
      std::copy(batch[i].begin(), batch[i].end(), t);
    }
    this->RunTest(&tl_in, tl_out, args, PadVerify<T>);
  }
};

class PadBasicTest : public PadTest {};
class Pad2DTest : public PadTest {};
class PadAllAxesTest : public PadTest {};
class Pad3D2AxesTest : public PadTest {};


std::vector<Arguments> basic_args = {{{"fill_value", -1.0f}, {"axes", std::vector<int>{0}}}};

std::vector<Arguments> two_d_args = {{{"fill_value", 42.0f}, {"axes", std::vector<int>{1}}}};

std::vector<Arguments> all_axes_args = {{{"fill_value", 0.0f}, {"axes", std::vector<int>{}}}};

std::vector<Arguments> two_axes_args = {{{"fill_value", 0.0f}, {"axes", std::vector<int>{0, 2}}}};


std::vector<Arguments> devices = {
    {{"device", std::string{"cpu"}}},
    {{"device", std::string{"gpu"}}},
};

TEST_P(PadBasicTest, BasicTest) {
  RunPadTest(batch_axes_0, {{5}, {2}, {3}});
}

TEST_P(Pad2DTest, Test2D) {
  RunPadTest(batch_axes_1, {{2, 4}, {2, 2}});
}

TEST_P(PadAllAxesTest, TestAllAxes) {
  RunPadTest(batch_axes_empty, {{2, 4}, {3, 2}});
}

TEST_P(Pad3D2AxesTest, Test3D2Axes) {
  RunPadTest(batch_axes_02, {{1, 2, 2}, {2, 1, 1}});
}

INSTANTIATE_TEST_SUITE_P(PadBasicTest, PadBasicTest,
                        ::testing::ValuesIn(cartesian(devices, basic_args)));

INSTANTIATE_TEST_SUITE_P(Pad2DTest, Pad2DTest,
                        ::testing::ValuesIn(cartesian(devices, two_d_args)));

INSTANTIATE_TEST_SUITE_P(PadAllAxesTest, PadAllAxesTest,
                        ::testing::ValuesIn(cartesian(devices, all_axes_args)));

INSTANTIATE_TEST_SUITE_P(Pad3D2AxesTest, Pad3D2AxesTest,
                        ::testing::ValuesIn(cartesian(devices, two_axes_args)));
}  // namespace testing
}  // namespace dali
