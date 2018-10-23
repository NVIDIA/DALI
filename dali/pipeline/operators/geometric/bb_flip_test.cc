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

#include <tuple>
#include "dali/test/dali_test_single_op.h"

namespace dali {

namespace {

constexpr int kBbStructSize = 4;

/**
 * Roi can be represented in two ways:
 * 1. Upper-left corner, width, height
 *    (x1, y1,  w,  h)
 * 2. Upper-left and Lower-right corners
 *    (x1, y1, x2, y2)
 *
 * Both of them have coordinates in image coordinate system
 * (i.e. 0.0-1.0)
 */
using Roi = std::array<float, kBbStructSize>;

constexpr int kTestDataSize = 10;

/**
 * Test data for BbFlip operator. Data consists of:
 * 0 -> reference data input
 * 1 -> reference data horizontally flipped
 * 2 -> reference data vertically flipped
 */
using TestData = std::array<std::tuple<Roi, Roi, Roi>, kTestDataSize>;

constexpr TestData wh_rois{
        {
                std::tuple<Roi, Roi, Roi>{{.2, .2, .4, .3},
                                          {.4, .2, .4, .3},
                                          {.2, .5, .4, .3}},
                std::tuple<Roi, Roi, Roi>{{.0, .0, .5, .5},
                                          {.5, .0, .5, .5},
                                          {.0, .5, .5, .5}},
                std::tuple<Roi, Roi, Roi>{{.3, .2, .1, .1},
                                          {.6, .2, .1, .1},
                                          {.3, .7, .1, .1}},
                std::tuple<Roi, Roi, Roi>{{.0, .0, .2, .3},
                                          {.8, .0, .2, .3},
                                          {.0, .7, .2, .3}},
                std::tuple<Roi, Roi, Roi>{{.0, .0, .1, .1},
                                          {.9, .0, .1, .1},
                                          {.0, .9, .1, .1}},
                std::tuple<Roi, Roi, Roi>{{.5, .5, .1, .1},
                                          {.4, .5, .1, .1},
                                          {.5, .4, .1, .1}},
                std::tuple<Roi, Roi, Roi>{{.0, .6, .7, .4},
                                          {.3, .6, .7, .4},
                                          {.0, .0, .7, .4}},
                std::tuple<Roi, Roi, Roi>{{.6, .2, .3, .3},
                                          {.1, .2, .3, .3},
                                          {.6, .5, .3, .3}},
                std::tuple<Roi, Roi, Roi>{{.4, .3, .5, .5},
                                          {.1, .3, .5, .5},
                                          {.4, .2, .5, .5}},
                std::tuple<Roi, Roi, Roi>{{.25, .25, .5, .5},
                                          {.25, .25, .5, .5},
                                          {.25, .25, .5, .5}},
        }
};

constexpr TestData two_pt_rois{
        {
                std::tuple<Roi, Roi, Roi>{{.2, .2, .6, .5},
                                          {.4, .2, .8, .5},
                                          {.2, .5, .6, .8}},
                std::tuple<Roi, Roi, Roi>{{.0, .0, .5, .5},
                                          {.5, .0, 1., .5},
                                          {.0, .5, .5, 1.}},
                std::tuple<Roi, Roi, Roi>{{.3, .2, .4, .3},
                                          {.6, .2, .7, .3},
                                          {.3, .7, .4, .8}},
                std::tuple<Roi, Roi, Roi>{{.0, .0, .2, .3},
                                          {.8, .0, 1., .3},
                                          {.0, .7, .2, 1.}},
                std::tuple<Roi, Roi, Roi>{{.0, .0, .1, .1},
                                          {.9, .0, 1., .1},
                                          {.0, .9, .1, 1.}},
                std::tuple<Roi, Roi, Roi>{{.5, .5, .6, .6},
                                          {.4, .5, .5, .6},
                                          {.5, .4, .6, .5}},
                std::tuple<Roi, Roi, Roi>{{.0, .6, .7, .9},
                                          {.3, .6, 1., .9},
                                          {.0, .1, .7, .4}},
                std::tuple<Roi, Roi, Roi>{{.6, .2, .9, .5},
                                          {.1, .2, .4, .5},
                                          {.6, .5, .9, .8}},
                std::tuple<Roi, Roi, Roi>{{.4, .3, .9, .8},
                                          {.1, .3, .6, .8},
                                          {.4, .2, .9, .7}},
                std::tuple<Roi, Roi, Roi>{{.25, .25, .75, .75},
                                          {.25, .25, .75, .75},
                                          {.25, .25, .75, .75}},
        }
};


/**
 * Injects either input values of test std::arrays (i.e. left-hand
 * Rois) or anticipated output values, which is the reference data.
 * @tparam DataIdx Index of tuple contained in TestData. Specifies
 *                 which values to inject
 */
template<typename DataType, int DataIdx>
void InjectTestData(const TestData &test_data, DataType *destination) {
  for (const auto &it : test_data) {
    auto _it = std::get<DataIdx>(it).data();
    std::memcpy(destination, _it, kBbStructSize * sizeof(DataType));
    destination += kBbStructSize;
  }
}

}  // namespace

template<typename ImageType>
class BbFlipTest : public DALISingleOpTest<ImageType> {
 protected:
  std::vector<TensorList<CPUBackend> *>
  Reference(const std::vector<TensorList<CPUBackend> *> &inputs, DeviceWorkspace *ws) override {
    using DataType = float;
    auto batch = new TensorList<CPUBackend>();
    batch->Resize(new_batch_size_);
    auto *batch_data = batch->template mutable_data<DataType>();

    DALI_ENFORCE(!(flip_type_horizontal_ && flip_type_vertical_), "No test data for combined case");

    if (flip_type_vertical_) {
      InjectTestData<DataType, 2>(*test_data_, batch_data);
    } else if (flip_type_horizontal_) {
      InjectTestData<DataType, 1>(*test_data_, batch_data);
    } else {
      InjectTestData<DataType, 0>(*test_data_, batch_data);
    }

    vector<TensorList<CPUBackend> *> ret(1);
    ret[0] = batch;

    return ret;
  }


  template<typename Backend>
  void LoadBbData(TensorList<Backend> &batch, const TestData *input_data) noexcept {  // NOLINT
    test_data_ = input_data;

    auto batch_size = input_data->size();
    this->SetBatchSize(static_cast<int>(batch_size));
    batch.set_type(TypeInfo::Create<float>());
    new_batch_size_ = std::vector<std::vector<long int>>(batch_size);  //NOLINT
    for (auto &sz : new_batch_size_) {
      sz = {kBbStructSize};
    }
    batch.Resize(new_batch_size_);

    auto batch_data = batch.template mutable_data<float>();

    InjectTestData<float, 0>(*test_data_, batch_data);
  }


  const OpSpec GetOperatorSpec(bool ltrb_coordinates_type,
                               bool vertical, bool horizontal) noexcept {
    flip_type_vertical_ = vertical;
    flip_type_horizontal_ = horizontal;
    return OpSpec("BbFlip")
            .AddArg("ltrb", ltrb_coordinates_type)
            .AddArg("vertical", vertical ? 1 : 0)
            .AddArg("horizontal", horizontal ? 1 : 0)
            .AddInput("bb_input", "cpu")
            .AddOutput("bb_output", "cpu");
  }


 private:
  const TestData *test_data_ = nullptr;
  std::vector<std::vector<long int>> new_batch_size_;  //NOLINT
  bool flip_type_vertical_, flip_type_horizontal_;
};

// XXX: `DALISingleOpTest` assumes, that input to the operator
//      is always image and is templated by ImageType.
//      Therefore this test had to be TYPED, regardless
//      of the fact, that it's unnecessary.
typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(BbFlipTest, Types);

TYPED_TEST(BbFlipTest, VerticalWHTest) {
  TensorList<CPUBackend> bb_test_data;
  this->LoadBbData(bb_test_data, &wh_rois);
  this->SetExternalInputs({std::make_pair("bb_input", &bb_test_data)});
  this->RunOperator(this->GetOperatorSpec(false, true, false), .001);
}


TYPED_TEST(BbFlipTest, Vertical2PTest) {
  TensorList<CPUBackend> bb_test_data;
  this->LoadBbData(bb_test_data, &two_pt_rois);
  this->SetExternalInputs({std::make_pair("bb_input", &bb_test_data)});
  this->RunOperator(this->GetOperatorSpec(true, true, false), .001);
}


TYPED_TEST(BbFlipTest, HorizontalWHTest) {
  TensorList<CPUBackend> bb_test_data;
  this->LoadBbData(bb_test_data, &wh_rois);
  this->SetExternalInputs({std::make_pair("bb_input", &bb_test_data)});
  this->RunOperator(this->GetOperatorSpec(false, false, true), .001);
}


TYPED_TEST(BbFlipTest, Horizontal2PTest) {
  TensorList<CPUBackend> bb_test_data;
  this->LoadBbData(bb_test_data, &two_pt_rois);
  this->SetExternalInputs({std::make_pair("bb_input", &bb_test_data)});
  this->RunOperator(this->GetOperatorSpec(true, false, true), .001);
}


TYPED_TEST(BbFlipTest, NoFlipTest) {
  TensorList<CPUBackend> bb_test_data;
  this->LoadBbData(bb_test_data, &wh_rois);
  this->SetExternalInputs({std::make_pair("bb_input", &bb_test_data)});
  this->RunOperator(this->GetOperatorSpec(false, false, false), .001);
}

}  // namespace dali
