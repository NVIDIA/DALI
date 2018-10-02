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

#include "dali/test/dali_test_single_op.h"

namespace dali {

namespace {

const int kBbStructSize = 4;

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
struct Roi {
  float roi[kBbStructSize];
};

const int kTestDataSize = 10;

std::array<std::pair<Roi, Roi>, kTestDataSize> wh_rois{
        {
                {{.2, .2, .4, .3}, {.4, .2, .4, .3}},
                {{.0, .0, .5, .5}, {.5, .0, .5, .5}},
                {{.3, .2, .1, .1}, {.6, .2, .1, .1}},
                {{.0, .0, .2, .3}, {.8, .0, .2, .3}},
                {{.0, .0, .1, .1}, {.9, .0, .1, .1}},
                {{.5, .5, .1, .1}, {.0, .5, .1, .1}},
                {{.0, .6, .7, .4}, {.3, .6, .7, .4}},
                {{.6, .2, .3, .3}, {.1, .2, .3, .3}},
                {{.4, .3, .5, .5}, {.1, .3, .5, .5}},
                {{.25, .25, .5, .5}, {.25, .25, .5, .5}},
        }
};

std::array<std::pair<Roi, Roi>, kTestDataSize> two_pt_rois{
        {
                {{.2, .2, .6, .5}, {.4, .2, .8, .5}},
                {{.0, .0, .5, .5}, {.5, .0, 1., .5}},
                {{.3, .2, .4, .3}, {.6, .2, .7, .3}},
                {{.0, .0, .2, .3}, {.8, .0, 1., .3}},
                {{.0, .0, .1, .1}, {.9, .0, 1., .1}},
                {{.5, .5, .6, .6}, {.4, .5, .5, .6}},
                {{.0, .6, .7, .9}, {.3, .6, 1., .9}},
                {{.6, .2, .3, .3}, {.1, .2, .4, .5}},
                {{.4, .3, .9, .8}, {.1, .3, .6, .8}},
                {{.25, .25, .75, .75}, {.25, .25, .75, .75}},
        }
};

using TestData = std::array<std::pair<Roi, Roi>, kTestDataSize>;

/**
 * Injects either input values of test std::arrays (i.e. left-hand
 * Rois) or anticipated output values, which is the reference data.
 * @param input If true, left-hand Rois will be injected.
 */
template<typename DataType>
void InjectTestData(const TestData &test_data, DataType *destination, bool input) {
  for (const auto &it : test_data) {
    auto _it = input ? it.first.roi : it.second.roi;
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
    auto batch = new TensorList<CPUBackend>();
    batch->Resize(new_batch_size_);
    auto *batch_data = batch->template mutable_data<float>();

    InjectTestData(*test_data_, batch_data, false);

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

    InjectTestData(*test_data_, batch_data, true);
  }


  const OpSpec DecodingOp(bool wh_coordinates_type) const noexcept {
    return OpSpec("BbFlip")
            .AddArg("coordinates_type", wh_coordinates_type)
            .AddInput("bb_input", "cpu")
            .AddOutput("bb_output", "cpu");
  }


 private:
  const TestData *test_data_ = nullptr;
  std::vector<std::vector<long int>> new_batch_size_;  //NOLINT
};

// XXX: `DALISingleOpTest` assumes, that input to the operator
//      is always image and is templated by ImageType.
//      Therefore this test had to be TYPED, regardless
//      of the fact, that it's unnecessary.
typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(BbFlipTest, Types);

TYPED_TEST(BbFlipTest, WidthHeightRepresentation) {
  TensorList<CPUBackend> bb_test_data;
  this->LoadBbData(bb_test_data, &wh_rois);
  this->SetExternalInputs({std::make_pair("bb_input", &bb_test_data)});
  this->RunOperator(this->DecodingOp(true), .01);
}


TYPED_TEST(BbFlipTest, TwoPointRepresentation) {
  TensorList<CPUBackend> bb_test_data;
  this->LoadBbData(bb_test_data, &two_pt_rois);
  this->SetExternalInputs({std::make_pair("bb_input", &bb_test_data)});
  this->RunOperator(this->DecodingOp(false), .01);
}

}  // namespace dali
