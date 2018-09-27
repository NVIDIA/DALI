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

const int BB_STRUCT_SIZE = 4;

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
  float roi[BB_STRUCT_SIZE];
};

/**
 * Functor for calculating Roi hash
 */
struct RoiHash {
  std::size_t operator()(Roi const &roi) const noexcept {
    std::stringstream ss;
    ss << std::to_string(roi.roi[0]) << std::to_string(roi.roi[1])
       << std::to_string(roi.roi[2]) << std::to_string(roi.roi[3]);
    return std::hash<std::string>{}(ss.str());
  }
};


bool operator==(const Roi &lh, const Roi &rh) noexcept {
  return RoiHash{}(lh) == RoiHash{}(rh);  //NOLINT
}


std::unordered_map<Roi, Roi, RoiHash> wh_rois = {
        {{.2,  .2,  .4, .3}, {.4,  .2,  .4, .3}},
        {{.0,  .0,  .5, .5}, {.5,  .0,  .5, .5}},
        {{.3,  .2,  .1, .1}, {.6,  .2,  .1, .1}},
        {{.0,  .0,  .2, .3}, {.8,  .0,  .2, .3}},
        {{.0,  .0,  .1, .1}, {.9,  .0,  .1, .1}},
        {{.5,  .5,  .1, .1}, {.0,  .5,  .1, .1}},
        {{.0,  .6,  .7, .4}, {.3,  .6,  .7, .4}},
        {{.6,  .2,  .3, .3}, {.1,  .2,  .3, .3}},
        {{.4,  .3,  .5, .5}, {.1,  .3,  .5, .5}},
        {{.25, .25, .5, .5}, {.25, .25, .5, .5}},
};

std::unordered_map<Roi, Roi, RoiHash> two_pt_rois = {
        {{.2,  .2,  .6,  .5},  {.4,  .2,  .8,  .5}},
        {{.0,  .0,  .5,  .5},  {.5,  .0,  1.,  .5}},
        {{.3,  .2,  .4,  .3},  {.6,  .2,  .7,  .3}},
        {{.0,  .0,  .2,  .3},  {.8,  .0,  1.,  .3}},
        {{.0,  .0,  .1,  .1},  {.9,  .0,  1.,  .1}},
        {{.5,  .5,  .6,  .6},  {.4,  .5,  .5,  .6}},
        {{.0,  .6,  .7,  .9},  {.3,  .6,  1.,  .9}},
        {{.6,  .2,  .3,  .3},  {.1,  .2,  .4,  .5}},
        {{.4,  .3,  .9,  .8},  {.1,  .3,  .6,  .8}},
        {{.25, .25, .75, .75}, {.25, .25, .75, .75}},
};

using RoiMap = std::unordered_map<Roi, Roi, RoiHash>;


/**
 * Flatten RoiMap, so that it is a vector of continuous floats
 * @param keys if true, the continuous float will be obtained from map keys
 *             if false - from map values
 * @return flattened vector
 */
std::vector<float> flatten(const RoiMap &roi_map, bool keys) {
  std::vector<float> ret;
  for (const auto &it : roi_map) {
    auto _it = keys ? it.first.roi : it.second.roi;
    ret.insert(ret.end(), _it, _it + BB_STRUCT_SIZE);
  }
  return ret;
}

}  // namespace

template<typename ImageType>
class BbFlipTest : public DALISingleOpTest<ImageType> {
 protected:
  std::vector<TensorList<CPUBackend> *>
  Reference(const std::vector<TensorList<CPUBackend> *> &inputs, DeviceWorkspace *ws) override {
    TensorList<CPUBackend> batch;
    batch.Resize(new_batch_size_);
    auto *out_data = batch.mutable_data<float>();

    auto rois = flatten(*test_data_, false);
    std::memcpy(out_data, rois.data(), BB_STRUCT_SIZE * sizeof(float) * test_data_->size());

    vector<TensorList<CPUBackend> *> ret(1);
    ret[0] = new TensorList<CPUBackend>();
    ret[0]->Copy(batch, nullptr);

    return ret;
  }


  template<typename Backend>
  void LoadBbData(TensorList<Backend> &batch, const RoiMap *input_data) noexcept {  // NOLINT
    test_data_ = input_data;

    auto batch_size = input_data->size();
    this->SetBatchSize(static_cast<int>(batch_size));
    batch.set_type(TypeInfo::Create<float>());
    new_batch_size_ = std::vector<std::vector<long int>>(batch_size);  //NOLINT
    for (auto &sz : new_batch_size_) {
      sz = {BB_STRUCT_SIZE};
    }
    batch.Resize(new_batch_size_);

    auto rois = flatten(*test_data_, true);

    auto ptr = batch.template mutable_data<float>();
    std::memcpy(ptr, rois.data(), BB_STRUCT_SIZE * sizeof(float) * batch_size);
  }


  const OpSpec DecodingOp(bool wh_coordinates_type) const noexcept {
    return OpSpec("BbFlip")
            .AddArg("coordinates_type", wh_coordinates_type)
            .AddInput("bb_input", "cpu")
            .AddOutput("bb_output", "cpu");
  }


 private:
  const RoiMap *test_data_ = nullptr;
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
