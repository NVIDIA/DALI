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

#include <cmath>

#include "dali/test/dali_operator_test.h"

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
using Roi = std::vector<float>;

constexpr int kTestDataSize = 10;

/**
 * Test data for BbFlip operator. Data consists of:
 * 0 -> reference data input
 * 1 -> reference data horizontally flipped
 * 2 -> reference data vertically flipped
 */
using TestData = std::array<std::array<Roi, 3>, kTestDataSize>;

const TestData wh_rois = {{
        {{{.2,  .2,  .4, .3}, {.4,  .2,  .4, .3}, {.2,  .5,  .4, .3}}},
        {{{.0,  .0,  .5, .5}, {.5,  .0,  .5, .5}, {.0,  .5,  .5, .5}}},
        {{{.3,  .2,  .1, .1}, {.6,  .2,  .1, .1}, {.3,  .7,  .1, .1}}},
        {{{.0,  .0,  .2, .3}, {.8,  .0,  .2, .3}, {.0,  .7,  .2, .3}}},
        {{{.0,  .0,  .1, .1}, {.9,  .0,  .1, .1}, {.0,  .9,  .1, .1}}},
        {{{.5,  .5,  .1, .1}, {.4,  .5,  .1, .1}, {.5,  .4,  .1, .1}}},
        {{{.0,  .6,  .7, .4}, {.3,  .6,  .7, .4}, {.0,  .0,  .7, .4}}},
        {{{.6,  .2,  .3, .3}, {.1,  .2,  .3, .3}, {.6,  .5,  .3, .3}}},
        {{{.4,  .3,  .5, .5}, {.1,  .3,  .5, .5}, {.4,  .2,  .5, .5}}},
        {{{.25, .25, .5, .5}, {.25, .25, .5, .5}, {.25, .25, .5, .5}}},
}};

// const TestData two_pt_rois = {{
//         {{{.2,  .2,  .6,  .5},  {.4,  .2,  .8,  .5},  {.2,  .5,  .6,  .8}}},
//         {{{.0,  .0,  .5,  .5},  {.5,  .0,  1.,  .5},  {.0,  .5,  .5,  1.}}},
//         {{{.3,  .2,  .4,  .3},  {.6,  .2,  .7,  .3},  {.3,  .7,  .4,  .8}}},
//         {{{.0,  .0,  .2,  .3},  {.8,  .0,  1.,  .3},  {.0,  .7,  .2,  1.}}},
//         {{{.0,  .0,  .1,  .1},  {.9,  .0,  1.,  .1},  {.0,  .9,  .1,  1.}}},
//         {{{.5,  .5,  .6,  .6},  {.4,  .5,  .5,  .6},  {.5,  .4,  .6,  .5}}},
//         {{{.0,  .6,  .7,  .9},  {.3,  .6,  1.,  .9},  {.0,  .1,  .7,  .4}}},
//         {{{.6,  .2,  .9,  .5},  {.1,  .2,  .4,  .5},  {.6,  .5,  .9,  .8}}},
//         {{{.4,  .3,  .9,  .8},  {.1,  .3,  .6,  .8},  {.4,  .2,  .9,  .7}}},
//         {{{.25, .25, .75, .75}, {.25, .25, .75, .75}, {.25, .25, .75, .75}}},
// }};

}  // namespace

class BbFlipTest : public testing::DaliOperatorTest<Roi, Roi> {
  std::vector<std::pair<Roi, Shape>> SetInputs() const override {
    std::vector<std::pair<Roi, Shape>> inputs;
    const Shape shape = {4};
    for (auto roi_set : wh_rois) {
      inputs.emplace_back(std::make_pair(roi_set[0], shape));
    }
    return inputs;
  }


  std::string SetOperator() const override {
    return "BbFlip";
  }


  bool Verify(Roi output, Roi anticipated_output) const override {
    DALI_ENFORCE(output.size() == anticipated_output.size(), "Sizes don't match");
    for (size_t i = 0; i < output.size(); i++) {
      if (std::fabs(output[i] - anticipated_output[i]) > epsilon_) {
        cout << output[i] << " - " << anticipated_output[i] << endl;
        return false;
      }
    }
    return true;
  }


 private:
  float epsilon_ = 0.001f;
};

TEST_F(BbFlipTest, HorizontalTest) {
  std::vector<Roi> anticipated_outputs;
  for (auto roi_set : wh_rois) {
    anticipated_outputs.emplace_back(roi_set[1]);
  }
  this->RunTest<CPUBackend>({{"horizontal", 1}, {"vertical", 0}}, anticipated_outputs);
}

TEST_F(BbFlipTest, VerticalTest) {
  std::vector<Roi> anticipated_outputs;
  for (auto roi_set : wh_rois) {
    anticipated_outputs.emplace_back(roi_set[2]);
  }
  this->RunTest<CPUBackend>({{"horizontal", 0}, {"vertical", 1}}, anticipated_outputs);
}

}  // namespace dali
