// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/test/dali_operator_test_utils.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

namespace testing {

namespace {

constexpr int kBbStructSize = 4;
constexpr float kEpsilon = 0.000001f;

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

/**
 * Test data for BbFlip operator. Data consists of:
 * 0 -> reference data input
 * 1 -> reference data horizontally flipped
 * 2 -> reference data vertically flipped
 */
using TestSample = Roi[3];


const TestSample rois_wh[] = {
        {{.2,  .2,  .4, .3}, {.4,  .2,  .4, .3}, {.2,  .5,  .4, .3}},
        {{.0,  .0,  .5, .5}, {.5,  .0,  .5, .5}, {.0,  .5,  .5, .5}},
        {{.3,  .2,  .1, .1}, {.6,  .2,  .1, .1}, {.3,  .7,  .1, .1}},
        {{.0,  .0,  .2, .3}, {.8,  .0,  .2, .3}, {.0,  .7,  .2, .3}},
        {{.0,  .0,  .1, .1}, {.9,  .0,  .1, .1}, {.0,  .9,  .1, .1}},
        {{.5,  .5,  .1, .1}, {.4,  .5,  .1, .1}, {.5,  .4,  .1, .1}},
        {{.0,  .6,  .7, .4}, {.3,  .6,  .7, .4}, {.0,  .0,  .7, .4}},
        {{.6,  .2,  .3, .3}, {.1,  .2,  .3, .3}, {.6,  .5,  .3, .3}},
        {{.4,  .3,  .5, .5}, {.1,  .3,  .5, .5}, {.4,  .2,  .5, .5}},
        {{.25, .25, .5, .5}, {.25, .25, .5, .5}, {.25, .25, .5, .5}},
};
const TestSample rois_ltrb[] = {
        {{.2,    .2,    .6,   .5},   {.4,    .2,    .8,   .5},    {.2,  .5,  .6,  .8}},
        {{.0,    .0,    .5,   .5},   {.5,    .0,    1.,   .5},    {.0,  .5,  .5,  1.}},
        {{.3,    .2,    .4,   .3},   {.6,    .2,    .7,   .3},    {.3,  .7,  .4,  .8}},
        {{.0,    .0,    .2,   .3},   {.8,    .0,    1.,   .3},    {.0,  .7,  .2,  1.}},
        {{.0,    .0,    .1,   .1},   {.9,    .0,    1.,   .1},    {.0,  .9,  .1,  1.}},
        {{.5,    .5,    .6,   .6},   {.4,    .5,    .5,   .6},    {.5,  .4,  .6,  .5}},
        {{.0,    .6,    .7,   .9},   {.3,    .6,    1.,   .9},    {.0,  .1,  .7,  .4}},
        {{.6,    .2,    .9,   .5},   {.1,    .2,    .4,   .5},    {.6,  .5,  .9,  .8}},
        {{.4,    .3,    .9,   .8},   {.1,    .3,    .6,   .8},    {.4,  .2,  .9,  .7}},
        {{.25,   .25,   .75,  .75},  {.25,   .25,   .75,  .75},   {.25, .25, .75, .75}},
        {{.5,    .5,    1.,   1.},   {.0,    .5,    .5,   1.},    {.5, .0, 1., .5}},
        {{.5,    .5,    1.01, 1.01}, {-0.01, .5,    .5,   1.01},  {.5, -0.01, 1.01, .5}},
        {{-0.01, -0.01, .5,   .5},   { .5,   -0.01, 1.01, .5},    {-0.01, .5, .5, 1.01}},
};


template<size_t N>
const TestSample &FindSample(const TestSample (&dataset)[N], const Roi &roi) {
  for (auto &sample : dataset) {
    if (sample[0] == roi) {
      return sample;
    }
  }
  DALI_FAIL("TestSample for provided `roi` has not been found");
}


/**
 * Create TensorList from TestSample data
 *
 * @tparam Idx index of data @see TestSample
 * @tparam Backend
 * @tparam N size of TestSample data
 * @param sample
 * @return
 */
template<size_t Idx, typename Backend, size_t N>
std::unique_ptr<TensorList<Backend>> ToTensorList(const TestSample (&sample)[N]) {
  std::unique_ptr<TensorList<Backend>> tl(new TensorList<Backend>());
  tl->Resize({N, {kBbStructSize}});
  auto ptr = tl->template mutable_data<float>();
  for (size_t n = 0; n < N; n++) {
    for (size_t i = 0; i < kBbStructSize; i++) {
      *ptr++ = sample[n][Idx][i];
    }
  }
  return tl;
}

template <typename Backend>
std::vector<Roi> FromTensorListPtr(const TensorList<Backend> *tl) {
  auto ptr = tl->template data<float>();
  std::vector<Roi> ret;
  for (size_t i = 0; i < tl->ntensor(); i++) {
    Roi roi;
    for (float &val : roi) {
      val = *ptr++;
    }
    ret.emplace_back(roi);
  }
  return ret;
}

template<typename Backend>
std::vector<Roi> FromTensorWrapper(TensorListWrapper tw) {
  if (tw.has<Backend>()) {
    auto *tl =  tw.get<Backend>();
    ASSERT_NE(nullptr, tl), std::vector<Roi>();
    ASSERT_LE(kBbStructSize, tl->size()), std::vector<Roi>();
    return FromTensorListPtr(tl);
  } else {
    return FromTensorListPtr(tw.CopyTo<Backend>().get());
  }
}


template<bool Ltrb>
void BbVerify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
  auto input_rois = FromTensorWrapper<CPUBackend>(input);
  auto output_rois = FromTensorWrapper<CPUBackend>(output);

  ASSERT_EQ(output_rois.size(), input_rois.size());
  for (size_t sample_idx = 0; sample_idx < output_rois.size(); sample_idx++) {
    auto input_roi = input_rois[sample_idx];
    auto output_roi = output_rois[sample_idx];

    DALI_ENFORCE(!(args["horizontal"].GetValue<int>() && args["vertical"].GetValue<int>()),
                 "No test data for given arguments");

    // Index of corresponding reference data in TestSample arrays
    int reference_data_idx =
            args["horizontal"].GetValue<int>() * 1 + args["vertical"].GetValue<int>() * 2;

    Roi anticipated_output_roi = Ltrb ? FindSample(rois_ltrb, input_roi)[reference_data_idx]
                                      : FindSample(rois_wh, input_roi)[reference_data_idx];

    ASSERT_EQ(anticipated_output_roi.size(), output_roi.size())
                          << "Inconsistent sizes (input vs output)";
    for (size_t i = 0; i < output_roi.size(); i++) {
      EXPECT_GT(kEpsilon, std::fabs(output_roi[i] - anticipated_output_roi[i]))
                    << "Error exceeds allowed value";
    }
  }
}

std::vector<Arguments> arguments = {
        {{"horizontal", 1}, {"vertical", 0}},
        {{"horizontal", 0}, {"vertical", 1}},
        {{"horizontal", 0}, {"vertical", 0}},
};

std::vector<Arguments> devices = {
        {{"device", std::string{"cpu"}}},
        {{"device", std::string{"gpu"}}},
};

}  // namespace

class BbFlipTest : public testing::DaliOperatorTest {
  GraphDescr GenerateOperatorGraph() const override {
    GraphDescr graph("BbFlip");
    return graph;
  }
};



TEST_P(BbFlipTest, WhRoisTest) {
  constexpr bool ltrb = false;
  auto args = GetParam();
  args.emplace("ltrb", ltrb);
  auto tlin = ToTensorList<0, CPUBackend>(rois_wh);
  TensorListWrapper tlout;
  this->RunTest(tlin.get(), tlout, args, BbVerify<ltrb>);
}


TEST_P(BbFlipTest, LtrbRoisTest) {
  constexpr bool ltrb = true;
  auto args = GetParam();
  args.emplace("ltrb", ltrb);
  auto tlin = ToTensorList<0, CPUBackend>(rois_ltrb);
  TensorListWrapper tlout;
  this->RunTest(tlin.get(), tlout, args, BbVerify<ltrb>);
}

INSTANTIATE_TEST_SUITE_P(RoisTest, BbFlipTest,
                        ::testing::ValuesIn(testing::cartesian(devices, arguments)));

}  // namespace testing
}  // namespace dali
