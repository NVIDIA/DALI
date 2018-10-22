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

#include "dali/test/dali_test_matching.h"

namespace dali {

template <typename ImgType>
class NormalizePermuteTest : public NormalizePermuteMatch<ImgType> {
 protected:
  //  To make all batch images the same size
  bool CroppingNeeded() const override                  { return true; }
  string GetInputOfTestedOperator() const override      { return "resized"; }
  const vector<int> *GetCrop() const override           { return &crop_; }
  float ResizeValue(int idx) const override             { return resize_[idx]; }
  inline void SetHeight(float h, float w)               { resize_[0] = h; resize_[1] = w; }
  inline void SetCrop(int h, int w)                     { crop_[0] = h; crop_[1] = w; }

 private:
  vector<int>crop_ = {280, 260};
  float resize_[2];
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(NormalizePermuteTest, Types);

static const char *opName = "NormalizePermute";
const bool addImageType = true;
static const char *meanValues[] = {"0.", "0., -0.2, 0.5"};
static const char *stdValues[]  = {"1.", "1., 1., 1."};


// In these tests all images are
//   (a) resized to size (H, W)
//   (b) cropped to size (height <= H, width <= W)
// After that they become the input of NormalizePermute operator

#define NORMALIZE_PERMUTE_TEST(testName, h, w, cropH, cropW, mean, std, ...)             \
          CONFORMITY_NORMALIZE_TEST(NormalizePermuteTest, testName,                      \
            this->SetHeight(h, w); this->SetCrop(cropH, cropW), mean, std, __VA_ARGS__)

#define NORMALIZE_PERMUTE_OUT(testName, outDtype)                                        \
            NORMALIZE_PERMUTE_TEST(testName, 512, 640, 256, 212, meanValues, stdValues,  \
                                        {{"output_dtype",   outDtype, DALI_INT32},       \
                                         {"height", "256", DALI_INT32},                  \
                                         {"width",  "212", DALI_INT32}})

NORMALIZE_PERMUTE_TEST(Output_DALI_DEFAULT, 512, 640, 256, 212, meanValues, stdValues,
                            {{"height", "256", DALI_INT32},
                             {"width",  "212", DALI_INT32}})

NORMALIZE_PERMUTE_OUT(Output_DALI_NO_TYPE, "-1")
NORMALIZE_PERMUTE_OUT(Output_DALI_UINT8,    "0")
NORMALIZE_PERMUTE_OUT(Output_DALI_UINT16,   "1")
NORMALIZE_PERMUTE_OUT(Output_DALI_UINT32,   "2")
NORMALIZE_PERMUTE_OUT(Output_DALI_UINT64,   "3")
NORMALIZE_PERMUTE_OUT(Output_DALI_FLOAT16,  "4")
NORMALIZE_PERMUTE_OUT(Output_DALI_FLOAT,    "5")

}  // namespace dali
