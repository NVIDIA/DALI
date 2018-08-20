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

#include "dali/test/dali_test_resize.h"

namespace dali {

// Values of eps for different tests
//      first column:   average deviation is calculated combined for all images
//      second column:  average deviation is calculated separately for each image
static double testEps[] = {
                    0.7, 1.8,
                    2.1, 4.2,
                    2.8, 5.6,
                    1.8, 3.4,
                    1.5, 2.9,
                    3.4, 5.3,
                    3.4, 5.3,
                    2.4, 4.1,
                    0.8, 2.1,
                    2.4, 4.9,
                    3.2, 6.4,
                    2.0, 4.0,
};

template <typename ImgType>
class ResizeTest : public GenericResizeTest<ImgType>  {
 protected:
  uint32_t getResizeOptions() const override          { return 0/*t_externSizes*/; }
  int getInterpType() const  override                 { return m_interpType; }
  inline void setInterpType(int interpType)           { m_interpType = interpType; }
  double *testEpsValues() const override              { return testEps; }
 private:
  int m_interpType = cv::INTER_LINEAR;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(ResizeTest, Types);

typedef enum {
  t_ResizeShorter_LINEAR,
  t_ResizeShorter_A_LINEAR,
  t_ResizeXY_LINEAR,
  t_ResizeXY_A_LINEAR,
  t_ResizeShorter_NN,
  t_ResizeShorter_A_NN,
  t_ResizeXY_NN,
  t_ResizeXY_A_NN,
  t_ResizeShorter_CUBIC,
  t_ResizeShorter_A_CUBIC,
  t_ResizeXY_CUBIC,
  t_ResizeXY_A_CUBIC
} test_ID;

typedef struct {
  cv::InterpolationFlags cvInterp;
  DALIInterpType daliInterp;
} InterpolationType;

InterpolationType interpType[] = {
  {cv::INTER_LINEAR,  DALI_INTERP_LINEAR},
  {cv::INTER_NEAREST, DALI_INTERP_NN},
  {cv::INTER_CUBIC,   DALI_INTERP_CUBIC}
};

enum {
  LINEAR,
  NN,
  CUBIC,
};

// Macro which allows to create pair of identical CPU/GPU tests
#define TYPED_TESTS(testName, idx, testArgs)                                  \
        TYPED_TEST(ResizeTest, testName##_##idx##_GPU) {                      \
          this->setInterpType(interpType[idx].cvInterp);                      \
          this->TstBody(this->DefaultSchema("Resize", "gpu")                  \
                          .AddArg("interp_type", interpType[idx].daliInterp)  \
                          testArgs, this->getEps(t_##testName##_##idx)); }  \
        TYPED_TEST(ResizeTest, testName##_##idx##_CPU) {                      \
          this->setInterpType(interpType[idx].cvInterp);                      \
          this->TstBody(this->DefaultSchema("Resize", "cpu")                  \
                          .AddArg("interp_type", interpType[idx].daliInterp)  \
                          testArgs, 1e-5); }

TYPED_TESTS(ResizeShorter,   LINEAR, .AddArg("resize_shorter", 480.f))

TYPED_TESTS(ResizeShorter_A, LINEAR, .AddArg("resize_shorter", 224.f))

TYPED_TESTS(ResizeXY,        LINEAR, .AddArg("resize_x", 224.f)         \
                                     .AddArg("resize_y", 224.f))
TYPED_TESTS(ResizeXY_A,      LINEAR, .AddArg("resize_x", 240.f)         \
                                     .AddArg("resize_y", 480.f))
TYPED_TESTS(ResizeShorter,       NN, .AddArg("resize_shorter", 480.f))

TYPED_TESTS(ResizeShorter_A,     NN, .AddArg("resize_shorter", 224.f))
TYPED_TESTS(ResizeXY,            NN, .AddArg("resize_x", 224.f)         \
                                     .AddArg("resize_y", 224.f))
TYPED_TESTS(ResizeXY_A,          NN, .AddArg("resize_x", 240.f)         \
                                     .AddArg("resize_y", 480.f))
TYPED_TESTS(ResizeShorter,    CUBIC, .AddArg("resize_shorter", 480.f))

TYPED_TESTS(ResizeShorter_A,  CUBIC, .AddArg("resize_shorter", 224.f))

TYPED_TESTS(ResizeXY,         CUBIC, .AddArg("resize_x", 224.f)         \
                                     .AddArg("resize_y", 224.f))
TYPED_TESTS(ResizeXY_A,       CUBIC, .AddArg("resize_x", 240.f)         \
                                     .AddArg("resize_y", 480.f))

}  // namespace dali
