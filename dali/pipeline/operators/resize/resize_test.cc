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

// Values of eps for different tests and different methods for calculation of average deviation
// When flag t_checkElements is NOT set, the average deviation is calculated COMBINED for ALL
//      images of the batch - the eps from first column will be used.
// When flag t_checkElements is set, the average deviation is calculated SEPARATELY for EACH image
//      of the batch - the eps from second column will be used.

static double testEps[] = {     //     TEST:
                    0.2, 0.3,   // ResizeShorter_LINEAR
                    0.2, 0.3,   // ResizeShorter_A_LINEAR
                    0.2, 0.3,   // ResizeLonger_LINEAR
                    0.2, 0.3,   // ResizeLonger_A_LINEAR
                    0.2, 0.3,   // ResizeXY_LINEAR
                    0.2, 0.3,   // ResizeXY_A_LINEAR
                    1.1, 2.2,   // ResizeShorter_NN
                    1.1, 2.2,   // ResizeShorter_A_NN
                    1.1, 2.2,   // ResizeLonger_NN
                    1.1, 2.2,   // ResizeLonger_A_NN
                    1.1, 2.2,   // ResizeXY_NN
                    1.1, 2.2,   // ResizeXY_A_NN
                    0.3, 0.6,   // ResizeShorter_CUBIC
                    0.3, 0.6,   // ResizeShorter_A_CUBIC
                    0.3, 0.6,   // ResizeLonger_CUBIC
                    0.3, 0.6,   // ResizeLonger_A_CUBIC
                    0.3, 0.6,   // ResizeXY_CUBIC
                    0.3, 0.6,   // ResizeXY_A_CUBIC
};

template <typename ImgType>
class ResizeTest : public GenericResizeTest<ImgType>  {
 protected:
  inline uint32_t getResizeOptions() const override   { return 0/*t_externSizes*/; }
  inline int getInterpType() const  override          { return m_interpType; }
  inline void setInterpType(int interpType)           { m_interpType = interpType; }
  double *testEpsValues() const override              { return testEps; }
  inline void SetTestCheckType(uint32_t checkType)    { m_testCheckType = checkType; }
  uint32_t GetTestCheckType() const  override         { return m_testCheckType; }
 private:
  int m_interpType = cv::INTER_LINEAR;
  int m_testCheckType = t_checkDefault;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(ResizeTest, Types);

typedef enum {
  t_ResizeShorter_LINEAR,
  t_ResizeShorter_A_LINEAR,
  t_ResizeLonger_LINEAR,
  t_ResizeLonger_A_LINEAR,
  t_ResizeXY_LINEAR,
  t_ResizeXY_A_LINEAR,
  t_ResizeShorter_NN,
  t_ResizeShorter_A_NN,
  t_ResizeLonger_NN,
  t_ResizeLonger_A_NN,
  t_ResizeXY_NN,
  t_ResizeXY_A_NN,
  t_ResizeShorter_CUBIC,
  t_ResizeShorter_A_CUBIC,
  t_ResizeLonger_CUBIC,
  t_ResizeLonger_A_CUBIC,
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
#define TESTS_WITH_CHECK(testName, interp, testArgs, checkType)                 \
        TYPED_TEST(ResizeTest, testName##_##checkType##_##interp##_GPU) {       \
          this->setInterpType(interpType[interp].cvInterp);                     \
          this->SetTestCheckType(t_check##checkType);                           \
          this->TstBody(this->DefaultSchema("Resize", "gpu")                    \
                          .AddArg("interp_type", interpType[interp].daliInterp) \
                          testArgs, this->getEps(t_##testName##_##interp)); }   \
        TYPED_TEST(ResizeTest, testName##_##checkType##_##interp##_CPU) {       \
          this->setInterpType(interpType[interp].cvInterp);                     \
          this->SetTestCheckType(t_check##checkType);                           \
          this->TstBody(this->DefaultSchema("Resize", "cpu")                    \
                          .AddArg("interp_type", interpType[interp].daliInterp) \
                          testArgs, this->getEps(t_##testName##_##interp)); }

// Macro which allows to create pair of identical tests for t_checkDefault/t_checkElements types
// of checking of average deviation of color values
#define TYPED_TESTS(testName, interp, testArgs)                         \
            TESTS_WITH_CHECK(testName, interp, testArgs, Default)       \
            TESTS_WITH_CHECK(testName, interp, testArgs, Elements)

TYPED_TESTS(ResizeShorter,   LINEAR, .AddArg("resize_shorter", 480.f))
TYPED_TESTS(ResizeShorter_A, LINEAR, .AddArg("resize_shorter", 224.f))
TYPED_TESTS(ResizeLonger,    LINEAR, .AddArg("resize_longer",  640.f))
TYPED_TESTS(ResizeLonger_A,  LINEAR, .AddArg("resize_longer",  960.f))
TYPED_TESTS(ResizeXY,        LINEAR, .AddArg("resize_x", 224.f)         \
                                     .AddArg("resize_y", 224.f))
TYPED_TESTS(ResizeXY_A,      LINEAR, .AddArg("resize_x", 240.f)         \
                                     .AddArg("resize_y", 480.f))
TYPED_TESTS(ResizeShorter,       NN, .AddArg("resize_shorter", 480.f))
TYPED_TESTS(ResizeShorter_A,     NN, .AddArg("resize_shorter", 224.f))
TYPED_TESTS(ResizeLonger,        NN, .AddArg("resize_longer",  640.f))
TYPED_TESTS(ResizeLonger_A,      NN, .AddArg("resize_longer",  960.f))
TYPED_TESTS(ResizeXY,            NN, .AddArg("resize_x", 224.f)         \
                                     .AddArg("resize_y", 224.f))
TYPED_TESTS(ResizeXY_A,          NN, .AddArg("resize_x", 240.f)         \
                                     .AddArg("resize_y", 480.f))
TYPED_TESTS(ResizeShorter,    CUBIC, .AddArg("resize_shorter", 480.f))
TYPED_TESTS(ResizeShorter_A,  CUBIC, .AddArg("resize_shorter", 224.f))
TYPED_TESTS(ResizeLonger,     CUBIC, .AddArg("resize_longer",  640.f))
TYPED_TESTS(ResizeLonger_A,   CUBIC, .AddArg("resize_longer",  960.f))
TYPED_TESTS(ResizeXY,         CUBIC, .AddArg("resize_x", 224.f)         \
                                     .AddArg("resize_y", 224.f))
TYPED_TESTS(ResizeXY_A,       CUBIC, .AddArg("resize_x", 240.f)         \
                                     .AddArg("resize_y", 480.f))

}  // namespace dali
