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

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <dali/pipeline/operators/util/optical_flow.h>
#include <dali/test/dali_operator_test.h>
#include <dali/test/dali_operator_test_utils.h>
#include <vector>
#include <string>
#include <memory>

namespace dali {
namespace testing {

namespace {
std::unique_ptr<TensorList<CPUBackend>> ToTensorList(const std::vector<cv::Mat> &images) {
  std::unique_ptr<TensorList<CPUBackend>> tl(new TensorList<CPUBackend>);
  auto img = images[0];
  tl->Resize({images.size(), {img.rows, img.cols, img.channels()}});
  auto tl_ptr = tl->template mutable_data<std::remove_pointer<decltype(img.data)>::type>();
  for (const auto &image : images) {
    auto img_ptr = image.data;
    for (decltype(image.rows) i = 0; i < image.rows * image.cols * image.channels(); i++) {
      *tl_ptr++ = img_ptr[i];
    }
  }
  return tl;
}


std::vector<Arguments> arguments = {
        {{"preset", .5f}, {"enable_hints", false}}
};

Arguments device_gpu = {{"device", std::string{"gpu"}}};


}  // namespace

// TODO(mszolucha): update for Dali_extra usage
std::string kImage = "/data/dali/test/test_images/410.jpg";  // NOLINT

TEST(OpticalFlowUtilsTest, ImageToTensorListCpu) {
  cv::Mat img = cv::imread(kImage);
  auto tl = ToTensorList({img});
  auto img_ptr = img.data;
  auto tl_ptr = tl->template data<uint8_t>();
  ASSERT_EQ(img.rows * img.cols * img.channels(), tl->size()) << "Sizes don't match";
  for (int i = 0; i < img.cols * img.rows * img.channels(); i++) {
    ASSERT_EQ(img_ptr[i], tl_ptr[i]) << "Test failed at i=" << i;
  }
}


class OpticalFlowTest : public DaliOperatorTest {
  GraphDescr GenerateOperatorGraph() const override {
    GraphDescr graph("OpticalFlow");
    return graph;
  }
};


void verify(const TensorListWrapper &input,
            const TensorListWrapper &output,
            const Arguments &args) {
  auto input_tl = input.CopyTo<CPUBackend>();
  auto output_tl = output.CopyTo<CPUBackend>();
  const uint8_t *input_data;
  const float *output_data;
  utils::pointer_to_data(*input_tl, input_data);
  utils::pointer_to_data(*output_tl, output_data);
  EXPECT_FLOAT_EQ(666.f, output_data[0]);
  EXPECT_FLOAT_EQ(333.f, output_data[1]);
}


TEST_P(OpticalFlowTest, StubImplementationTest) {
  cv::Mat img = cv::imread(kImage);
  auto tl = ToTensorList({img, img});
  TensorListWrapper tlout;
  auto args = GetParam();
  this->RunTest(tl.get(), tlout, args, verify);
}


INSTANTIATE_TEST_CASE_P(OpticalFlowStubImplementationsTest, OpticalFlowTest,
                        ::testing::ValuesIn(testing::cartesian({device_gpu}, arguments)));

}  // namespace testing
}  // namespace dali
