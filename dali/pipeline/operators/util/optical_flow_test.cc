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
#include <dali/test/dali_operator_test.h>
#include <dali/pipeline/operators/util/optical_flow.h>

namespace dali {
namespace testing {

namespace {
template<typename Backend>
std::unique_ptr<TensorList<Backend>> ToTensorList(const cv::Mat &image) {
  DALI_FAIL("Non-existent specialization. Feel free to write your own.");
}


template<>
std::unique_ptr<TensorList<CPUBackend>> ToTensorList(const cv::Mat &image) {
  std::unique_ptr<TensorList<CPUBackend>> tl(new TensorList<CPUBackend>);
  tl->Resize({{image.rows, image.cols, image.channels()}});
  auto img_ptr = image.data;
  auto tl_ptr = tl->template mutable_data<std::remove_pointer<decltype(img_ptr)>::type>();

  for (decltype(image.rows) i = 0; i < image.rows * image.cols * image.channels(); i++) {
    tl_ptr[i] = img_ptr[i];
  }
  return tl;
}

}  // namespace

class OpticalFlowTest : public DaliOperatorTest {
  GraphDescr GenerateOperatorGraph() const override {
    GraphDescr graph("OpticalFlow");
    return graph;
  }
};

Arguments argums = {{"device",       std::string{"cpu"}},
                    {"preset",       .5f},
                    {"enable_hints", true}};


void verify(const TensorListWrapper &input,
            const TensorListWrapper &output,
            const Arguments &) {
//  auto ptr = input.CopyTo<CPUBackend>()->data<float>();
auto ptr = input.get<CPUBackend>()->data<float>();
  cout << "OF\n" << ptr[0] << endl << ptr[1] << endl;
}


std::string kImage = "/home/mszolucha/Pictures/pokoj.png";


TEST_F(OpticalFlowTest, StubImplementationTest) {
  cv::Mat img = cv::imread(kImage);
  auto tl = ToTensorList<CPUBackend>(img);
  TensorListWrapper tlout;
  this->RunTest(tl.get(), tlout, argums, verify);
}


TEST(OpticalFlowUtilsTest, ImageToTensorList) {
  cv::Mat img = cv::imread(kImage);
  auto tl = ToTensorList<CPUBackend>(img);
  auto img_ptr = img.data;
  auto tl_ptr = tl->template data<uint8_t>();
  for (int i = 0; i < img.cols * img.rows * img.channels(); i++) {
    ASSERT_EQ(img_ptr[i], tl_ptr[i]) << "Test failed at i=" << i;
  }
}

}  // namespace testing
}  // namespace dali
