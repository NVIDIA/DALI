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
#include <vector>
#include <tuple>
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast_gpu.cuh"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast_test_utils.h"

namespace dali {
namespace kernels {
namespace brightness_contrast {
namespace test {

namespace {

template <class In, class Out>
std::enable_if_t<std::is_integral<Out>::value, Out> custom_round(float val) {
  return std::round(val);//TODO round to nearest even
}


template <class In, class Out>
std::enable_if_t<!std::is_integral<Out>::value, Out> custom_round(float val) {
  return val;
}


}  // namespace

template <class InputOutputTypes>
class BrightnessContrastCudaKernelTest : public ::testing::Test {
  using In = typename InputOutputTypes::In;
  using Out = typename InputOutputTypes::Out;

 protected:
  BrightnessContrastCudaKernelTest() {
    input_host_.resize(dataset_size());
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_host_, rng, 0., 10.);
    calc_output();
    CUDA_CALL(cudaMalloc(&input_device_, sizeof(In) * dataset_size()));
    CUDA_CALL(cudaMemcpy(input_device_, input_host_.data(), input_host_.size() * sizeof(In),
                         cudaMemcpyDefault));
    CUDA_CALL(cudaMallocManaged(&output_, dataset_size() * sizeof(Out)));
    cudaDeviceSynchronize();
  }


  In *input_device_;
  Out *output_;
  std::vector<In> input_host_;
  std::vector<Out> ref_output_;
//  TensorShape<3> shape_ = {2, 4, 3};
  TensorShape<3> shape_ = {243, 456, 3};
  float brightness_ = 4;
  float contrast_ = 3;
  static constexpr size_t ndims = 3;


  void calc_output() {
    for (auto in : input_host_) {
      ref_output_.push_back(custom_round<In, Out>(in * contrast_ + brightness_));
    }
  }


  size_t dataset_size() {
    return dali::volume(shape_);
  }
};

//using TestTypes = std::tuple<uint8_t>;
using TestTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, int32_t, float>;
INPUT_OUTPUT_TYPED_TEST_SUITE(BrightnessContrastCudaKernelTest, TestTypes);

TYPED_TEST(BrightnessContrastCudaKernelTest, cuda_kernel_test) {
  int width = this->shape_[1];
  int height = this->shape_[0];
  int channels = this->shape_[2];

  BrightnessContrastKernel(this->input_device_, this->output_, width, height, channels,
                           this->brightness_, this->contrast_);
  cudaDeviceSynchronize();

  ASSERT_EQ(this->ref_output_.size(), this->dataset_size()) << "Number of elements doesn't match";
  for (size_t i = 0; i < this->ref_output_.size(); i++) {
    EXPECT_FLOAT_EQ(this->ref_output_[i], this->output_[i]) << "Failed at idx: " << i;
  }
}


TYPED_TEST(BrightnessContrastCudaKernelTest, cuda_kernel_test_with_roi) {
  auto width = this->shape_[1];
  auto height = this->shape_[0];
  auto channels = this->shape_[2];

  Box<2, int> roi{{0,0},
                  {width-1, height-1}};

  auto mat = to_mat<3>(this->ref_output_.data(), roi,
                       this->shape_[0], this->shape_[1]);

  BrightnessContrastKernel(this->input_device_, this->output_, width, height, channels, roi.lo.x,
                           roi.lo.y, roi.extent().x, roi.extent().y, this->brightness_,
                           this->contrast_);
  cudaDeviceSynchronize();

  auto ptr = reinterpret_cast<typename TypeParam::Out *>(mat.data);
  for (int i = 0; i < mat.rows * mat.cols * mat.channels(); i++) {
    EXPECT_FLOAT_EQ(ptr[i], this->output_[i]) << "Failed at idx: " << i;
  }
}


TEST(BrightnessContrastGpuTest, test) {
  EXPECT_EQ(detail::divide_ceil(1, 4), 1ul);
  EXPECT_EQ(detail::divide_ceil(2, 4), 1ul);
  EXPECT_EQ(detail::divide_ceil(3, 4), 1ul);
  EXPECT_EQ(detail::divide_ceil(4, 4), 1ul);
  EXPECT_EQ(detail::divide_ceil(5, 4), 2ul);
  EXPECT_EQ(detail::divide_ceil(6, 4), 2ul);
  EXPECT_EQ(detail::divide_ceil(7, 4), 2ul);
  EXPECT_EQ(detail::divide_ceil(8, 4), 2ul);
  EXPECT_EQ(detail::divide_ceil(9, 4), 3ul);
}


}  // namespace test
}  // namespace brightness_contrast
}  // namespace kernels
}  // namespace dali
