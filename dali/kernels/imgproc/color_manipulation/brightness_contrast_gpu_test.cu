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
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast_gpu.cuh"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast_test_utils.h"

namespace dali {
namespace kernels {
namespace test {

using Type = float;

namespace {




template <typename OutType>
std::enable_if_t<std::is_integral<OutType>::value, OutType> custom_round(float val) {
  auto integral = static_cast<OutType>(val);
  return integral % 2 == 0 ? integral : integral + 1;
}

template <typename OutType>
std::enable_if_t<!std::is_integral<OutType>::value, OutType> custom_round(float val) {
  return val;
}


}  // namespace


class BrightnessContrastGpuKernelTest : public ::testing::Test {
 protected:
  BrightnessContrastGpuKernelTest() {
    input_host_.resize(dataset_size());
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_host_, rng, 0., 10.);
    calc_output();
    CUDA_CALL(cudaMalloc(&input_device_, sizeof(Type) * dataset_size()));
    CUDA_CALL(cudaMemcpy(input_device_, input_host_.data(), input_host_.size() * sizeof(Type),
                         cudaMemcpyDefault));
    CUDA_CALL(cudaMallocManaged(&output_, dataset_size() * sizeof(Type)));
    cudaDeviceSynchronize();
  }


  Type *input_device_;
  Type* output_;
  std::vector<Type> input_host_;
  std::vector<Type> ref_output_;
  TensorShape<3> shape_ = {2, 4,3};
//  TensorShape<3> shape_ = {243,456,13};
  float brightness_ = 4;
  float contrast_ = 3;
  static constexpr size_t ndims = 3;


  void calc_output() {
    for (auto in : input_host_) {
      ref_output_.push_back(custom_round<Type>(in * contrast_ + brightness_));
    }
  }


  size_t dataset_size() {
    return dali::volume(shape_);
  }
};

TEST_F(BrightnessContrastGpuKernelTest, cuda_kernel_test) {
  int width = shape_[1];
  int height = shape_[0];
  int channels = shape_[2];

::dali::kernels::brightness_contrast::BrightnessContrastKernel(input_device_, output_, width,height,3, brightness_,contrast_);
  cudaDeviceSynchronize();

  ASSERT_EQ(ref_output_.size(), dataset_size()) << "Number of elements doesn't match";
  for (int i = 0; i < ref_output_.size(); i++) {
    EXPECT_FLOAT_EQ(ref_output_[i], output_[i]) << "Failed at idx: " << i;
  }
}

TEST_F(BrightnessContrastGpuKernelTest, cuda_kernel_test_with_roi){
//  auto width=shape_[1];
//  auto height=shape_[0];
//  auto channels=shape_[2];
//
//  Box<2,int> roi{{0,0},{width, height}};
//
//  auto mat = brightness_contrast::to_mat<3>(this->ref_output_.data(), roi,
//                                            this->shape_[0], this->shape_[1]);
//
//  cout<<mat<<endl;


//  ::dali::kernels::brightness_contrast::BrightnessContrastKernel(input_device_, output_, 0,0,3,brightness_,contrast_,width,height);
//  cudaDeviceSynchronize();

//  auto ptr = reinterpret_cast<Type *>(mat.data);
//  for (int i = 0; i < mat.rows*mat.cols*mat.channels(); i++) {
//    EXPECT_FLOAT_EQ(ptr[i], output_[i]) << "Failed at idx: " << i;
//  }
}





}  // namespace test
}  // namespace kernels
}  // namespace dali
