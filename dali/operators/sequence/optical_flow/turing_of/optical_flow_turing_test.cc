// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include <utility>

#include "dali/kernels/common/copy.h"
#include "dali/operators/sequence/optical_flow/turing_of/optical_flow_turing.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/dev_buffer.h"

namespace dali {
namespace optical_flow {
namespace testing {

namespace {
constexpr float kFlowVectorEpsilon = 1.f / 32;

}  // namespace

namespace kernel {

class OpticalFlowTuringKernelTest : public ::testing::Test {
 protected:
  void SetUp() final {
  }


  void TearDown() final {
  }


  template<bool Grayscale, typename ColorConversion>
  void ColorConversionTest(ColorConversion cc, std::vector<uint8_t> input_data,
                           std::vector<uint8_t> reference_data) {
    DeviceBuffer<uint8_t> input, tested;
    std::vector<uint8_t> tested_host;
    input.from_host(input_data);
    tested.resize(reference_data.size());
    tested_host.resize(reference_data.size());

    auto w = Grayscale ? width_gray_ : width_;
    auto h = Grayscale ? height_gray_ : height_;
    auto p = Grayscale ? pitch_gray_ : pitch_;

    cudaMemset(tested.data(), 0x5C, tested.size()*sizeof(tested[0]));
    cc(input.data(), tested.data(), p, w, h, 0);
    CUDA_CALL(cudaDeviceSynchronize());
    copyD2H(tested_host.data(), tested.data(), tested_host.size());

    for (size_t i = 0; i < reference_data.size(); i++) {
      EXPECT_EQ(reference_data[i], tested_host[i]) << "Failed on index: " << i;
    }
  }


  const std::vector<uint8_t> reference_data_ = {
          73, 5, 47, 255, 71, 30, 1, 255, 0x5C, 0x5C,
          80, 41, 60, 255, 60, 85, 41, 255, 0x5C, 0x5C,
          55, 66, 4, 255, 94, 59, 47, 255, 0x5C, 0x5C,
          64, 83, 96, 255, 61, 30, 95, 255, 0x5C, 0x5C,
          88, 95, 63, 255, 96, 78, 16, 255, 0x5C, 0x5C,
          81, 89, 81, 255, 2, 18, 35, 255, 0x5C, 0x5C,
  };
  const std::vector<uint8_t> rgb_data_ = {
          73, 5, 47, 71, 30, 1,
          80, 41, 60, 60, 85, 41,
          55, 66, 4, 94, 59, 47,
          64, 83, 96, 61, 30, 95,
          88, 95, 63, 96, 78, 16,
          81, 89, 81, 2, 18, 35,
  };
  const std::vector<uint8_t> bgr_data_ = {
          47, 5, 73, 1, 30, 71,
          60, 41, 80, 41, 85, 60,
          4, 66, 55, 47, 59, 94,
          96, 83, 64, 95, 30, 61,
          63, 95, 88, 16, 78, 96,
          81, 89, 81, 35, 18, 2,
  };
  const std::vector<uint8_t> gray_data_ = {
          73, 5, 47, 255, 71, 30, 1, 255,
          80, 41, 60, 255, 60, 85, 41, 255,
          55, 66, 4, 255, 94, 59, 47, 255,
          64, 83, 96, 255, 61, 30, 95, 255,
          88, 95, 63, 255, 96, 78, 16, 255,
          81, 89, 81, 255, 2, 18, 35, 255,
  };

 private:
  size_t width_ = 2, pitch_ = 10, height_ = 6;
  size_t width_gray_ = 8, pitch_gray_ = 10, height_gray_ = 6;
};

TEST_F(OpticalFlowTuringKernelTest, RgbToRgbaTest) {
  ColorConversionTest<false>(optical_flow::kernel::RgbToRgba, this->rgb_data_,
                             this->reference_data_);
}


TEST_F(OpticalFlowTuringKernelTest, BgrToRgbaTest) {
  ColorConversionTest<false>(optical_flow::kernel::BgrToRgba, this->bgr_data_,
                             this->reference_data_);
}


TEST_F(OpticalFlowTuringKernelTest, GrayTest) {
  ColorConversionTest<true>(optical_flow::kernel::Gray, this->gray_data_, this->reference_data_);
}


TEST_F(OpticalFlowTuringKernelTest, FlowVectorTest) {
  const std::vector<int16_t> reference_data = {
          73, 5, 47, 255, 71, 30, 1, 255, 0x5C5C, 0x5C5C,
          80, 41, 60, 255, 60, 85, 41, 255, 0x5C5C, 0x5C5C,
          55, 66, 4, 255, 94, 59, 47, 255, 0x5C5C, 0x5C5C,
          64, 83, 96, 255, 61, 30, 95, 255, 0x5C5C, 0x5C5C,
          88, 95, 63, 255, 96, 78, 16, 255, 0x5C5C, 0x5C5C,
          81, 89, 81, 255, 2, 18, 35, 255, 0x5C5C, 0x5C5C,
  };
  const std::vector<float> test_data = {
          2.28125, 0.15625, 1.46875, 7.96875, 2.21875, 0.93750, 0.03125, 7.96875,
          2.50000, 1.28125, 1.87500, 7.96875, 1.87500, 2.65625, 1.28125, 7.96875,
          1.71875, 2.06250, 0.12500, 7.96875, 2.93750, 1.84375, 1.46875, 7.96875,
          2.00000, 2.59375, 3.00000, 7.96875, 1.90625, 0.93750, 2.96875, 7.96875,
          2.75000, 2.96875, 1.96875, 7.96875, 3.00000, 2.43750, 0.50000, 7.96875,
          2.53125, 2.78125, 2.53125, 7.96875, 0.06250, 0.56250, 1.09375, 7.96875,
  };
  size_t width = 8, pitch = 10, height = 6;
  DeviceBuffer<float> input;
  DeviceBuffer<int16_t> tested;
  std::vector<int16_t> tested_host;
  input.from_host(test_data);
  tested.resize(reference_data.size());
  tested_host.resize(reference_data.size());

  cudaMemset(tested.data(), 0x5C, tested.size()*sizeof(tested[0]));
  optical_flow::kernel::EncodeFlowComponents(input.data(), tested.data(), pitch, width, height, 0);
  CUDA_CALL(cudaDeviceSynchronize());
  copyD2H(tested_host.data(), tested.data(), reference_data.size());

  for (size_t i = 0; i < reference_data.size(); i++) {
    EXPECT_EQ(reference_data[i], tested_host[i]) << "Failed on index: " << i;
  }
}

}  // namespace kernel


TEST(OpticalFlowTuringTest, DecodeFlowVectorTest) {
  // (In)sanity test
  using std::vector;
  vector<int16_t> test_data = {101, -32376, 676, 3453, -23188};
  vector<float> ref_data = {3.15625f, -1011.75f, 21.125f, 107.90625f, -724.625f};
  for (size_t i = 0; i < test_data.size(); i++) {
    EXPECT_NEAR(ref_data[i], optical_flow::kernel::decode_flow_component(test_data[i]),
                kFlowVectorEpsilon);
  }
}

}  // namespace testing
}  // namespace optical_flow

}  // namespace dali
