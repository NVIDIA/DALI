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

#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include <numeric>

#include "dali/kernels/alloc.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/test/mat2tensor.h"
#include "dali/pipeline/operators/optical_flow/turing_of/optical_flow_turing.h"
#include "dali/test/dali_test_config.h"
#include "dali/util/cuda_utils.h"

namespace dali {
namespace optical_flow {
namespace testing {

namespace {
constexpr float kFlowVectorEpsilon = 1.f / 32;


std::tuple<kernels::TensorView<kernels::StorageGPU, uint8_t, 3>,
        kernels::memory::KernelUniquePtr<uint8_t>>
mat_to_tensor(cv::Mat &mat) {
  auto tvcpu = kernels::view_as_tensor<uint8_t, 3>(mat);
  auto mem = kernels::memory::alloc_unique<uint8_t>(kernels::AllocType::Unified,
                                                    mat.cols * mat.rows * mat.channels());
  auto tvgpu = kernels::make_tensor_gpu<3>(mem.get(), {mat.rows, mat.cols, mat.channels()});
  kernels::copy(tvgpu, tvcpu);
  return std::forward_as_tuple(tvgpu, std::move(mem));
}

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
    uint8_t *input, *tested;
    CUDA_CALL(cudaMallocManaged(&input, input_data.size()));
    CUDA_CALL(cudaMallocManaged(&tested, reference_data.size()));
    CUDA_CALL(cudaMemcpy(input, input_data.data(), input_data.size(), cudaMemcpyDefault));

    auto w = Grayscale ? width_gray_ : width_;
    auto h = Grayscale ? height_gray_ : height_;
    auto p = Grayscale ? pitch_gray_ : pitch_;

    cc(input, tested, p, w, h, 0);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < reference_data.size(); i++) {
      EXPECT_EQ(reference_data[i], tested[i]) << "Failed on index: " << i;
    }

    CUDA_CALL(cudaFree(tested));
    CUDA_CALL(cudaFree(input));
  }


  const std::vector<uint8_t> reference_data_ = {
          73, 5, 47, 255, 71, 30, 1, 255, 0, 0,
          80, 41, 60, 255, 60, 85, 41, 255, 0, 0,
          55, 66, 4, 255, 94, 59, 47, 255, 0, 0,
          64, 83, 96, 255, 61, 30, 95, 255, 0, 0,
          88, 95, 63, 255, 96, 78, 16, 255, 0, 0,
          81, 89, 81, 255, 2, 18, 35, 255, 0, 0,
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
          73, 5, 47, 255, 71, 30, 1, 255, 0, 0,
          80, 41, 60, 255, 60, 85, 41, 255, 0, 0,
          55, 66, 4, 255, 94, 59, 47, 255, 0, 0,
          64, 83, 96, 255, 61, 30, 95, 255, 0, 0,
          88, 95, 63, 255, 96, 78, 16, 255, 0, 0,
          81, 89, 81, 255, 2, 18, 35, 255, 0, 0,
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
  float *input;
  int16_t *tested;
  CUDA_CALL(cudaMallocManaged(&input, test_data.size() * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&tested, reference_data.size() * sizeof(int16_t)));
  CUDA_CALL(
          cudaMemcpy(input, test_data.data(), test_data.size() * sizeof(float), cudaMemcpyDefault));

  optical_flow::kernel::EncodeFlowComponents(input, tested, pitch, width, height, 0);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < reference_data.size(); i++) {
    EXPECT_EQ(reference_data[i], tested[i]) << "Failed on index: " << i;
  }

  CUDA_CALL(cudaFree(tested));
  CUDA_CALL(cudaFree(input));
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


// DISABLED due to lack of test data. Enable on next possible chance
TEST(OpticalFlowTuringTest, DISABLED_CudaDecodeFlowVectorTest) {
  using std::ifstream;
  using std::string;
  using std::vector;
  auto test_data_path = dali::testing::dali_extra_path() + "/db/optical_flow/flow_vector_decode/";
  ifstream infile(test_data_path + "flow_vector_test_data.txt"),
          reffile(test_data_path + "flow_vector_ground_truth.txt");
  ASSERT_TRUE(infile && reffile) << "Error accessing test data";
  vector<int16_t> test_data;
  int16_t valin;
  float valref;
  while (infile >> valin) {
    test_data.push_back(valin);
  }
  vector<float> ref_data;
  while (reffile >> valref) {
    ref_data.push_back(valref);
  }
  ASSERT_EQ(ref_data.size(), test_data.size());
  void *incuda;
  float *outcuda;
  size_t inpitch;

  size_t inwidth = 200, inheight = 50;
  ASSERT_EQ(test_data.size(), inwidth * inheight) << "Provided dims don't match with data size";

  CUDA_CALL(cudaMallocPitch(&incuda, &inpitch, inwidth * sizeof(int16_t), inheight));
  CUDA_CALL(cudaMemcpy2D(incuda, inpitch, test_data.data(), inwidth * sizeof(int16_t),
                         inwidth * sizeof(int16_t), inheight, cudaMemcpyDefault));
  CUDA_CALL(cudaMallocManaged(&outcuda, ref_data.size() * sizeof(float)));
  optical_flow::kernel::DecodeFlowComponents(static_cast<int16_t *> (incuda), outcuda,
                                             inpitch, inwidth, inheight);
  CUDA_CALL(cudaDeviceSynchronize());
  for (size_t i = 0; i < ref_data.size(); i++) {
    EXPECT_NEAR(ref_data[i], outcuda[i], kFlowVectorEpsilon) << "Failed on idx: " << i;
  }
  CUDA_CALL(cudaFree(incuda));
  CUDA_CALL(cudaFree(outcuda));
}


// DISABLED due to lack of test data. Enable on next possible chance
TEST(OpticalFlowTuringTest, DISABLED_CalcOpticalFlowTest) {
  using namespace std;  // NOLINT

  auto test_data_path = dali::testing::dali_extra_path() + "/db/optical_flow/slow_preset/";

  // Reference
  auto matref = cv::imread(test_data_path + string("frame_reference.png"));
  cv::cvtColor(matref, matref, cv::COLOR_BGR2RGB);
  assert(matref.isContinuous() && matref.channels() == 3);
  auto ref = mat_to_tensor(matref);
  auto tvref = get<0>(ref);
  auto memref = reinterpret_cast<uint8_t *>(get<1>(ref).get());

  // Input
  auto matin = cv::imread(test_data_path + string("frame_input.png"));
  cv::cvtColor(matin, matin, cv::COLOR_BGR2RGB);
  assert(matin.isContinuous() && matin.channels() == 3);
  auto in = mat_to_tensor(matin);
  auto tvin = get<0>(in);
  auto memin = reinterpret_cast<uint8_t *>(get<1>(in).get());

  ASSERT_EQ(matref.size, matin.size) << "Sizes of test data don't match";
  auto width = matref.cols;
  auto height = matref.rows;
  auto channels = matref.channels();


  OpticalFlowParams params = {0.0, VectorGridSize::SIZE_4, true, false};
  try {
    OpticalFlowTuring of(params, width, height, channels, DALIImageType::DALI_RGB);

    // Output
    auto out_shape = of.GetOutputShape().to_static<3>();
    auto memout = kernels::memory::alloc_unique<float>(kernels::AllocType::Unified,
                                                       volume(out_shape));
    auto tvout = kernels::make_tensor_gpu(memout.get(), out_shape);

    of.CalcOpticalFlow(tvref, tvin, tvout);
    CUDA_CALL(cudaDeviceSynchronize());

    // Read reference data
    ifstream reffile(test_data_path + "decoded_flow_vector.dat");
    vector<float> reference_data;
    copy(istream_iterator<float>(reffile),
         istream_iterator<float>(),
         back_inserter(reference_data));

    ASSERT_EQ(reference_data.size(), tvout.num_elements())
                          << "Number of output elements doesn't match";
    vector<float> distances(reference_data.size());
    for (size_t i = 0; i < distances.size(); i++) {
      distances[i] = abs(reference_data[i] - tvout.data[i]);
    }
    float mean_err = accumulate(distances.begin(), distances.end(), 0.f) / distances.size();
    // Expecting, that average error would be less than 0.5[px]
    ASSERT_GT(0.5, mean_err);
  } catch (unsupported_exception &) {
    GTEST_SKIP() << "Test skipped due to module unavailability";
  }
}


// DISABLED due to lack of test data. Enable on next possible chance
TEST(OpticalFlowTuringTest, DISABLED_CalcOpticalFlowExternalHintsTest) {
  using namespace std;  // NOLINT

  auto test_data_path = dali::testing::dali_extra_path() + "/db/optical_flow/slow_preset/";

  // Reference frame
  auto matref = cv::imread(test_data_path + string("frame_reference.png"));
  cv::cvtColor(matref, matref, cv::COLOR_BGR2RGB);
  assert(matref.isContinuous() && matref.channels() == 3);
  auto ref = mat_to_tensor(matref);
  auto tvref = get<0>(ref);
  auto memref = reinterpret_cast<uint8_t *>(get<1>(ref).get());


  // Input frame
  auto matin = cv::imread(test_data_path + string("frame_input.png"));
  cv::cvtColor(matin, matin, cv::COLOR_BGR2RGB);
  assert(matin.isContinuous() && matin.channels() == 3);
  auto in = mat_to_tensor(matin);
  auto tvin = get<0>(in);
  auto memin = reinterpret_cast<uint8_t *>(get<1>(in).get());

  ASSERT_EQ(matref.size, matin.size) << "Sizes of test data don't match";
  auto width = matref.cols;
  auto height = matref.rows;
  auto channels = matref.channels();


  // Read reference data (used also as external hints)
  ifstream reffile(test_data_path + "decoded_flow_vector.dat");
  vector<float> reference_data;
  copy(istream_iterator<float>(reffile),
       istream_iterator<float>(),
       back_inserter(reference_data));
  auto tvhints = kernels::make_tensor_gpu<3>(reference_data.data(), {
          (height + 3) / 4, (width + 3) / 4, 2  // Shape has 2 "channels": y and x components
  });


  OpticalFlowParams params = {0.0, VectorGridSize::SIZE_4, false, true};
  try {
    OpticalFlowTuring of(params, width, height, channels, DALIImageType::DALI_RGB);

    // Output frame
    auto out_shape = of.GetOutputShape().to_static<3>();
    auto memout = kernels::memory::alloc_unique<float>(kernels::AllocType::Unified,
                                                       volume(out_shape));
    auto tvout = kernels::make_tensor_gpu(memout.get(), out_shape);

    of.CalcOpticalFlow(tvref, tvin, tvout, tvhints);
    CUDA_CALL(cudaDeviceSynchronize());


    ASSERT_EQ(reference_data.size(), tvout.num_elements())
                          << "Number of output elements doesn't match";
    vector<float> distances(reference_data.size());
    for (size_t i = 0; i < distances.size(); i++) {
      distances[i] = abs(reference_data[i] - tvout.data[i]);
    }
    float mean_err = accumulate(distances.begin(), distances.end(), 0.f) / distances.size();
    // Expecting, that average error would be less than 0.5[px]
    EXPECT_GT(0.5, mean_err);
  } catch (unsupported_exception &) {
    GTEST_SKIP() << "Test skipped due to module unavailability";
  }
}
}  // namespace testing
}  // namespace optical_flow

}  // namespace dali
