// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cmath>
#include <vector>

#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/test/dali_operator_test.h"

namespace dali {
namespace testing {

// Global tolerance for CPU vs GPU RMSE in CLAHE tests
constexpr double kClaheCpuGpuTolerance = 5.0;

class ClaheOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    batch_size_ = 4;
    height_ = 256;
    width_ = 256;
    channels_ = 3;
    device_id_ = 0;
  }

  // Create test data - simple gradient pattern
  void CreateTestData(TensorList<CPUBackend> &data) {
    data.Resize(uniform_list_shape(batch_size_, {height_, width_, channels_}), DALI_UINT8);

    for (int i = 0; i < batch_size_; i++) {
      auto *tensor_data = data.mutable_tensor<uint8_t>(i);

      // Create a test pattern with varying contrast in different regions
      for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
          for (int c = 0; c < channels_; c++) {
            int idx = (y * width_ + x) * channels_ + c;

            // Create different patterns in different quadrants
            uint8_t value;
            if (y < height_ / 2 && x < width_ / 2) {
              // Low contrast gradient
              value = static_cast<uint8_t>(64 + (x + y) * 32 / (height_ + width_));
            } else if (y < height_ / 2) {
              // High contrast blocks
              value = ((x / 32) % 2) ? 200 : 50;
            } else if (x < width_ / 2) {
              // Medium contrast sine pattern
              value = static_cast<uint8_t>(128 + 64 * sin(x * 0.1f) * sin(y * 0.1f));
            } else {
              // Dark region with some detail
              value = static_cast<uint8_t>(32 + (x + y) * 16 / (height_ + width_));
            }

            tensor_data[idx] = value;
          }
        }
      }
    }
  }

  // Compare two tensor lists and return RMSE
  double CompareTensorLists(const TensorList<CPUBackend> &tl1, const TensorList<CPUBackend> &tl2) {
    EXPECT_EQ(tl1.num_samples(), tl2.num_samples());

    double total_squared_error = 0.0;
    int total_elements = 0;

    for (int i = 0; i < tl1.num_samples(); i++) {
      EXPECT_EQ(tl1.tensor_shape(i), tl2.tensor_shape(i));

      auto data1 = tl1.tensor<uint8_t>(i);
      auto data2 = tl2.tensor<uint8_t>(i);
      int num_elements = tl1.tensor_shape(i).num_elements();

      for (int j = 0; j < num_elements; j++) {
        double diff = static_cast<double>(data1[j]) - static_cast<double>(data2[j]);
        total_squared_error += diff * diff;
      }

      total_elements += num_elements;
    }

    return std::sqrt(total_squared_error / total_elements);
  }

  // Test CPU vs GPU CLAHE implementation
  void TestCpuGpuEquivalence(int tiles_x, int tiles_y, float clip_limit, bool luma_only) {
    // Create test data
    TensorList<CPUBackend> input_data;
    CreateTestData(input_data);

    // CPU Pipeline
    Pipeline cpu_pipe(batch_size_, 1, device_id_);
    cpu_pipe.AddExternalInput("input");
    cpu_pipe.AddOperator(OpSpec("Clahe")
                             .AddArg("device", "cpu")
                             .AddArg("tiles_x", tiles_x)
                             .AddArg("tiles_y", tiles_y)
                             .AddArg("clip_limit", clip_limit)
                             .AddArg("luma_only", luma_only)
                             .AddInput("input", StorageDevice::CPU)
                             .AddOutput("output", StorageDevice::CPU));

    std::vector<std::pair<std::string, std::string>> cpu_outputs = {{"output", "cpu"}};
    cpu_pipe.Build(cpu_outputs);

    // GPU Pipeline
    Pipeline gpu_pipe(batch_size_, 1, device_id_);
    gpu_pipe.AddExternalInput("input");
    gpu_pipe.AddOperator(OpSpec("Clahe")
                             .AddArg("device", "gpu")
                             .AddArg("tiles_x", tiles_x)
                             .AddArg("tiles_y", tiles_y)
                             .AddArg("clip_limit", clip_limit)
                             .AddArg("luma_only", luma_only)
                             .AddInput("input", StorageDevice::GPU)
                             .AddOutput("output", StorageDevice::GPU));

    std::vector<std::pair<std::string, std::string>> gpu_outputs = {{"output", "gpu"}};
    gpu_pipe.Build(gpu_outputs);

    // Run CPU pipeline
    cpu_pipe.SetExternalInput("input", input_data);
    Workspace cpu_ws;
    cpu_pipe.Run();
    cpu_pipe.Outputs(&cpu_ws);

    // Run GPU pipeline
    gpu_pipe.SetExternalInput("input", input_data);
    Workspace gpu_ws;
    gpu_pipe.Run();
    gpu_pipe.Outputs(&gpu_ws);

    // Copy GPU results to CPU for comparison
    auto &cpu_output = cpu_ws.Output<CPUBackend>(0);
    auto &gpu_output_device = gpu_ws.Output<GPUBackend>(0);

    TensorList<CPUBackend> gpu_output;
    gpu_output.Copy(gpu_output_device);

    // Compare results
    double rmse = CompareTensorLists(cpu_output, gpu_output);


  EXPECT_LT(rmse, kClaheCpuGpuTolerance) << "RMSE between CPU and GPU CLAHE too high: " << rmse
                      << " (tiles=" << tiles_x << "x" << tiles_y << ", clip=" << clip_limit
                      << ", luma_only=" << luma_only << ")";

    std::cout << "CLAHE CPU vs GPU RMSE: " << rmse << " (tiles=" << tiles_x << "x" << tiles_y
              << ", clip=" << clip_limit << ", luma_only=" << luma_only << ")" << std::endl;
  }

  int batch_size_;
  int height_, width_, channels_;
  int device_id_;
};

// Test basic functionality
TEST_F(ClaheOpTest, BasicCpuGpuEquivalence) {
  TestCpuGpuEquivalence(8, 8, 2.0f, true);
}

// Test different luma modes
TEST_F(ClaheOpTest, LumaOnlyVsPerChannel) {
  TestCpuGpuEquivalence(8, 8, 2.0f, true);   // Luma only
  TestCpuGpuEquivalence(8, 8, 2.0f, false);  // Per channel
}

// Test different tile sizes
TEST_F(ClaheOpTest, DifferentTileSizes) {
  TestCpuGpuEquivalence(4, 4, 2.0f, true);
  TestCpuGpuEquivalence(16, 16, 2.0f, true);
  TestCpuGpuEquivalence(4, 8, 2.0f, true);  // Non-square tiles
}

// Test different clip limits
TEST_F(ClaheOpTest, DifferentClipLimits) {
  TestCpuGpuEquivalence(8, 8, 1.0f, true);  // Low enhancement
  TestCpuGpuEquivalence(8, 8, 4.0f, true);  // High enhancement
}

// Test error handling
TEST_F(ClaheOpTest, ErrorHandling) {
  TensorList<CPUBackend> input_data;
  CreateTestData(input_data);

  Pipeline pipe(batch_size_, 1, device_id_);
  pipe.AddExternalInput("input");

  // Test invalid tile count (should not crash, but may throw)
  EXPECT_NO_THROW({
    pipe.AddOperator(OpSpec("Clahe")
                         .AddArg("device", "cpu")
                         .AddArg("tiles_x", 1)
                         .AddArg("tiles_y", 1)
                         .AddArg("clip_limit", 2.0f)
                         .AddArg("luma_only", true)
                         .AddInput("input", StorageDevice::CPU)
                         .AddOutput("output", StorageDevice::CPU));
  });
}

}  // namespace testing
}  // namespace dali
