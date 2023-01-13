// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/hist.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/lookup.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace hist {
namespace test {

constexpr cudaStream_t cuda_stream = 0;

class EqualizeHistGpuTest : public ::testing::Test {
 protected:
  void Run(int batch_size) {
    // Check(out_view_cpu, baseline_.cpu(), EqualEps(grad_step));
  }

  std::mt19937_64 rng{12345};
};

TEST_F(EqualizeHistGpuTest, Channel1Batch1) {
  this->Run(1);
}

TEST_F(EqualizeHistGpuTest, Channel3Batch1) {
  this->Run(1);
}

TEST_F(EqualizeHistGpuTest, Channel3Batch17) {
  this->Run(1);
}

TEST_F(EqualizeHistGpuTest, Channel4Batch5) {
  this->Run(1);
}

TEST_F(EqualizeHistGpuTest, Channel7Batch17) {
  this->Run(1);
}

}  // namespace test
}  // namespace hist
}  // namespace equalize
}  // namespace kernels
}  // namespace dali
