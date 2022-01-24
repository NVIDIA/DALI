// Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/test/dali_test_utils.h"
#include <gtest/gtest.h>
#include <libgen.h>
#include <limits.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace test {

std::string CurrentExecutableDir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        return dirname(result);
    }
    return {};
}

void MakeRandomBatch(TensorList<CPUBackend> &data, int N, int ndim,
                     int64_t min_extent, int64_t max_extent) {
  std::mt19937_64 rng(1234);
  TensorListShape<3> tl_sh(N);
  std::uniform_int_distribution<int> dist(min_extent, max_extent);
  for (int sample_idx = 0; sample_idx < N; sample_idx++) {
    auto sh = tl_sh.tensor_shape_span(sample_idx);
    for (int d = 0; d < ndim; d++)
      sh[d] = dist(rng);
  }
  std::uniform_int_distribution<int> dist2(0, 255);
  data.Resize(tl_sh, DALI_UINT8);
  for (int sample_idx = 0; sample_idx < N; sample_idx++) {
    int64_t vol = volume(tl_sh.tensor_shape_span(sample_idx));
    uint8_t *ptr = data.mutable_tensor<uint8_t>(sample_idx);
    for (int64_t i = 0; i < vol; i++)
      ptr[i] = dist2(rng);
  }
}

void CheckResults(DeviceWorkspace ws, int batch_size, int i,
                  TensorList<CPUBackend> &data) {
  ASSERT_EQ(ws.NumOutput(), 1);
  ASSERT_EQ(ws.NumInput(), 0);
  ASSERT_TRUE(ws.OutputIsType<GPUBackend>(0));

  TensorList<GPUBackend> &res1 = ws.Output<GPUBackend>(0);
  TensorList<CPUBackend> res_cpu;
  res_cpu.Copy(res1);
  CUDA_CALL(cudaDeviceSynchronize());

  auto res_cpu_view = view<uint8_t>(res_cpu);
  auto data_view = view<uint8_t>(data);
  for (int j = 0; j < batch_size; ++j) {
    int data_idx = (i * batch_size + j) % data_view.shape.num_samples();
    Check(res_cpu_view[j], data_view[data_idx]);
  }
}

}  // namespace test
}  // namespace dali
