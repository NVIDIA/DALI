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
#include <limits>
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

void MakeRandomBatch(TensorList<CPUBackend> &data, int N,
                     const TensorShape<> &min_sh,
                     const TensorShape<> &max_sh) {
  assert(min_sh.sample_dim() == max_sh.sample_dim());
  int ndim = min_sh.sample_dim();
  for (int d = 0; d < ndim; d++)
    assert(min_sh[d] <=  max_sh[d]);

  std::mt19937_64 rng(1234);
  TensorListShape<> tl_sh(N, ndim);
  for (int sample_idx = 0; sample_idx < N; sample_idx++) {
    auto sh = tl_sh.tensor_shape_span(sample_idx);
    for (int d = 0; d < ndim; d++) {
      std::uniform_int_distribution<int> dist(min_sh[d], max_sh[d]);
      sh[d] = dist(rng);
    }
  }
  std::uniform_int_distribution<int> dist2(0, std::numeric_limits<uint8_t>::max());
  data.Resize(tl_sh, DALI_UINT8);
  for (int sample_idx = 0; sample_idx < N; sample_idx++) {
    int64_t vol = volume(tl_sh.tensor_shape_span(sample_idx));
    uint8_t *ptr = data.mutable_tensor<uint8_t>(sample_idx);
    for (int64_t i = 0; i < vol; i++)
      ptr[i] = dist2(rng);
  }
}

void CheckResults(const DeviceWorkspace& ws, int batch_size, int i,
                  TensorList<CPUBackend> &data, int output_idx) {
  TensorList<CPUBackend> res_cpu;
  if (ws.OutputIsType<GPUBackend>(output_idx)) {
    res_cpu.Copy(ws.Output<GPUBackend>(output_idx));
    CUDA_CALL(cudaDeviceSynchronize());
  } else {
    res_cpu.Copy(ws.Output<CPUBackend>(output_idx));
  }

  auto res_cpu_view = view<uint8_t>(res_cpu);
  auto data_view = view<uint8_t>(data);
  for (int j = 0; j < batch_size; ++j) {
    int data_idx = (i * batch_size + j) % data_view.shape.num_samples();
    Check(res_cpu_view[j], data_view[data_idx]);
  }
}

}  // namespace test
}  // namespace dali
