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
#include <memory>
#include "dali/pipeline/util/copy_with_stride.h"
#include "dali/core/dev_buffer.h"

namespace dali {

TEST(CopyWithStrideTest, OneDim) {
  float data[] = {1, 2, 3, 4, 5, 6};
  std::array<float, 3> out;
  Index stride = 2 * sizeof(float);
  Index shape = 3;
  CopyWithStride<CPUBackend>(out.data(), data, &stride, &shape, 1, sizeof(float));
  ASSERT_TRUE((out == std::array<float, 3>{1, 3, 5}));
}

TEST(CopyWithStrideTest, TwoDims)  {
  size_t data[] = {11, 12, 13, 14,
                   21, 22, 23, 24,
                   31, 32, 33, 34,
                   41, 42, 43, 44};
  std::array<size_t, 8> out;
  Index stride[] = {8 * sizeof(size_t), sizeof(size_t)};
  Index shape[] = {2, 4};
  CopyWithStride<CPUBackend>(out.data(), data, stride, shape, 2, sizeof(size_t));
  ASSERT_TRUE((out == std::array<size_t, 8>{11, 12, 13, 14,
                                            31, 32, 33, 34}));
}

TEST(CopyWithStrideTest, SimpleCopy) {
  uint8 data[] = {1, 2,
                  3, 4,

                  5, 6,
                  7, 8};
  std::array<uint8, 8> out;
  Index stride[] = {4, 2, 1};
  Index shape[] = {2, 2, 2};
  CopyWithStride<CPUBackend>(out.data(), data, stride, shape, 3, 1);
  ASSERT_TRUE((out == std::array<uint8, 8>{1, 2,
                                           3, 4,

                                           5, 6,
                                           7, 8}));
}

TEST(CopyWithStrideTest, OneDimGPU) {
  float h_data[] = {1, 2, 3, 4, 5, 6};
  Index stride = 2 * sizeof(float);
  Index shape = 3;
  DeviceBuffer<float> data, out;
  data.from_host(h_data);
  out.resize(data.size());
  CopyWithStride<GPUBackend>(out, data, &stride, &shape, 1, sizeof(float));
  std::array<float, 3> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), out, 3 * sizeof(float), cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<float, 3>{1, 3, 5}));
}

TEST(CopyWithStrideTest, TwoDimsGPU) {
  size_t h_data[] = {11, 12, 13, 14,
                     21, 22, 23, 24,
                     31, 32, 33, 34,
                     41, 42, 43, 44};
  Index stride[] = {8 * sizeof(size_t), sizeof(size_t)};
  Index shape[] = {2, 4};
  DeviceBuffer<size_t> data, out;
  data.from_host(h_data);
  out.resize(data.size());
  CopyWithStride<GPUBackend>(out, data, stride, shape, 2, sizeof(size_t));
  std::array<size_t , 8> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), out, 8 * sizeof(size_t), cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<size_t, 8>{11, 12, 13, 14,
                                              31, 32, 33, 34}));
}

TEST(CopyWithStrideTest, SimpleCopyGPU) {
  uint8 h_data[] = {1, 2,
                    3, 4,

                    5, 6,
                    7, 8};
  Index stride[] = {4, 2, 1};
  Index shape[] = {2, 2, 2};
  DeviceBuffer<uint8> data, out;
  data.from_host(h_data);
  out.resize(data.size());
  CopyWithStride<GPUBackend>(out, data, stride, shape, 3, sizeof(uint8));
  std::array<uint8 , 8> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), out, 8 * sizeof(uint8), cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<uint8, 8>{1, 2,
                                             3, 4,

                                             5, 6,
                                             7, 8}));
}

}  // namespace dali
