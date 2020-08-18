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
#include <cstring>
#include "dali/kernels/test/resampling_test/transpose_outer.h"
#include "dali/kernels/alloc.h"

namespace dali {
namespace testing {

void TransposeOuterCPU(void *_out, const void *_in,
                       int64_t rows, int64_t cols, int64_t inner_size) {
  char *out = static_cast<char *>(_out);
  const char *in = static_cast<const char *>(_in);
  ptrdiff_t ostride = rows * inner_size;
  ptrdiff_t istride = cols * inner_size;

  int64_t itile = 32;
  int64_t jtile = 128 / inner_size;
  if (!jtile) jtile = 1;

  for (int64_t i = 0; i < rows; i += itile) {
    int64_t imax = i + itile;
    if (imax > rows) imax = rows;
    for (int64_t j = 0; j <  cols; j += jtile) {
      int jmax = j + jtile;
      if (jmax > cols) jmax = cols;
      for (int64_t ii = i; ii < imax; ii++) {
        for (int64_t jj = j; jj < jmax; jj++) {
          std::memcpy(&out[jj * ostride + ii * inner_size],
                      &in[ii * istride + jj * inner_size],
                      inner_size);
        }
      }
    }
  }
}

template <typename T>
__global__ void TransposeOuterKernel(T *out, const T *in,
                                     int64_t rows, int64_t cols, int64_t inner_size) {
  ptrdiff_t ostride = rows * inner_size;
  ptrdiff_t istride = cols * inner_size;

  int i = blockIdx.y;
  int j = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= rows || j >= cols)
    return;
  auto *out_span = &out[ostride * j + i * inner_size];
  auto *in_span = &in[istride * i + j * inner_size];
  for (int64_t k = threadIdx.x; k < inner_size; k += blockDim.x) {
    out_span[k] = in_span[k];
  }
}

template <typename T>
void TransposeOuterGPU(T *out, const T *in, int64_t rows, int64_t cols, int64_t inner_size,
                  cudaStream_t stream) {
  dim3 block = { 32, 32, 1 };
  unsigned gx = div_ceil(cols, 32);
  unsigned gy = rows;
  dim3 grid = { gx, gy, 1 };
  TransposeOuterKernel<<<grid, block, 0, stream>>>(out, in, rows, cols, inner_size);
}

void TransposeOuterGPU(void *out, const void *in, int64_t rows, int64_t cols, int64_t inner_size,
                       cudaStream_t stream) {
  if ((inner_size & 7) == 0) {
    TransposeOuterGPU(static_cast<int64_t*>(out), static_cast<const int64_t*>(in),
                      rows, cols, inner_size >> 3, stream);
  } else if ((inner_size & 3) == 0) {
    TransposeOuterGPU(static_cast<int32_t*>(out), static_cast<const int32_t*>(in),
                      rows, cols, inner_size >> 2, stream);
  } else if ((inner_size & 1) == 0) {
    TransposeOuterGPU(static_cast<int16_t*>(out), static_cast<const int16_t*>(in),
                      rows, cols, inner_size >> 1, stream);
  } else {
    TransposeOuterGPU(static_cast<char *>(out), static_cast<const char*>(in),
                      rows, cols, inner_size, stream);
  }
}

template <typename Storage = StorageCPU, typename T, int A0, int A1>
TensorView<Storage, T, 2> as_tensor(T (&a)[A0][A1]) {
  return { &a[0][0], { A0, A1 } };
}

template <typename Storage = StorageCPU, typename T, int A0, int A1, int A2>
TensorView<Storage, T, 3> as_tensor(T (&a)[A0][A1][A2]) {
  return { &a[0][0][0], { A0, A1, A2 } };
}

TEST(TransposeOuterTest, CPU) {
  int in2[3][4] = {
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
    { 9, 10, 11, 12 }
  };
  int in3[3][4][2] = {
    { { 1, 10 }, { 2, 20 }, { 3, 30 }, { 4, 40 } },
    { { 5, 50 }, { 6, 60 }, { 7, 70 }, { 8, 80 } },
    { { 9, 90 }, { 10, 100 }, { 11, 110 }, { 12, 120 } }
  };
  int out_buf[4*3*2];

  {
    auto out = make_tensor_cpu<2>(out_buf, { 4, 3 });
    TransposeOuter(out, as_tensor(in2));
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        EXPECT_EQ(*out(i, j), in2[j][i]);
      }
    }
  }
  {
    auto out = make_tensor_cpu<3>(out_buf, { 4, 3, 2});
    TransposeOuter(out, as_tensor(in3));
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 2; k++) {
          EXPECT_EQ(*out(i, j, k), in3[j][i][k]);
        }
      }
    }
  }
}

TEST(TransposeOuterTest, GPU) {
  int d = 143;
  int h = 231;
  int w = 55;
  int N = w*h*d;
  TensorShape<3> in_shape(d, h, w);
  TensorShape<3> out_shape(h, d, w);
  std::vector<int> host_in(N), host_out(N);
  for (int i = 0; i < N; i++)
    host_in[i] = i + 1;
  auto dev_in = kernels::memory::alloc_unique<int>(kernels::AllocType::GPU, N);
  auto dev_out = kernels::memory::alloc_unique<int>(kernels::AllocType::GPU, N);
  cudaMemcpy(dev_in.get(), host_in.data(), N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(dev_out.get(), 0, N * sizeof(int));
  TransposeOuter(make_tensor_gpu(dev_out.get(), out_shape),
                 make_tensor_gpu(dev_in.get(), in_shape));
  cudaMemcpy(host_out.data(), dev_out.get(), N * sizeof(int), cudaMemcpyDeviceToHost);

  auto cpu_in = make_tensor_cpu(host_in.data(), in_shape);
  auto cpu_out = make_tensor_cpu(host_out.data(), out_shape);
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < d; j++) {
      for (int k = 0; k < w; k++) {
        ASSERT_EQ(*cpu_out(i, j, k), *cpu_in(j, i, k)) << " at " << i << ", " << j << ", " << k;
      }
    }
  }
}

}  // namespace testing
}  // namespace dali
