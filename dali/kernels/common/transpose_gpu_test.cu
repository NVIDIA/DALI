#// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/common/transpose_gpu_impl.cuh"   // NOLINT
#include "dali/kernels/common/transpose_gpu_setup.cuh"  // NOLINT
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include "dali/core/dev_buffer.h"
#include "dali/kernels/common/utils.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

using namespace transpose_impl;  // NOLINT

template <typename T, typename Extent>
void RefTranspose(T *out, const T *in, const uint64_t *out_strides, const uint64_t *in_strides, const Extent *shape, int ndim) {
  if (ndim == 0) {
    *out = *in;
  } else {
    for (Extent i = 0; i < *shape; i++) {
      RefTranspose(out, in, out_strides + 1, in_strides + 1, shape + 1, ndim - 1);
      out += *out_strides;
      in += *in_strides;
    }
  }
}

template <typename T, typename Extent>
void RefTranspose(T *out, const T *in, const Extent *in_shape, const int *perm, int ndim) {
  uint64_t out_strides[32], in_strides[32], tmp_strides[32], out_shape[32];
  CalcStrides(tmp_strides, in_shape, ndim);
  for (int i = 0; i < ndim; i++) {
    out_shape[i] = in_shape[perm[i]];
  }
  CalcStrides(out_strides, out_shape, ndim);
  for (int i = 0; i < ndim; i++) {
    in_strides[i] = tmp_strides[perm[i]];
  }

  RefTranspose(out, in, out_strides, in_strides, out_shape, ndim);
}

TEST(TransposeTiled, Transpose1032) {
  TiledTransposeDesc<int> desc = {};
  const int A = 43;
  const int B = 20;
  const int C = 3;
  const int D = 40;
  uint64_t in_shape[] = { A, B, C, D };
  const uint64_t size = A*B*C*D;
  DeviceBuffer<int> in_gpu, out_gpu;
  vector<int> in, out, ref;
  in.resize(size);
  out.resize(size);
  ref.resize(size);
  in_gpu.resize(size);
  out_gpu.resize(size);
  cudaMemset(out_gpu, 0xff, size*sizeof(int));
  desc.in = in_gpu;
  desc.out = out_gpu;
  desc.shape[0] = A;
  desc.shape[1] = B;
  desc.shape[2] = C;
  desc.shape[3] = D;

  uint64_t out_shape[] = { B, A, D, C };
  CalcStrides(desc.out_strides, out_shape, 4);
  CalcStrides(desc.in_strides, in_shape, 4);
  std::swap(desc.out_strides[0], desc.out_strides[1]);
  std::swap(desc.out_strides[2], desc.out_strides[3]);


  desc.tiles_x = div_ceil(desc.shape[3], 32);
  desc.tiles_y = div_ceil(desc.shape[2], 32);
  desc.tiles_per_slice = desc.tiles_x * desc.tiles_y;
  desc.total_tiles = A*B*desc.tiles_per_slice;
  desc.ndim = 4;
  int grid_size = 32;
  desc.tiles_per_block = div_ceil(desc.total_tiles, grid_size);

  int perm[] = { 1, 0, 3, 2 };

  for (int i = 0; i < size; i++)
    in[i] = i;

  copyH2D(in_gpu.data(), in.data(), size);

  TransposeTiledSingle<<<grid_size, dim3(32, 16)>>>(desc);
  copyD2H(out.data(), out_gpu.data(), size);
  RefTranspose(ref.data(), in.data(), in_shape, perm, 4);

  for (int i = 0; i < size; i++) {
    ASSERT_EQ(out[i], ref[i]) << " at " << i;
  }
}

TEST(TransposeTiled, BuildDesc) {
  TensorShape<> shape = { 19, 57, 37, 53 };  // a bunch of primes, just to make it harder
  int size = volume(shape);
  vector<int> in_cpu(size), out_cpu(size), ref(size);
  std::iota(in_cpu.begin(), in_cpu.end(), 0);
  DeviceBuffer<int> in_gpu, out_gpu;
  in_gpu.resize(size);
  out_gpu.resize(size);
  cudaMemset(out_gpu, 0xff, size*sizeof(int));
  copyH2D(in_gpu.data(), in_cpu.data(), size);

  SmallVector<int, 6> perm = { 3, 0, 2, 1 };

  int grid_size = 1024;
  TiledTransposeDesc<int> desc;
  InitTiledTranspose(desc, grid_size, out_gpu, in_gpu, shape, make_span(perm));
  TransposeTiledSingle<<<grid_size, dim3(32, 16)>>>(desc);
  copyD2H(out_cpu.data(), out_gpu.data(), size);
  RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm.data(), perm.size());

  for (int i = 0; i < size; i++) {
    ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
  }

}

TEST(SimplifyPermute, NoSimplification) {
  int64_t shape[] = { 2, 3, 4, 5 };
  int perm[] = { 0, 3, 2, 1 };
  TensorShape<> s_shape, ref_shape;
  SmallVector<int, 6> s_perm, ref_perm;
  SimplifyPermute(s_shape, s_perm, shape, perm, 4);
  ref_shape = { 2, 3, 4, 5 };
  ref_perm = { 0, 3, 2, 1 };
  EXPECT_EQ(s_shape, ref_shape);
  EXPECT_EQ(s_perm, ref_perm);
}

TEST(SimplifyPermute, CollapseUnitDims) {
  int64_t shape[] = { 2, 1, 3, 4, 1, 5 };
  int perm[] = { 0, 5, 1, 3, 2, 4 };
  TensorShape<> s_shape, ref_shape;
  SmallVector<int, 6> s_perm, ref_perm;
  SimplifyPermute(s_shape, s_perm, shape, perm, 6);
  ref_shape = { 2, 3, 4, 5 };
  ref_perm = { 0, 3, 2, 1 };
  EXPECT_EQ(s_shape, ref_shape);
  EXPECT_EQ(s_perm, ref_perm);
}

TEST(SimplifyPermute, Collapse) {
  int64_t shape[] = { 2, 1, 3, 4, 1, 5 };
  int perm[] = { 3, 4, 5, 0, 1, 2 };
  TensorShape<> s_shape, ref_shape;
  SmallVector<int, 6> s_perm, ref_perm;
  SimplifyPermute(s_shape, s_perm, shape, perm, 6);
  ref_shape = { 6, 20 };
  ref_perm = { 1, 0 };
  EXPECT_EQ(s_shape, ref_shape);
  EXPECT_EQ(s_perm, ref_perm);
}

}  // namespace kernels
}  // namespace dali
