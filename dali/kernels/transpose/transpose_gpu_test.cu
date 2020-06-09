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

#include "dali/kernels/transpose/transpose_gpu_impl.cuh"   // NOLINT
#include "dali/kernels/transpose/transpose_gpu_setup.cuh"  // NOLINT
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

namespace {
// All 4-element permutations
const int Permutations4[][4] = {
  { 0, 1, 2, 3 },
  { 0, 1, 3, 2 },
  { 0, 2, 1, 3 },
  { 0, 2, 3, 1 },
  { 0, 3, 1, 2 },
  { 0, 3, 2, 1 },
  { 1, 0, 2, 3 },
  { 1, 0, 3, 2 },
  { 1, 2, 0, 3 },
  { 1, 2, 3, 0 },
  { 1, 3, 0, 2 },
  { 1, 3, 2, 0 },
  { 2, 0, 1, 3 },
  { 2, 0, 3, 1 },
  { 2, 1, 0, 3 },
  { 2, 1, 3, 0 },
  { 2, 3, 0, 1 },
  { 2, 3, 1, 0 },
  { 3, 0, 1, 2 },
  { 3, 0, 2, 1 },
  { 3, 1, 0, 2 },
  { 3, 1, 2, 0 },
  { 3, 2, 0, 1 },
  { 3, 2, 1, 0 }
};
}  // namespace

TEST(TransposeTiled, AllPerm4DInnermost) {
  TensorShape<> shape = { 19, 57, 37, 53 };  // a bunch of primes, just to make it harder
  int size = volume(shape);
  vector<int> in_cpu(size), out_cpu(size), ref(size);
  std::iota(in_cpu.begin(), in_cpu.end(), 0);
  DeviceBuffer<int> in_gpu, out_gpu;
  in_gpu.resize(size);
  out_gpu.resize(size);
  copyH2D(in_gpu.data(), in_cpu.data(), size);

  int grid_size = 2;
  ASSERT_LT(grid_size * 512, size) << "Weak test error: Grid too large to test grid loop";

  for (auto &perm : Permutations4) {
    if (perm[3] == 3)
      continue;  // innermost dim must be permuted

    std::cerr << "Testing permutation "
      << perm[0] << " " << perm[1] << " " << perm[2] << " " << perm[3] << "\n";
    cudaMemset(out_gpu, 0xff, size*sizeof(int));

    TiledTransposeDesc<int> desc;
    memset(&desc, 0xCC, sizeof(desc));
    InitTiledTranspose(desc, out_gpu, in_gpu, shape, make_span(perm), grid_size);
    TransposeTiledSingle<<<grid_size, dim3(32, 16), kTiledTransposeMaxSharedMem>>>(desc);
    copyD2H(out_cpu.data(), out_gpu.data(), size);
    RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm, 4);

    for (int i = 0; i < size; i++) {
      ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
    }
  }
}


TEST(TransposeTiled, BuildDescVectorized) {
  TensorShape<> shape = { 57, 37, 53, 4 };  // a bunch of primes, just to make it harder
  int size = volume(shape);
  vector<int> in_cpu(size), out_cpu(size), ref(size);
  std::iota(in_cpu.begin(), in_cpu.end(), 0);
  DeviceBuffer<int> in_gpu, out_gpu;
  in_gpu.resize(size);
  out_gpu.resize(size);
  cudaMemset(out_gpu, 0xff, size*sizeof(int));
  copyH2D(in_gpu.data(), in_cpu.data(), size);

  SmallVector<int, 6> perm = { 1, 2, 0, 3 };

  int grid_size = 1024;
  TiledTransposeDesc<int> desc;
  memset(&desc, 0xCC, sizeof(desc));
  InitTiledTranspose(desc, out_gpu, in_gpu, shape, make_span(perm), grid_size);
  EXPECT_EQ(desc.lanes, 4) << "Lanes not detected";
  EXPECT_EQ(desc.ndim, 3) << "Number of dimensions should have shrunk in favor of lanes";
  TransposeTiledSingle<<<grid_size, dim3(32, 16), kTiledTransposeMaxSharedMem>>>(desc);
  copyD2H(out_cpu.data(), out_gpu.data(), size);
  RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm.data(), perm.size());

  for (int i = 0; i < size; i++) {
    ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
  }
}


TEST(TransposeDeinterleave, AllPerm4DInnermost) {
  TensorShape<> shape = { 19, 57, 37, 3 };  // small inner dimension
  int size = volume(shape);
  vector<int> in_cpu(size), out_cpu(size), ref(size);
  std::iota(in_cpu.begin(), in_cpu.end(), 0);
  DeviceBuffer<int> in_gpu, out_gpu;
  in_gpu.resize(size);
  out_gpu.resize(size);
  copyH2D(in_gpu.data(), in_cpu.data(), size);

  int grid_size = 3;
  ASSERT_LT(grid_size * 512 * 3, size) << "Weak test error: Grid too large to test grid loop";

  for (auto &perm : Permutations4) {
    if (perm[3] == 3)
      continue;  // innermost dim must be permuted

    std::cerr << "Testing permutation "
      << perm[0] << " " << perm[1] << " " << perm[2] << " " << perm[3] << "\n";
    cudaMemset(out_gpu, 0xff, size*sizeof(int));

    DeinterleaveDesc<int> desc;
    memset(&desc, 0xCC, sizeof(desc));
    InitDeinterleave(desc, out_gpu, in_gpu, shape, make_span(perm));
    TransposeDeinterleaveSingle<<<grid_size, dim3(512)>>>(desc);
    copyD2H(out_cpu.data(), out_gpu.data(), size);
    RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm, 4);

    for (int i = 0; i < size; i++) {
      ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
    }
  }
}

TEST(TransposeGeneric, AllPerm4D) {
  TensorShape<> shape = { 31, 43, 53, 47 };
  int size = volume(shape);
  vector<int> in_cpu(size), out_cpu(size), ref(size);
  std::iota(in_cpu.begin(), in_cpu.end(), 0);
  DeviceBuffer<int> in_gpu, out_gpu;
  in_gpu.resize(size);
  out_gpu.resize(size);
  copyH2D(in_gpu.data(), in_cpu.data(), size);

  int grid_size = 3;
  ASSERT_LT(grid_size * 512, size) << "Weak test error: Grid too large to test grid loop";

  for (auto &perm : Permutations4) {
    std::cerr << "Testing permutation "
      << perm[0] << " " << perm[1] << " " << perm[2] << " " << perm[3] << "  input shape "
      << shape << "\n";
    cudaMemset(out_gpu, 0xff, size*sizeof(int));

    GenericTransposeDesc<int> desc;
    memset(&desc, 0xCC, sizeof(desc));
    InitGenericTanspose(desc, out_gpu, in_gpu, shape, make_span(perm));
    TransposeGenericSingle<<<grid_size, dim3(512)>>>(desc);
    copyD2H(out_cpu.data(), out_gpu.data(), size);
    RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm, 4);

    for (int i = 0; i < size; i++) {
      ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
    }

    TensorShape<> simplified_shape;
    SmallVector<int, 6> simplified_perm;
    SimplifyPermute(simplified_shape, simplified_perm, shape.data(), perm, 4);

    if (simplified_shape == shape) {
      for (int i = 0; i < 4; i++) {
        ASSERT_EQ(simplified_perm[i], perm[i]);
      }
      // no simplification, don't repeat the test
      continue;
    }

    std::cerr << "Testing permutation ";
    for (auto i : simplified_perm)
      std::cerr << i << " ";
    std::cerr << " input shape " << simplified_shape << "\n";

    memset(&desc, 0xCC, sizeof(desc));
    InitGenericTanspose(desc, out_gpu, in_gpu, simplified_shape, make_span(simplified_perm));
    TransposeGenericSingle<<<grid_size, dim3(512)>>>(desc);
    copyD2H(out_cpu.data(), out_gpu.data(), size);

    for (int i = 0; i < size; i++) {
      ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
    }

  }
}


}  // namespace kernels
}  // namespace dali
