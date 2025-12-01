// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_event.h"
#include "dali/kernels/transpose/transpose_test.h"
#include "dali/core/cuda_rt_utils.h"

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

TEST(TransposeGPU, GetTransposeMethod) {
  {
    TensorShape<> shape = { 640*480, 3 };
    int perm[] = { 1, 0 };
    EXPECT_EQ(GetTransposeMethod(shape.data(), perm, 2, sizeof(int)),
              TransposeMethod::Deinterleave);
  }
  {
    TensorShape<> shape = { 3, 640*480 };
    int perm[] = { 1, 0 };  // interleave
    EXPECT_EQ(GetTransposeMethod(shape.data(), perm, 2, sizeof(int)),
              TransposeMethod::Interleave);
  }
  {
    TensorShape<> shape = { 640, 480 };
    int perm[] = { 1, 0 };  // scalar tiled
    EXPECT_EQ(GetTransposeMethod(shape.data(), perm, 2, sizeof(int)),
              TransposeMethod::Tiled);
  }
  {
    TensorShape<> shape = { 20, 640, 480 };
    int perm[] = { 1, 2, 0 };  // scalar tiled
    EXPECT_EQ(GetTransposeMethod(shape.data(), perm, 3, sizeof(int)),
              TransposeMethod::Tiled);
  }
  {
    TensorShape<> shape = { 640, 480, 3 };
    int perm[] = { 1, 0, 2 };  // vectorized tiled
    EXPECT_EQ(GetTransposeMethod(shape.data(), perm, 3, sizeof(int)),
              TransposeMethod::Tiled);
  }
  {
    TensorShape<> shape = { 640, 3, 480 };
    int perm[] = { 1, 2, 0 };  // some mess
    EXPECT_EQ(GetTransposeMethod(shape.data(), perm, 3, sizeof(int)),
              TransposeMethod::Generic);
  }
  {
    TensorShape<> shape = { 640, 480, 50 };
    int perm[] = { 1, 0, 2 };  // generic stuff
    EXPECT_EQ(GetTransposeMethod(shape.data(), perm, 3, sizeof(int)),
              TransposeMethod::Generic);
  }
  {
    TensorShape<> shape = { 640*480 };
    int perm[] = { 0 };  // identity
    EXPECT_EQ(GetTransposeMethod(shape.data(), perm, 1, sizeof(int)),
              TransposeMethod::Copy);
  }
}


template <typename Function>
inline int GetMaxBlockHeight(int preferred_size, const Function &f) {
  int max_threads = MaxThreadsPerBlock(f);
  assert(max_threads >= kTileSize);

  int block_y = 16;  // start with 32x16 block and try smaller until found
  while (kTileSize * block_y > max_threads)
    block_y >>= 1;

  return block_y;
}

TEST(TransposeTiled, AllPerm4DInnermost) {
  TensorShape<> shape = { 19, 57, 37, 53 };  // a bunch of primes, just to make it harder
  int size = volume(shape);
  vector<int> in_cpu(size), out_cpu(size), ref(size);
  std::iota(in_cpu.begin(), in_cpu.end(), 0);
  DeviceBuffer<int> in_gpu, out_gpu;
  in_gpu.resize(size);
  out_gpu.resize(size);
  copyH2D(in_gpu.data(), in_cpu.data(), size);
  auto start = CUDAEvent::CreateWithFlags(0);
  auto end = CUDAEvent::CreateWithFlags(0);

  int grid_size = std::max(1, size / 512);
  ASSERT_LT(grid_size * 512, size) << "Weak test error: Grid too large to test grid loop";

  int block_y = GetMaxBlockHeight(16, TransposeTiledSingle<int>);

  for (auto &perm : testing::Permutations4) {
    if (perm[3] == 3)
      continue;  // innermost dim must be permuted

    std::cerr << "Testing permutation "
      << perm[0] << " " << perm[1] << " " << perm[2] << " " << perm[3] << "\n";
    CUDA_CALL(cudaMemset(out_gpu, 0xff, size*sizeof(*in_gpu.data())));

    TiledTransposeDesc<int> desc;
    memset(&desc, 0xCC, sizeof(desc));
    InitTiledTranspose(desc, shape, make_span(perm), out_gpu, in_gpu, grid_size);
    CUDA_CALL(cudaEventRecord(start));
    TransposeTiledSingle<<<grid_size, dim3(32, block_y), kTiledTransposeMaxSharedMem>>>(desc);
    CUDA_CALL(cudaEventRecord(end));
    copyD2H(out_cpu.data(), out_gpu.data(), size);
    testing::RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm, 4);

    float time;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));
    time *= 1e+6;
    std::cerr << 2*size*sizeof(*in_gpu.data()) / time << " GB/s" << "\n";

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
  CUDA_CALL(cudaMemset(out_gpu, 0xff, size*sizeof(*in_gpu.data())));
  copyH2D(in_gpu.data(), in_cpu.data(), size);

  SmallVector<int, 6> perm = { 1, 2, 0, 3 };

  int block_y = GetMaxBlockHeight(16, TransposeTiledSingle<int>);

  int grid_size = 1024;
  TiledTransposeDesc<int> desc;
  memset(&desc, 0xCC, sizeof(desc));
  InitTiledTranspose(desc, shape, make_span(perm), out_gpu, in_gpu, grid_size);
  EXPECT_EQ(desc.lanes, 4) << "Lanes not detected";
  EXPECT_EQ(desc.ndim, 3) << "Number of dimensions should have shrunk in favor of lanes";
  TransposeTiledSingle<<<grid_size, dim3(32, block_y), kTiledTransposeMaxSharedMem>>>(desc);
  copyD2H(out_cpu.data(), out_gpu.data(), size);
  testing::RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm.data(), perm.size());

  for (int i = 0; i < size; i++) {
    ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
  }
}

TEST(TransposeTiled, BuildDescAndForceMisalignment) {
  TensorShape<> shape = { 57, 37, 52, 4 };  // a bunch of primes, just to make it harder
  int size = volume(shape);
  vector<uint8_t> in_cpu(size + 4), out_cpu(size + 4);
  vector<uint8_t> ref(size + 4);

  DeviceBuffer<uint8_t> in_gpu, out_gpu;
  in_gpu.resize(size + 4);
  out_gpu.resize(size + 4);

  int block_y = GetMaxBlockHeight(16, TransposeTiledSingle<uint8_t>);;

  for (uintptr_t offset = 0; offset < 4; offset++) {
    std::iota(in_cpu.begin(), in_cpu.end(), 0);
    CUDA_CALL(cudaMemset(out_gpu, 0xff, size*sizeof(*in_gpu.data())));

    copyH2D(in_gpu.data() + offset, in_cpu.data(), size);

    SmallVector<int, 6> perm = { 1, 2, 0, 3 };

    int grid_size = 1024;
    TiledTransposeDesc<uint8_t> desc;
    memset(&desc, 0xCC, sizeof(desc));
    InitTiledTranspose(desc, shape, make_span(perm), out_gpu.data() + offset,
                       in_gpu.data() + offset, grid_size);
    EXPECT_EQ(desc.lanes, 4) << "Lanes not detected";
    EXPECT_EQ(desc.ndim, 3) << "Number of dimensions should have shrunk in favor of lanes";

    TransposeTiledSingle<<<grid_size, dim3(32, block_y), kTiledTransposeMaxSharedMem>>>(desc);
    copyD2H(out_cpu.data(), out_gpu.data() + offset, size);
    testing::RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm.data(), perm.size());

    for (int i = 0; i < size; i++) {
      ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
    }
  }
}

TEST(TransposeTiled, BuildDescVectorized16BitOpt) {
  TensorShape<> shape = { 57, 37, 53, 4 };  // a bunch of primes, just to make it harder
  int size = volume(shape);
  vector<uint16_t> in_cpu(size), out_cpu(size);
  vector<uint16_t> ref(size);
  std::iota(in_cpu.begin(), in_cpu.end(), 0);
  DeviceBuffer<uint16_t> in_gpu, out_gpu;
  in_gpu.resize(size);
  out_gpu.resize(size);
  CUDA_CALL(cudaMemset(out_gpu, 0xff, size*sizeof(*in_gpu.data())));
  copyH2D(in_gpu.data(), in_cpu.data(), size);

  SmallVector<int, 6> perm = { 1, 2, 0, 3 };

  int block_y = GetMaxBlockHeight(16, TransposeTiledSingle<uint16_t>);

  int grid_size = 1024;
  TiledTransposeDesc<uint16_t> desc;
  memset(&desc, 0xCC, sizeof(desc));
  InitTiledTranspose(desc, shape, make_span(perm), out_gpu, in_gpu, grid_size);
  EXPECT_EQ(desc.lanes, 4) << "Lanes not detected";
  EXPECT_EQ(desc.ndim, 3) << "Number of dimensions should have shrunk in favor of lanes";

  TransposeTiledSingle<<<grid_size, dim3(32, block_y), kTiledTransposeMaxSharedMem>>>(desc);
  copyD2H(out_cpu.data(), out_gpu.data(), size);
  testing::RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm.data(), perm.size());

  for (int i = 0; i < size; i++) {
    ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
  }
}

TEST(TransposeTiled, HighDimensionTest) {
  TensorShape<> shape = {3, 3, 5, 7, 23, 3, 37, 4 };  // a bunch of primes, just to make it harder
  int size = volume(shape);
  vector<uint8_t> in_cpu(size), out_cpu(size);
  vector<uint8_t> ref(size);

  DeviceBuffer<uint8_t> in_gpu, out_gpu;
  in_gpu.resize(size);
  out_gpu.resize(size);

  int block_y = GetMaxBlockHeight(16, TransposeTiledSingle<uint8_t>);

  for (int size_of_last_dim = 1; size_of_last_dim <= 4; size_of_last_dim++) {
    shape = { 3, 3, 5, 7, 23, 3, 37, size_of_last_dim };
    size = volume(shape);
    std::iota(in_cpu.begin(), in_cpu.end(), 0);
    CUDA_CALL(cudaMemset(out_gpu, 0xff, size*sizeof(*in_gpu.data())));

    copyH2D(in_gpu.data(), in_cpu.data(), size);

    SmallVector<int, 8> perm = { 1, 0, 4, 2, 6, 3, 5, 7 };

    int grid_size = 1024;
    TiledTransposeDesc<uint8_t> desc;
    memset(&desc, 0xCC, sizeof(desc));
    InitTiledTranspose(desc, shape, make_span(perm), out_gpu.data(), in_gpu.data(), grid_size);

    TransposeTiledSingle<<<grid_size, dim3(32, block_y), kTiledTransposeMaxSharedMem>>>(desc);
    copyD2H(out_cpu.data(), out_gpu.data(), size);
    testing::RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm.data(), perm.size());

    for (int i = 0; i < size; i++) {
      ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
    }
  }
}

TEST(TransposeDeinterleave, AllPerm4DInnermost) {
  int channels = 3;
  TensorShape<> shape = { 19, 157, 137, channels };  // small inner dimension
  int size = volume(shape);
  vector<int> in_cpu(size), out_cpu(size), ref(size);
  std::iota(in_cpu.begin(), in_cpu.end(), 0);
  DeviceBuffer<int> in_gpu, out_gpu;
  in_gpu.resize(size);
  out_gpu.resize(size);
  copyH2D(in_gpu.data(), in_cpu.data(), size);

  int block_size = 256;
  int grid_size = std::max(1, size / (block_size * channels));
  ASSERT_LT(grid_size * block_size * channels, size)
      << "Weak test error: Grid too large to test grid loop";

  auto start = CUDAEvent::CreateWithFlags(0);
  auto end = CUDAEvent::CreateWithFlags(0);

  for (auto &perm : testing::Permutations4) {
    if (perm[3] == 3)
      continue;  // innermost dim must be permuted

    std::cerr << "Testing permutation "
      << perm[0] << " " << perm[1] << " " << perm[2] << " " << perm[3] << "\n";
      CUDA_CALL(cudaMemset(out_gpu, 0xff, size*sizeof(*in_gpu.data())));

    DeinterleaveDesc<int> desc;
    memset(&desc, 0xCC, sizeof(desc));
    InitDeinterleave(desc, shape, make_span(perm), out_gpu, in_gpu);
    CUDA_CALL(cudaEventRecord(start));
    TransposeDeinterleaveSingle<<<grid_size, block_size>>>(desc);
    CUDA_CALL(cudaEventRecord(end));
    copyD2H(out_cpu.data(), out_gpu.data(), size);
    testing::RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm, 4);

    float time;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));
    time *= 1e+6;
    std::cerr << 2*size*sizeof(*in_gpu.data()) / time << " GB/s" << "\n";


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

  int grid_size = 2048;
  int block_size = 256;
  ASSERT_LT(grid_size * block_size, size) << "Weak test error: Grid too large to test grid loop";

  for (auto &perm : testing::Permutations4) {
    std::cerr << "Testing permutation "
      << perm[0] << " " << perm[1] << " " << perm[2] << " " << perm[3] << "  input shape "
      << shape << "\n";
      CUDA_CALL(cudaMemset(out_gpu, 0xff, size*sizeof(*in_gpu.data())));

    GenericTransposeDesc<int> desc;
    memset(&desc, 0xCC, sizeof(desc));
    InitGenericTranspose(desc, shape, make_span(perm), out_gpu, in_gpu);
    TransposeGenericSingle<<<grid_size, block_size>>>(desc);
    copyD2H(out_cpu.data(), out_gpu.data(), size);

    testing::RefTranspose(ref.data(), in_cpu.data(), shape.data(), perm, 4);

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
    CUDA_CALL(cudaMemset(out_gpu, 0xff, size*sizeof(*in_gpu.data())));
    InitGenericTranspose(desc, simplified_shape, make_span(simplified_perm), out_gpu, in_gpu);
    TransposeGenericSingle<<<grid_size, block_size>>>(desc);
    copyD2H(out_cpu.data(), out_gpu.data(), size);

    for (int i = 0; i < size; i++) {
      ASSERT_EQ(out_cpu[i], ref[i]) << " at " << i;
    }
  }
}

}  // namespace kernels
}  // namespace dali
