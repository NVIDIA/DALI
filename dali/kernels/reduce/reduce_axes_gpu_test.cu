// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <random>
#include <utility>
#include <vector>
#include "dali/core/dev_buffer.h"
#include "dali/kernels/reduce/reduce_axes_gpu_impl.cuh"
#include "dali/kernels/reduce/reduce_test.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/format.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/cuda_event.h"

namespace dali {
namespace kernels {

template <typename Out, typename Reduction, typename T>
void RefReduceInner(Out *out, const T *in, int64_t n_outer, int64_t n_inner, const Reduction &R) {
  for (int64_t outer = 0, offset = 0; outer < n_outer; outer++, offset += n_inner) {
    out[outer] = RefReduce<Out>(make_span(in + offset, n_inner), R);
  }
}

template <typename Out, typename Reduction, typename T>
void RefReduceMiddle(Out *out, const T *in,
                     int64_t n_outer, int64_t n_reduced, int64_t n_inner,
                     const Reduction &R) {
  int64_t outer_stride = n_reduced * n_inner;
  for (int64_t outer = 0; outer < n_outer; outer++) {
    int64_t offset = outer * outer_stride;
    for (int64_t inner = 0; inner < n_inner; inner++, offset++) {
      out[outer * n_inner + inner] = RefReduce<Out>(in + offset, n_reduced, n_inner, R);
    }
  }
}


using int_dist = std::uniform_int_distribution<int>;

template <typename Reduction>
class ReduceInnerGPUTest : public ::testing::Test {
 public:
  using SampleDesc = ReduceSampleDesc<float, float>;

  void PrepareData(int N, int_dist outer_shape_dist, int_dist inner_shape_dist) {
    std::mt19937_64 rng;

    TensorListShape<2> tls;
    TensorListShape<2> out_tls;

    this->N = N;
    tls.resize(N);
    out_tls.resize(N);
    for (int i = 0; i < N; i++) {
      int outer = outer_shape_dist(rng);
      int inner = inner_shape_dist(rng);
      tls.set_tensor_shape(i, { outer, inner });
    }
    in.reshape(tls);
    auto cpu_in = in.cpu();
    UniformRandomFill(cpu_in, rng, 0, 1);

    for (int i = 0; i < N; i++) {
      auto ts = tls[i];
      SampleDesc desc;
      desc.n_outer = ts[0];
      desc.n_reduced = ts[1];
      desc.n_inner = 1;  // no inner dimensions
      desc.num_macroblocks = 1;
      desc.macroblock_size = desc.n_reduced;
      while (desc.macroblock_size > 0x8000) {
        desc.num_macroblocks <<= 1;
        desc.macroblock_size = div_ceil(desc.n_reduced, desc.num_macroblocks);
      }
      out_tls.set_tensor_shape(i, { desc.n_outer, desc.num_macroblocks });
      cpu_descs.push_back(desc);
    }
    out.reshape(out_tls);

    auto gpu_in = in.gpu();
    auto gpu_out = out.gpu();
    for (int i = 0; i < N; i++) {
      SampleDesc &desc = cpu_descs[i];
      desc.in = gpu_in.data[i];
      desc.out = gpu_out.data[i];
    }
  }

  void Run() {
    auto gpu_in = in.gpu();
    auto gpu_out = out.gpu();

    int xgrid = std::max(32, 1024 / N);
    dim3 grid(xgrid, N);
    dim3 block(32, 32);
    auto start = CUDAEvent::CreateWithFlags(0);
    auto end =   CUDAEvent::CreateWithFlags(0);
    gpu_descs.from_host(cpu_descs);
    CUDA_CALL(cudaEventRecord(start));
    ReduceInnerKernel<float><<<grid, block>>>(gpu_descs.data(), reduction);
    CUDA_CALL(cudaEventRecord(end));
    CUDA_CALL(cudaDeviceSynchronize());
    float t = 0;
    cudaEventElapsedTime(&t, start, end);
    t /= 1000;  // convert to seconds
    int64_t read = gpu_in.num_elements() * sizeof(float);
    int64_t written = gpu_out.num_elements() * sizeof(float);
    std::cerr << (read + written) / t * 1e-9 << " GB/s" << endl;
    CheckResult();
  }

  void CheckResult() {
    auto cpu_out = out.cpu();
    auto cpu_in = in.cpu();
    auto in_shape = cpu_in.shape;
    auto out_shape = cpu_out.shape;

    vector<float> ref_out;
    vector<float> full_out;  // when out is not a full reduction, we calculate the second stage here
    for (int i = 0; i < N; i++) {
      auto ts = in_shape[i];
      int64_t outer = ts[0];
      int64_t inner = ts[1];
      ref_out.resize(outer);
      RefReduceInner<float>(ref_out.data(), cpu_in.data[i], outer, inner, reduction);
      auto out = cpu_out[i];
      if (out.shape[1] > 1) {
        full_out.resize(outer);
        RefReduceInner(full_out.data(), out.data, outer, out.shape[1], reduction);
        out = make_tensor_cpu<2>(full_out.data(), { outer, 1 });
      }
      auto ref = make_tensor_cpu<2>(ref_out.data(), { outer, 1 });
      Check(out, ref, EqualEpsRel(1e-6, 1e-6));
    }
  }

  Reduction reduction;
  int N;
  TestTensorList<float, 2> in, out;
  std::vector<SampleDesc> cpu_descs;
  DeviceBuffer<SampleDesc> gpu_descs;
};


using ReductionTestTypes = ::testing::Types<reductions::sum, reductions::min, reductions::max>;

TYPED_TEST_SUITE(ReduceInnerGPUTest, ReductionTestTypes);

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_2_63) {
  this->PrepareData(10, int_dist(10000, 200000), int_dist(2, 63));
  this->Run();
}

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_32_256) {
  this->PrepareData(10, int_dist(10000, 20000), int_dist(32, 256));
  this->Run();
}

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_64_1024) {
  this->PrepareData(10, int_dist(10000, 20000), int_dist(64, 1024));
  this->Run();
}

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_1k_32k) {
  this->PrepareData(10, int_dist(1, 100), int_dist(1024, 32*1024));
  this->Run();
}

TYPED_TEST(ReduceInnerGPUTest, ReduceInner_16k_1M) {
  this->PrepareData(10, int_dist(1, 10), int_dist(16*1024, 1024*1024));
  this->Run();
}



template <typename Reduction>
class ReduceMiddleGPUTest : public ::testing::Test {
 public:
  using SampleDesc = ReduceSampleDesc<float, float>;

  void PrepareData(int N,
                   int_dist outer_shape_dist,
                   int_dist reduced_shape_dist,
                   int_dist inner_shape_dist) {
    std::uniform_real_distribution<float> dist(0, 1);
    std::mt19937_64 rng;

    TensorListShape<3> tls;
    TensorListShape<3> out_tls;

    this->N = N;
    tls.resize(N);
    out_tls.resize(N);
    for (int i = 0; i < N; i++) {
      int outer = outer_shape_dist(rng);
      int reduced = reduced_shape_dist(rng);
      int inner = inner_shape_dist(rng);
      tls.set_tensor_shape(i, { outer, reduced, inner });
    }
    in.reshape(tls);
    auto cpu_in = in.cpu();
    UniformRandomFill(cpu_in, rng, 0, 1);

    for (int i = 0; i < N; i++) {
      auto ts = tls[i];
      SampleDesc desc;
      desc.n_outer = ts[0];
      desc.n_reduced = ts[1];
      desc.n_inner = ts[2];
      desc.num_macroblocks = 1;
      desc.macroblock_size = desc.n_reduced;
      while (desc.macroblock_size > 0x8000) {
        desc.num_macroblocks <<= 1;
        desc.macroblock_size = div_ceil(desc.n_reduced, desc.num_macroblocks);
      }
      out_tls.set_tensor_shape(i, { desc.n_outer, desc.num_macroblocks, desc.n_inner });
      cpu_descs.push_back(desc);
    }
    out.reshape(out_tls);

    auto gpu_in = in.gpu();
    auto gpu_out = out.gpu();
    for (int i = 0; i < N; i++) {
      SampleDesc &desc = cpu_descs[i];
      desc.in = gpu_in.data[i];
      desc.out = gpu_out.data[i];
    }
  }

  void Run() {
    auto gpu_in = in.gpu();
    auto gpu_out = out.gpu();
    gpu_descs.from_host(cpu_descs);

    int xgrid = std::max(32, 1024 / N);
    dim3 grid(xgrid, N);
    dim3 block(32, 24);
    auto start = CUDAEvent::CreateWithFlags(0);
    auto end =   CUDAEvent::CreateWithFlags(0);
    CUDA_CALL(cudaEventRecord(start));
    ReduceMiddleKernel<float><<<grid, block, sizeof(float)*32*33>>>(gpu_descs.data(), reduction);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaEventRecord(end));
    CUDA_CALL(cudaDeviceSynchronize());
    float t = 0;
    CUDA_CALL(cudaEventElapsedTime(&t, start, end));
    t /= 1000;  // convert to seconds
    int64_t read = gpu_in.num_elements() * sizeof(float);
    int64_t written = gpu_out.num_elements() * sizeof(float);
    std::cerr << (read + written) / t * 1e-9 << " GB/s" << endl;
    CheckResult();
  }

  void CheckResult() {
    auto cpu_out = out.cpu();
    auto cpu_in = in.cpu();
    auto in_shape = cpu_in.shape;
    auto out_shape = cpu_out.shape;

    vector<float> ref_out;
    vector<float> full_out;  // when out is not a full reduction, we calculate the second stage here
    for (int i = 0; i < N; i++) {
      auto ts = in_shape[i];
      print(std::cerr, "Checking sample #", i, " of shape ", ts, "\n");
      int64_t outer = ts[0];
      int64_t reduced = ts[1];
      int64_t inner = ts[2];
      ref_out.resize(outer * inner);
      RefReduceMiddle(ref_out.data(), cpu_in.data[i], outer, reduced, inner, reduction);
      auto out = cpu_out[i];
      if (out.shape[1] > 1) {
        full_out.resize(outer * inner);
        RefReduceMiddle(full_out.data(), out.data, outer, out.shape[1], inner, reduction);
        out = make_tensor_cpu<3>(full_out.data(), { outer, 1, inner });
      }
      auto ref = make_tensor_cpu<3>(ref_out.data(), { outer, 1, inner });
      Check(out, ref, EqualEpsRel(1e-6, 1e-6));
    }
  }

  Reduction reduction;
  int N;
  TestTensorList<float, 3> in, out;
  std::vector<SampleDesc> cpu_descs;
  DeviceBuffer<SampleDesc> gpu_descs;
};


using ReductionTestTypes = ::testing::Types<reductions::sum, reductions::min, reductions::max>;

TYPED_TEST_SUITE(ReduceMiddleGPUTest, ReductionTestTypes);

TYPED_TEST(ReduceMiddleGPUTest, ReduceMiddle_Medium_Small_Small) {
  this->PrepareData(10, int_dist(1000, 20000), int_dist(2, 63), int_dist(1, 32));
  this->Run();
}

TYPED_TEST(ReduceMiddleGPUTest, ReduceMiddle_Medium_Medium_Small) {
  this->PrepareData(10, int_dist(1000, 2000), int_dist(32, 256), int_dist(1, 32));
  this->Run();
}

TYPED_TEST(ReduceMiddleGPUTest, ReduceMiddle_Small_Large_Small) {
#ifdef NDEBUG
  this->PrepareData(10, int_dist(10, 20), int_dist(20480, 102400), int_dist(10, 32));
#else
  this->PrepareData(10, int_dist(3, 5), int_dist(2<<10, 5<<10), int_dist(10, 32));
#endif
  this->Run();
}

TYPED_TEST(ReduceMiddleGPUTest, ReduceMiddle_Small_Large_Medium) {
#ifdef NDEBUG
  this->PrepareData(10, int_dist(3, 5), int_dist(16<<10, 100<<10), int_dist(32, 256));
#else
  this->PrepareData(10, int_dist(3, 5), int_dist(2<<10, 5<<10), int_dist(32, 256));
#endif
  this->Run();
}

TYPED_TEST(ReduceMiddleGPUTest, ReduceOuter_Large_Small) {
#ifdef NDEBUG
  this->PrepareData(10, int_dist(1, 1), int_dist(1<<18, 1<<20), int_dist(3, 8));
#else
  this->PrepareData(10, int_dist(1, 1), int_dist(1<<15, 1<<18), int_dist(3, 8));
#endif
  this->Run();
}

#ifdef NDEBUG
TYPED_TEST(ReduceMiddleGPUTest, ReduceOuter_B1_16M_3) {
  this->PrepareData(1, int_dist(1, 1), int_dist(16<<20, 16<<20), int_dist(3, 3));
  this->Run();
}
#endif

TEST(ReduceSamples, Sum) {
  std::mt19937_64 rng(12345);
  TestTensorList<float, 2> in, out;
  TensorShape<2> sample_shape = { 480, 640 };
#ifdef NDEBUG
  int N = 64;
#else
  int N = 16;
#endif
  reductions::sum R;
  auto in_shape = uniform_list_shape(N, sample_shape);
  auto out_shape = uniform_list_shape(1, sample_shape);
  in.reshape(in_shape);
  out.reshape(out_shape);

  auto cpu_in = in.cpu();
  UniformRandomFill(cpu_in, rng, 0, 1);

  auto gpu_in = in.gpu();
  DeviceBuffer<const float *> sample_ptrs;
  sample_ptrs.resize(N);

  int64_t n = volume(sample_shape);
  auto gpu_out = out.gpu();

  auto start = CUDAEvent::CreateWithFlags(0);
  auto end =   CUDAEvent::CreateWithFlags(0);
  sample_ptrs.from_host(gpu_in.data);
  CUDA_CALL(cudaEventRecord(start));
  ReduceSamplesKernel<float><<<256, 1024>>>(gpu_out.data[0], sample_ptrs.data(), n, N, R);
  CUDA_CALL(cudaEventRecord(end));
  auto cpu_out = out.cpu();
  float t = 0;
  CUDA_CALL(cudaEventElapsedTime(&t, start, end));
  t /= 1000;  // convert to seconds
  int64_t read = sizeof(float) * in_shape.num_elements();
  int64_t written = sizeof(float) * out_shape.num_elements();
  std::cerr << (read + written) / t * 1e-9 << " GB/s" << endl;

  for (int64_t j = 0; j < n; j++) {
    OnlineReducer<float, decltype(R)> red;
    red = {};
    for (int i = 0; i < N; i++) {
      red.add(cpu_in.data[i][j]);
    }
    float ref = red.result();
    EXPECT_NEAR(cpu_out.data[0][j], ref, 1e-6f) << " at offset " << j;
  }
}


}  // namespace kernels
}  // namespace dali
