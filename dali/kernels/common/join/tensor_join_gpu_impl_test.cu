// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <iostream>
#include <random>
#include "dali/kernels/common/join/tensor_join_cpu.h"
#include "dali/kernels/common/join/tensor_join_gpu_impl.h"
#include "dali/kernels/common/join/tensor_join_gpu_impl.cuh"
#include "dali/core/dev_buffer.h"
#include "dali/core/cuda_event.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {
namespace kernels {

template <typename T>
void RefTLSJoin(const OutListCPU<T> &out, span<const InListCPU<T> *const> in, int axis) {
  SmallVector<InTensorCPU<T>, 8> in_tensors;
  int njoin = in.size();
  in_tensors.resize(njoin);

  int N = in[0]->num_samples();

  for (int i = 0; i < N; i++) {
    for (int t = 0; t < njoin; t++)
      in_tensors[t] = (*in[t])[i];
    tensor_join::ConcatenateTensors(out[i], make_cspan(in_tensors), axis);
  }
}

struct TensorJoinGPUTest : public ::testing::Test {

  void TestRawKernel() {
    InitData();

    RunRef();
    RunRawKernel();
    CheckResult();
  }

  void RunRef() {
    RefTLSJoin(ref.cpu(), make_cspan(in_cpu_ptrs), axis);
  }

  void CheckResult() {
    Check(out.cpu(), ref.cpu());
  }

  void RunRawKernel() {
    vector<tensor_join::InputDesc<int>> input_descs(N * njoin);
    vector<tensor_join::OutputDesc<int>> output_descs(N);

    auto out_tlv = out.gpu();
    FillDescs(make_span(output_descs), make_span(input_descs),
              out_tlv, make_cspan(in_gpu_ptrs), axis);


    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);

    DeviceBuffer<tensor_join::InputDesc<int>> gpu_input_descs;
    DeviceBuffer<tensor_join::OutputDesc<int>> gpu_output_descs;
    gpu_input_descs.from_host(input_descs);
    gpu_output_descs.from_host(output_descs);

    dim3 grid(1024, N);
    dim3 block(32 * 8);
    cudaEventRecord(start, 0);
    JoinTensorsKernel<<<grid, block>>>(gpu_output_descs.data(), gpu_input_descs.data(), njoin);
    cudaEventRecord(end, 0);
    CUDA_CALL(cudaDeviceSynchronize());

    float time = 0;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));
    time *= 1e+6;  // to nanoseconds
    int64_t size = 2 * out_tlv.num_elements() * sizeof(int);  // 2x because in+out
    std::cerr << "Throughput: " << size / time << " GB/s\n";

  }

  void InitData() {
    GenerateShapes();
    FillInputs();
    GenerateListPointers();
    out.reshape(out_shape);
    ref.reshape(out_shape);
    FillOutput();
  }

  void FillOutput() {
    auto out_tl = out.gpu();
    cudaMemset(out_tl.data[0], 0, sizeof(out_tl.data[0][0]) * out_tl.num_elements());
  }

  void FillInputs() {
    assert(static_cast<int>(in_shapes.size()) == njoin);
    in.resize(in_shapes.size());

    // Fill each tensor list with sequential values.
    // Each tensor list starts at a round base-10 number to improve readability in case of errors.
    int mul = 10;
    for (int t = 0; t < njoin; t++) {
      int n = in_shapes[t].num_elements();
      while (mul < n)
        mul *= 10;
    }

    for (int t = 0; t < njoin; t++) {
      int counter = t * mul + 1;
      in[t].reshape(in_shapes[t]);
      auto tl_cpu = in[t].cpu();
      Fill(tl_cpu, [&]() { return counter++; });
    }
  }


  void GenerateListPointers() {
    in_cpu_tls.resize(njoin);
    in_gpu_tls.resize(njoin);
    in_cpu_ptrs.resize(njoin);
    in_gpu_ptrs.resize(njoin);

    for (int t = 0; t < njoin; t++) {
      in_cpu_tls[t] = in[t].cpu();
      in_gpu_tls[t] = in[t].gpu();
      in_cpu_ptrs[t] = &in_cpu_tls[t];
      in_gpu_ptrs[t] = &in_gpu_tls[t];
    }
  }


  void GenerateShapes() {
    auto outer_extent_dist = uniform_distribution(1, max_outer_extent);
    auto inner_extent_dist = uniform_distribution(1, max_inner_extent);

    in_shapes.resize(njoin);

    in_shapes[0].resize(N, ndim);
    for (int i = 0; i < N; i++) {
      int a = 0;
      for (; a < axis; a++) {
        in_shapes[0].tensor_shape_span(i)[a] = outer_extent_dist(rng);
      }
      for (; a < ndim; a++) {
        in_shapes[0].tensor_shape_span(i)[a] = inner_extent_dist(rng);
      }
    }

    for (int t = 1; t < njoin; t++) {
      in_shapes[t] = in_shapes[0];
      if (!new_axis) {
        for (int i = 0; i < N; i++)
          in_shapes[t].tensor_shape_span(i)[axis] = inner_extent_dist(rng);
      }
    }

    tensor_join::JoinedShape(out_shape,
                             [&](int i) { return in_shapes[i]; }, in_shapes.size(),
                             axis, new_axis);
  }


  int N = 10, ndim = 3, njoin = 256, axis = 2, max_outer_extent = 100, max_inner_extent = 100;
  bool new_axis = true;
  std::mt19937_64 rng{12345};

  vector<TensorListShape<>> in_shapes;
  TensorListShape<> out_shape;
  vector<TestTensorList<int>> in;
  TestTensorList<int> out, ref;

  vector<InListCPU<int>> in_cpu_tls;
  vector<InListGPU<int>> in_gpu_tls;
  vector<const InListCPU<int> *> in_cpu_ptrs;
  vector<const InListGPU<int> *> in_gpu_ptrs;

};

TEST_F(TensorJoinGPUTest, Perf_ConcatLongFew) {
  max_outer_extent = 10;
  max_inner_extent = 10000;
  njoin = 7;
  new_axis = false;
  TestRawKernel();
}

TEST_F(TensorJoinGPUTest, Perf_ConcatInterleaveFew) {
  max_outer_extent = 500;
  max_inner_extent = 10;
  njoin = 7;
  new_axis = false;
  TestRawKernel();
}

TEST_F(TensorJoinGPUTest, Perf_StackMedium) {
  max_outer_extent = 200;
  max_inner_extent = 10;
  njoin = 64;
  new_axis = true;
  TestRawKernel();
}

TEST_F(TensorJoinGPUTest, Perf_StackInterleaveMany) {
  max_outer_extent = 30;
  max_inner_extent = 1;
  njoin = 1024;
  new_axis = true;
  TestRawKernel();
}


}  // namespace kernels
}  // namespace dali
