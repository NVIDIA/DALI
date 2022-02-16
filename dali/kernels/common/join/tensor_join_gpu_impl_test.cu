// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include "dali/kernels/common/join/tensor_join_gpu.h"
#include "dali/kernels/common/join/tensor_join_cpu.h"
#include "dali/kernels/common/join/tensor_join_gpu_impl.cuh"
#include "dali/kernels/kernel_manager.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/geom/mat.h"

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

template <typename T>
struct TensorJoinGPUTest : public ::testing::Test {
  void TestRawKernel() {
    CUDAStream stream = CUDAStream::Create(true);
    InitData(stream);
    RunRef();
    RunRawKernel(stream);
    CheckResult(stream);
  }

  void TestFullKernel(int num_iter = 3) {
    if (new_axis)
      TestFullKernelImpl<true>(num_iter);
    else
      TestFullKernelImpl<false>(num_iter);
  }

  template <bool new_axis>
  void TestFullKernelImpl(int num_iter) {
    using Kernel = TensorJoinGPU<T, new_axis>;
    CUDAStream stream = CUDAStream::Create(true);
    KernelManager mgr;
    mgr.Resize<Kernel>(1);
    KernelContext ctx;
    ctx.gpu.stream = stream;

    for (int iter = 0; iter < num_iter; iter++) {
      InitData(stream);
      RunRef();

      KernelRequirements &req = mgr.Setup<Kernel>(0, ctx, make_cspan(in_gpu_tls), axis);
      ASSERT_EQ(req.output_shapes.size(), 1);
      ASSERT_EQ(req.output_shapes[0], out_shape);
      mgr.Run<Kernel>(0, ctx, out.gpu(stream), make_cspan(in_gpu_tls));

      CUDA_CALL(cudaStreamSynchronize(stream));
      CheckResult(stream);
    }
  }

  void RunRef() {
    RefTLSJoin(ref.cpu(), make_cspan(in_cpu_ptrs), axis);
  }

  void CheckResult(cudaStream_t stream) {
    Check(out.cpu(stream), ref.cpu(stream));
  }

  void RunRawKernel(cudaStream_t stream) {
    vector<tensor_join::InputDesc<T>> input_descs(N * njoin);
    vector<tensor_join::OutputDesc<T>> output_descs(N);

    auto out_tlv = out.gpu(stream);
    FillDescs(make_span(output_descs), make_span(input_descs),
              out_tlv, make_cspan(in_gpu_ptrs), axis);


    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);

    DeviceBuffer<tensor_join::InputDesc<T>> gpu_input_descs;
    DeviceBuffer<tensor_join::OutputDesc<T>> gpu_output_descs;
    gpu_input_descs.from_host(input_descs, stream);
    gpu_output_descs.from_host(output_descs, stream);

    dim3 grid(1024, N);
    dim3 block(32 * 8);
    CUDA_CALL(cudaEventRecord(start, stream));

    tensor_join::JoinTensorsKernel<<<grid, block, 0, stream>>>(
        gpu_output_descs.data(), gpu_input_descs.data(), njoin);

    CUDA_CALL(cudaEventRecord(end, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    float time = 0;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));
    time *= 1e+6;  // to nanoseconds
    int64_t size = 2 * out_tlv.num_elements() * sizeof(T);  // 2x because in+out
    std::cerr << "Throughput: " << size / time << " GB/s\n";
  }

  void InitData(cudaStream_t stream) {
    GenerateShapes();
    FillInputs(stream);
    GenerateListPointers(stream);
    out.reshape(out_shape);
    ref.reshape(out_shape);
    ClearOutput(stream);
  }

  void ClearOutput(cudaStream_t stream) {
    auto out_tl = out.gpu(stream);
    CUDA_CALL(
      cudaMemsetAsync(out_tl.data[0], 0, sizeof(out_tl.data[0][0]) * out_tl.num_elements(),
                      stream));
  }

  void FillInputs(cudaStream_t stream) {
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
      in[t].invalidate_gpu();
    }
  }


  void GenerateListPointers(cudaStream_t stream) {
    in_cpu_tls.resize(njoin);
    in_gpu_tls.resize(njoin);
    in_cpu_ptrs.resize(njoin);
    in_gpu_ptrs.resize(njoin);

    for (int t = 0; t < njoin; t++) {
      in_cpu_tls[t] = in[t].cpu();
      in_gpu_tls[t] = in[t].gpu(stream);
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
                             [&](int i) { return &in_shapes[i]; }, in_shapes.size(),
                             axis, new_axis);
  }


  int N = 10, ndim = 3, njoin = 256, axis = 1, max_outer_extent = 100, max_inner_extent = 100;
  bool new_axis = true;
  std::mt19937_64 rng{12345};

  vector<TensorListShape<>> in_shapes;
  TensorListShape<> out_shape;
  vector<TestTensorList<T>> in;
  TestTensorList<T> out, ref;

  vector<InListCPU<T>> in_cpu_tls;
  vector<InListGPU<T>> in_gpu_tls;
  vector<const InListCPU<T> *> in_cpu_ptrs;
  vector<const InListGPU<T> *> in_gpu_ptrs;
};


using TensorJoinTypes = ::testing::Types<uint16_t, float, int64_t>;
TYPED_TEST_SUITE(TensorJoinGPUTest, TensorJoinTypes);

TYPED_TEST(TensorJoinGPUTest, RawConcatLongFew) {
  this->max_outer_extent = 10;
  this->max_inner_extent = 500;
  this->njoin = 7;
  this->new_axis = false;
  this->TestRawKernel();
}

TYPED_TEST(TensorJoinGPUTest, RawConcatInterleave2) {
  this->max_outer_extent = 400;
  this->max_inner_extent = 10;
  this->axis = 2;
  this->njoin = 2;
  this->new_axis = false;
  this->TestRawKernel();
}

TYPED_TEST(TensorJoinGPUTest, RawConcatInterleave3) {
  this->max_outer_extent = 400;
  this->max_inner_extent = 10;
  this->axis = 2;
  this->njoin = 3;
  this->new_axis = false;
  this->TestRawKernel();
}

TYPED_TEST(TensorJoinGPUTest, RawConcatInterleave4) {
  this->max_outer_extent = 400;
  this->max_inner_extent = 10;
  this->axis = 2;
  this->njoin = 4;
  this->new_axis = false;
  this->TestRawKernel();
}


TYPED_TEST(TensorJoinGPUTest, RawConcatInterleave7) {
  this->max_outer_extent = 300;
  this->max_inner_extent = 10;
  this->axis = 2;
  this->njoin = 7;
  this->new_axis = false;
  this->TestRawKernel();
}


TYPED_TEST(TensorJoinGPUTest, RawConcatInterleave15) {
  this->max_outer_extent = 200;
  this->max_inner_extent = 10;
  this->axis = 2;
  this->njoin = 15;
  this->new_axis = false;
  this->TestRawKernel();
}

TYPED_TEST(TensorJoinGPUTest, RawStackMedium) {
  this->max_outer_extent = 2000;
  this->max_inner_extent = 4;
  this->ndim = 3;
  this->axis = 1;
  this->njoin = 64;
  this->new_axis = true;
  this->TestRawKernel();
}

TYPED_TEST(TensorJoinGPUTest, RawStackInterleave2) {
  this->max_outer_extent = 1000;
  this->ndim = 2;
  this->axis = 2;
  this->njoin = 2;
  this->new_axis = true;
  this->TestRawKernel();
}


TYPED_TEST(TensorJoinGPUTest, RawStackInterleave3) {
  this->max_outer_extent = 800;
  this->ndim = 2;
  this->axis = 2;
  this->njoin = 3;
  this->new_axis = true;
  this->TestRawKernel();
}

TYPED_TEST(TensorJoinGPUTest, RawStackInterleave4) {
  this->max_outer_extent = 700;
  this->ndim = 2;
  this->axis = 2;
  this->njoin = 4;
  this->new_axis = true;
  this->TestRawKernel();
}

TYPED_TEST(TensorJoinGPUTest, RawStackInterleave7) {
  this->max_outer_extent = 500;
  this->ndim = 2;
  this->axis = 2;
  this->njoin = 7;
  this->new_axis = true;
  this->TestRawKernel();
}

TYPED_TEST(TensorJoinGPUTest, RawStackInterleave15) {
  this->max_outer_extent = 500;
  this->ndim = 2;
  this->axis = 2;
  this->njoin = 15;
  this->new_axis = true;
  this->TestRawKernel();
}

TYPED_TEST(TensorJoinGPUTest, RawStackInterleaveMany) {
  this->max_outer_extent = 100;
  this->ndim = 2;
  this->axis = 2;
  this->njoin = 1024;
  this->new_axis = true;
  this->TestRawKernel();
}


TYPED_TEST(TensorJoinGPUTest, ConcatOuter) {
  this->max_outer_extent = 1;
  this->max_inner_extent = 500;
  this->njoin = 7;
  this->axis = 0;
  this->ndim = 2;
  this->new_axis = false;
  this->TestFullKernel();
}


TYPED_TEST(TensorJoinGPUTest, ConcatMiddle) {
  this->max_outer_extent = 100;
  this->max_inner_extent = 100;
  this->njoin = 7;
  this->axis = 1;
  this->new_axis = false;
  this->TestFullKernel();
}


TYPED_TEST(TensorJoinGPUTest, ConcatInner) {
  this->max_outer_extent = 1;
  this->max_inner_extent = 400;
  this->njoin = 7;
  this->axis = this->ndim - 1;
  this->new_axis = false;
  this->TestFullKernel();
}


TYPED_TEST(TensorJoinGPUTest, ConcatInterleave) {
  this->max_outer_extent = 200;
  this->max_inner_extent = 1;
  this->njoin = 7;
  this->ndim = 2;
  this->axis = this->ndim - 1;
  this->new_axis = false;
  this->TestFullKernel();
}

TYPED_TEST(TensorJoinGPUTest, StackOuter) {
  this->max_outer_extent = 1;
  this->max_inner_extent = 100;
  this->njoin = 7;
  this->ndim = 2;
  this->axis = 0;
  this->new_axis = true;
  this->TestFullKernel();
}


TYPED_TEST(TensorJoinGPUTest, StackMiddle) {
  this->max_outer_extent = 400;
  this->max_inner_extent = 400;
  this->njoin = 9;
  this->ndim = 2;
  this->axis = 1;
  this->new_axis = true;
  this->TestFullKernel();
}

TYPED_TEST(TensorJoinGPUTest, StackInterleave) {
  this->max_outer_extent = 400;
  this->max_inner_extent = 400;
  this->njoin = 17;
  this->ndim = 2;
  this->axis = 2;
  this->new_axis = true;
  this->TestFullKernel();
}

}  // namespace kernels
}  // namespace dali
