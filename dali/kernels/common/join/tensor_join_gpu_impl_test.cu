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
#include "dali/kernels/common/join/tensor_join_gpu_impl.h"
#include "dali/kernels/common/join/tensor_join_gpu_impl.cuh"
#include "dali/core/dev_buffer.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {
namespace kernels {

template <typename T>
void RefTensorJoin(const OutTensorCPU<T> &out, span<const InTensorCPU<T>> in, int axis) {
  int njoin = in.size();

  SmallVector<ptrdiff_t, 8> copy_sizes;
  copy_sizes.resize(in.size());
  for (int t = 0; t < njoin; t++)
    copy_sizes[t] = volume(in[t].shape.begin() + axis, in[t].shape.end());

  ptrdiff_t nouter = volume(out.shape.begin(), out.shape.begin() + axis);
  T *dst = out.data;
  for (ptrdiff_t outer = 0; outer < nouter; outer++) {
    for (int t = 0; t < njoin; t++) {
      const T *src = in[t].data + outer * copy_sizes[t];
      for (ptrdiff_t inner = 0; inner < copy_sizes[t]; inner++) {
        *dst++ = src[inner];
      }
    }
  }
}

constexpr int joined_ndim(int ndim, bool newaxis) {
  return ndim < 0 ? ndim : ndim + newaxis;
}

template <int out_ndim, int ndim>
void JoinedShape(TensorListShape<out_ndim> &out,
                 span<const TensorListShape<ndim>> in, int axis, bool newaxis) {
  static_assert(out_ndim == ndim || (ndim >= 0 && out_ndim == ndim + 1));
  int njoin = in.size();
  if (njoin == 0) {
    out.resize(0, joined_ndim(in[0].sample_dim(), newaxis));
  }

  int N = in[0].num_samples();
  out.resize(N, joined_ndim(in[0].sample_dim(), newaxis));
  int d = in[0].sample_dim();

  int64_t in_volume = 0;
  for (auto &tls : in)
    in_volume += tls.num_elements();

  for (int i = 0; i < N; i++) {
    auto out_ts = out.tensor_shape_span(i);

    // copy outer extents, up to `axis`
    int oa = 0, ia = 0;  // input axis, output axis
    for (; ia < axis; ia++, oa++)
      out_ts[oa] = in[0].tensor_shape_span(i)[ia];

    if (newaxis) {
      out_ts[oa++] = njoin;  // new axis - number of joined tensor
    } else {
      // join along existing axis - sum the extents
      for (int t = 0; t < njoin; t++) {
        out_ts[oa] += in[t].tensor_shape_span(i)[ia];
      }
      oa++, ia++;  // advance both input and output
    }

    // copy remaining inner extents
    for (; ia < d; ia++, oa++)
      out_ts[oa] = in[0].tensor_shape_span(i)[ia];
  }

  assert(out.num_elements() == in_volume);
}

template <typename T>
void RefTLSJoin(const OutListCPU<T> &out, span<const InListCPU<T> *const> in, int axis) {
  SmallVector<InTensorCPU<T>, 8> in_tensors;
  int njoin = in.size();
  in_tensors.resize(njoin);

  int N = in[0]->num_samples();

  for (int i = 0; i < N; i++) {
    for (int t = 0; t < njoin; t++)
      in_tensors[t] = (*in[t])[i];
    RefTensorJoin(out[i], make_cspan(in_tensors), axis);
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
    Check(ref.cpu(), out.cpu());
  }

  void RunRawKernel() {
    vector<tensor_join::InputDesc<int>> input_descs(N * njoin);
    vector<tensor_join::OutputDesc<int>> output_descs(N);

    auto out_tls = out.gpu();

    for (int i = 0; i < N; i++) {
      int64_t join_offset = 0;
      for (int t = 0; t < njoin; t++) {
        auto tensor = in_gpu_tls[t][i];
        auto &desc = input_descs[i*njoin+t];
        desc.data = tensor.data;
        desc.outer_stride = volume(tensor.shape.begin() + axis, tensor.shape.end());
        desc.join_offset = join_offset;
        join_offset += desc.outer_stride;
      }

      auto out_tensor = out_tls[i];
      auto &out_desc = output_descs[i];
      out_desc.data = out_tensor.data;
      auto join_size = volume(out_tensor.shape.begin() + axis, out_tensor.shape.end());
      out_desc.outer_stride = join_size;
      out_desc.total_size = volume(out_tensor.num_elements());
      out_desc.guess_tensor_mul = 1.0 * njoin / join_size;
    }

    DeviceBuffer<tensor_join::InputDesc<int>> gpu_input_descs;
    DeviceBuffer<tensor_join::OutputDesc<int>> gpu_output_descs;
    gpu_input_descs.from_host(input_descs);
    gpu_output_descs.from_host(output_descs);

    dim3 grid(1024, N);
    dim3 block(256);
    JoinTensorsKernel<<<grid, block>>>(gpu_output_descs.data(), gpu_input_descs.data(), njoin);
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

    JoinedShape(out_shape, make_cspan(in_shapes), axis, new_axis);
  }


  int N = 1, ndim = 3, njoin = 7, axis = 1, max_outer_extent = 100, max_inner_extent = 100;
  bool new_axis = true;
  std::mt19937_64 rng;

  vector<TensorListShape<>> in_shapes;
  TensorListShape<> out_shape;
  vector<TestTensorList<int>> in;
  TestTensorList<int> out, ref;

  vector<InListCPU<int>> in_cpu_tls;
  vector<InListGPU<int>> in_gpu_tls;
  vector<const InListCPU<int> *> in_cpu_ptrs;
  vector<const InListGPU<int> *> in_gpu_ptrs;

};

TEST_F(TensorJoinGPUTest, Test_Small) {
  max_outer_extent = 10;
  max_inner_extent = 10;
  new_axis = false;
  TestRawKernel();
}

}  // namespace kernels
}  // namespace dali
