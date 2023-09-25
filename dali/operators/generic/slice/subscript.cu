// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "dali/operators/generic/slice/subscript.h"
#include "dali/kernels/common/type_erasure.h"
#include "dali/kernels/slice/slice_gpu.cuh"

namespace dali {

template <>
template <int ndim, int element_size>
void TensorSubscript<GPUBackend>::RunTyped(Workspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  int N = input.num_samples();
  using T = kernels::type_of_size<element_size>;
  using Kernel = kernels::SliceGPU<T, T, ndim>;
  kmgr_.Resize<Kernel>(1);

  struct Ctx {
    TensorListView<StorageGPU, const T, ndim> tmp_in;
    TensorListView<StorageGPU, T, ndim> tmp_out;
    vector<kernels::SliceArgs<T, ndim>> args;
  };
  Ctx *ctx = std::any_cast<Ctx>(&ctx_);
  if (!ctx) {
    ctx_ = Ctx();
    ctx = &std::any_cast<Ctx&>(ctx_);
  }

  ctx->tmp_in.resize(N);
  ctx->tmp_out.resize(N);
  ctx->args.resize(N);
  ctx->tmp_in.shape = simplified_in_shape_.to_static<ndim>();
  ctx->tmp_out.shape = simplified_out_shape_.to_static<ndim>();
  for (int i = 0; i < N; i++) {
    ctx->tmp_in.data[i] = static_cast<const T *>(input.raw_tensor(i));
    ctx->tmp_out.data[i] = static_cast<T *>(output.raw_mutable_tensor(i));
    ctx->args[i].shape = ctx->tmp_out.shape[i];
    ctx->args[i].anchor = simplified_anchor_[i];
    ctx->args[i].step = simplified_step_[i];
  }

  kernels::KernelContext kctx;
  kctx.gpu.stream = ws.stream();
  kmgr_.Setup<Kernel>(0, kctx, ctx->tmp_in, ctx->args);
  kmgr_.Run<Kernel>(0, kctx, ctx->tmp_out, ctx->tmp_in, ctx->args);
}

DALI_REGISTER_OPERATOR(TensorSubscript, TensorSubscript<GPUBackend>, GPU);

}  // namespace dali
