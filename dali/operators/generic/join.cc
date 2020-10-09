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

#include <cassert>
#include <utility>
#include "dali/core/format.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/operators/generic/join.h"
#include "dali/kernels/common/join/tensor_join_cpu.h"
#include "dali/kernels/common/join/tensor_join_gpu.h"

namespace dali {

#define TENSOR_JOIN_TYPES (bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
                          uint64_t, int64_t, float16, float, double)

template <typename Backend, bool new_axis>
bool TensorJoin<Backend, new_axis>::SetupImpl(
        vector<OutputDesc> &outputs, const workspace_t<Backend> &ws) {
  int njoin = ws.NumRegularInput();
  outputs.resize(1);
  outputs[0].type = ws.template InputRef<Backend>(0).type();
  auto &output_shape = outputs[0].shape;

  // Check that all inputs have the same type
  DALIDataType out_type = outputs[0].type.id();
  for (int i = 1; i < njoin; i++) {
    DALIDataType type_id = ws.template InputRef<Backend>(i).type().id();
    DALI_ENFORCE(type_id == out_type, make_string(
        "All inputs must have the same type.\nType of input #0: ", out_type,
        "\nType of input #", i, ": ", type_id));
  }

  // Get the join axis index
  axis_ = this->spec_.template GetArgument<int>("axis");

  // Run ove inputs and store them in a vector
  TYPE_SWITCH(out_type, type2id, T, TENSOR_JOIN_TYPES, (
    SetupTyped<T>(output_shape, ws);
  ), (DALI_FAIL(make_string("The element type ", out_type, " is not supported."))));  // NOLINT
}

template <typename Backend, bool new_axis>
template <typename T>
void TensorJoin<Backend, new_axis>::SetupTyped(
      TensorListShape<> &output_shape, const workspace_t<Backend> &ws) {
  auto &inputs = this->template inputs<T>();
  inputs.clear();
  int njoin = ws.NumRegularInput();
  for (int i = 0; i < njoin; i++) {
    auto tlv = view<T>(ws.template InputRef<Backend>(i));
    if (tlv.num_elements() > 0) {
      inputs.push_back(std::move(tlv));
    }
  }

  JoinedShape(output_shape, [&](int index) {
    return &inputs[index].shape;
  }, axis_);
  DALI_ENFORCE(axis_ < output_shape.sample_dim(), make_string("Invalid axis index: ", axis_));
}

template <typename Backend, bool new_axis>
TensorLayout TensorJoin<Backend, new_axis>::GetLayout(const workspace_t<Backend> &ws) {
  if (new_axis)
    return {};
  int njoin = ws.NumRegularInput();
  for (int i = 0; i < njoin; i++) {
    auto &in = ws.template InputRef<Backend>(0).GetLayout();
    TensorLayout tl = in.GetLayout();
    if (!tl.empty())
        return tl;
  }
  return {};
}


template <typename Backend, bool new_axis>
void TensorJoin<Backend, new_axis>::RunImpl(workspace_t<Backend> &ws) {
  auto &out = ws.template OutputRef<Backend>(0);
  out.SetLayout(GetLayout(ws));
  TYPE_SWITCH(out.type().id(), type2id, T, TENSOR_JOIN_TYPES, (
    RunTyped(view<T>(out, ws));
  ), (DALI_FAIL("Internal error: unsupported type reached RunImpl function")));  // NOLINT
}

template <typename Backend, bool new_axis>
template <typename T>
void TensorJoin<Backend, new_axis>::RunTyped(
    const TensorListView<Storage, T> &out, HostWorkspace &ws) {
  using Kernel = kernels::TensorJoinCPU<T, new_axis>;
  ThreadPool &tp = ws.GetThreadPool();
  int num_threads = tp.size();
  kmgr_.Resize<Kernel>(num_threads, num_threads);
  auto &inputs = this->template inputs<T>();
  SmallVector<kernels::InTensorCPU<T>, 128> in_tensors;
  SmallVector<TensorShape<>, 128> in_shapes;

  int N = out.num_samples();
  int njoin = inputs.size();
  in_tensors.resize(num_threads * njoin);
  in_shapes.resize(num_threads * njoin);

  for (int i = 0; i < N; i++) {
    tp.AddWork([&, i](int tid) {
      kernels::KernelContext ctx;
      auto sample_in_tensors = make_cspan(&in_tensors[tid * njoin], njoin);
      auto sample_in_shapes = make_cspan(&in_shapes[tid * njoin], njoin);
      for (int t = 0; t < njoin; t++) {
        sample_in_tensors[t] = inputs[t][i];
        sample_in_shapes[t] = sample_in_tensors[t].shape;
      }
      kmgr_.Setup<Kernel>(ctx, sample_in_shapes, axis_);
      kmgr_.Run<Kernel>(ctx, out[i], sample_in_tensors);
    }, volume(out.tensor_shape_span[i]));
  }
}

template <typename Backend, bool new_axis>
template <typename T>
void TensorJoin<Backend, new_axis>::RunTyped(
    const TensorListView<Storage, T> &out, DeviceWorkspace &ws) {
  using Kernel = kernels::TensorJoinGPU<T, new_axis>;
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();

  auto &inputs = this->template inputs<T>();

  kmgr_.Resize<Kernel>(1);
  kmgr_.Setup(ctx, make_span(inputs), axis_);
  kmgr_.Run(ctx, out, make_span(inputs));
}


}  // namespace dali
