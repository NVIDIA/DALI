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
#include <string>
#include <utility>
#include <vector>
#include "dali/core/format.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/operators/generic/join.h"
#include "dali/kernels/common/join/tensor_join_cpu.h"
#include "dali/kernels/common/join/tensor_join_gpu.h"

namespace dali {

DALI_SCHEMA(Cat)
  .DocStr(R"(Joins the input tensors along an existing axis.

The shapes of the inputs must match in all dimensions except the concatenation axis.)")
  .AddOptionalArg<int>("axis", "Axis along which the input tensors are concatenated.", 0, false)
  .AddOptionalArg<string>("axis_name", R"(Name of the axis along which the tensors are concatenated.

This argument is mutually exclusive with ``axis``.
This argument requires that at least one input has a non-empty layout and that all non-empty
input layouts match.)", nullptr, false)
  .NumInput(1, 999)
  .NumOutput(1);

DALI_SCHEMA(Stack)
  .DocStr(R"(Joins the input tensors along a new axis.

The shapes of respective tensors in the inputs must match.)")
  .AddOptionalArg<int>("axis", R"(The axis in the output tensor along which the inputs are stacked.

The axis is inserted before a corresponding axis in the inputs. A value of 0 indicates that whole
tensors are stacked. Specifying ``axis`` equal to the number of dimensions in the inputs causes
the values from the inputs to be interleaved)", 0, false)
  .AddOptionalArg<string>("axis_name", R"(Name of the new axis to be inserted.

A one-character string that will denote the new axis in the output layout. The output layout will be
constructed by inserting that character into the input layout at the position indicated by ``axis``.
For example, specifying ``axis = 0`` and ``axis_name = "C"`` with input layout "HW" will yield
the output layout "CHW")", nullptr, false)
  .NumInput(1, 999)
  .NumOutput(1);

#define TENSOR_JOIN_TYPES (bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
                          uint64_t, int64_t, float16, float, double)

template <typename Backend, bool new_axis>
bool TensorJoin<Backend, new_axis>::SetupImpl(
        vector<OutputDesc> &outputs, const workspace_t<Backend> &ws) {
  int njoin = this->spec_.NumRegularInput();
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

  GetInputLayout(ws);
  SetupAxis();
  SetOutputLayout(ws);

  // Run over the inputs and store them in a vector
  TYPE_SWITCH(out_type, type2id, T, TENSOR_JOIN_TYPES, (
    SetupTyped<T>(output_shape, ws);
  ), (DALI_FAIL(make_string("The element type ", out_type, " is not supported."))));  // NOLINT
  return true;
}

template <typename Backend, bool new_axis>
template <typename T>
void TensorJoin<Backend, new_axis>::SetupTyped(
      TensorListShape<> &output_shape, const workspace_t<Backend> &ws) {
  auto &inputs = this->template inputs<T>();
  inputs.clear();
  int ninp = this->spec_.NumRegularInput();

  copy_idx_ = 0;
  for (int i = 0; i < ninp; i++) {
    auto tlv = view<T>(ws.template InputRef<Backend>(i));
    if (new_axis || tlv.num_elements() > 0) {  // when concatenating, we can skip empty inputs
      if (inputs.empty())
        copy_idx_ = i;
      else
        copy_idx_ = -1;
      inputs.push_back(std::move(tlv));
    }
  }

  // No non-empty inputs? Use the first one, even if it's empty.
  if (inputs.empty()) {
    inputs.push_back(view<T>(ws.template InputRef<Backend>(0)));
  }

  kernels::tensor_join::JoinedShape(output_shape, [&](int index) {
    return &inputs[index].shape;
  }, inputs.size(), axis_, new_axis);
  DALI_ENFORCE(axis_ < output_shape.sample_dim(), make_string("Invalid axis index: ", axis_));
}

template <typename Backend, bool new_axis>
void TensorJoin<Backend, new_axis>::GetInputLayout(const workspace_t<Backend> &ws) {
  input_layout_ = {};
  if (new_axis && !has_axis_name_)
    return;

  int ninp = this->spec_.NumRegularInput();
  for (int i = 0; i < ninp; i++) {
    auto &in = ws.template InputRef<Backend>(0);
    TensorLayout tl = in.GetLayout();
    if (!tl.empty()) {
        if (!input_layout_.empty())
          DALI_ENFORCE(input_layout_ == tl, make_string("All non-empty input layouts must match.\n"
            "Offending values: \"", input_layout_, "\" and \"", tl, "\""));

        input_layout_ = tl;
    }
  }
}

template <typename Backend, bool new_axis>
void TensorJoin<Backend, new_axis>::SetOutputLayout(const workspace_t<Backend> &ws) {
  if (new_axis) {
    output_layout_ = {};
    if (has_axis_name_) {
      if (input_layout_.empty() && ws.template InputRef<Backend>(0).shape().sample_dim() > 0) {
        DALI_FAIL("Specifying the new axis name with ``axis_name`` with non-scalar input requires "
            "a non-empty input layout.");
      }
      output_layout_ =
          input_layout_.first(axis_) +
          TensorLayout(&axis_name_arg_, 1) +
          input_layout_.sub(axis_);
    }
  } else {
    output_layout_ = input_layout_;
  }
}

template <typename Backend, bool new_axis>
void TensorJoin<Backend, new_axis>::SetupAxis() {
  // axis_name indicates the join axis for concatenation only;
  // for stacking, it's the name of the new axis
  if (has_axis_name_ && !new_axis) {
    axis_ = input_layout_.find(axis_name_arg_);
    DALI_ENFORCE(axis_ >= 0, make_string("``axis_name`` specifies an undefined axis '",
      axis_name_arg_, "' for layout \"", input_layout_, "\""));
  } else {
    axis_ = axis_arg_;  // this will be validated by the Setup routine
  }
}

template <typename Backend, bool new_axis>
void TensorJoin<Backend, new_axis>::RunImpl(workspace_t<Backend> &ws) {
  auto &out = ws.template OutputRef<Backend>(0);
  if (copy_idx_ >= 0) {
    // just one non-empty input - copy it to the output and return
    TensorListShape<> shape;
    if (new_axis)
      shape = out.shape();
    out.Copy(ws.template InputRef<Backend>(copy_idx_), ws.has_stream() ? ws.stream() : 0);
    if (new_axis)
      out.Resize(shape);
    out.SetLayout(output_layout_);
    return;
  }

  out.SetLayout(output_layout_);
  TYPE_SWITCH(auto type_id = out.type().id(), type2id, T, TENSOR_JOIN_TYPES, (
    RunTyped(view<T>(out), ws);
  ), (throw std::logic_error(make_string("Internal error: RunImpl encountered a type that "  // NOLINT
    "should have been rejected by Setup. Was Setup called?\nOffending type: ", type_id))
  ));  // NOLINT
}

template <typename Backend, bool new_axis>
template <typename T>
void TensorJoin<Backend, new_axis>::RunTyped(
    const TensorListView<Storage, T> &out, HostWorkspace &ws) {
  using Kernel = kernels::TensorJoinCPU<T, new_axis>;
  ThreadPool &tp = ws.GetThreadPool();
  int num_threads = tp.NumThreads();
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
      auto sample_in_tensors = make_span(&in_tensors[tid * njoin], njoin);
      auto sample_in_shapes = make_span(&in_shapes[tid * njoin], njoin);
      for (int t = 0; t < njoin; t++) {
        sample_in_tensors[t] = inputs[t][i];
        sample_in_shapes[t] = sample_in_tensors[t].shape;
      }
      kmgr_.Setup<Kernel>(tid, ctx, sample_in_shapes, axis_);
      kmgr_.Run<Kernel>(tid, tid, ctx, out[i], sample_in_tensors);
    }, volume(out.tensor_shape_span(i)));
  }
  tp.RunAll(true);
}

template <typename Backend, bool new_axis>
template <typename T>
void TensorJoin<Backend, new_axis>::RunTyped(
    const TensorListView<Storage, T> &out, DeviceWorkspace &ws) {
  using Kernel = kernels::TensorJoinGPU<T, new_axis>;
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();

  auto &inputs = this->template inputs<T>();

  kmgr_.Resize<Kernel>(1, 1);
  kmgr_.Setup<Kernel>(0, ctx, make_cspan(inputs), axis_);
  kmgr_.Run<Kernel>(0, 0, ctx, out, make_cspan(inputs));
}

template <typename Backend>
using ConcatOp = TensorJoin<Backend, false>;

template <typename Backend>
using StackOp = TensorJoin<Backend, true>;

DALI_REGISTER_OPERATOR(Cat, ConcatOp<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Cat, ConcatOp<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(Stack, StackOp<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Stack, StackOp<GPUBackend>, GPU);

}  // namespace dali
