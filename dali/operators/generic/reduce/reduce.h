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

#ifndef DALI_OPERATORS_GENERIC_REDUCE_REDUCE_H__
#define DALI_OPERATORS_GENERIC_REDUCE_REDUCE_H__

#include <vector>
#include <algorithm>

#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/reduce/reductions.h"
#include "dali/kernels/reduce/reduce_cpu.h"
#include "dali/kernels/reduce/reduce_gpu.h"
#include "dali/kernels/reduce/reduce_setup_utils.h"

#define REDUCE_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float)  // NOLINT

namespace dali {
template <
  template <typename T, typename R> class ReductionType,
  typename Backend,
  template <template <typename X, typename Y> class RType, typename BType> class ImplType>
class Reduce : public Operator<Backend> {
 public:
  explicit inline Reduce(const OpSpec &spec) :
    Operator<Backend>(spec),
    axes_(spec.GetRepeatedArgument<int>("axes")),
    keep_dims_(spec.GetArgument<bool>("keep_dims")) {
      if (!spec.TryGetArgument<DALIDataType>(output_type_, "dtype")) {
        output_type_ = DALI_NO_TYPE;
      }
  }

  bool CanInferOutputs() const override { return true; }

  inline ~Reduce() override = default;

  bool SetupImpl(
    std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    output_desc.resize(1);
    auto &input = ws.template InputRef<Backend>(0);

    int batch_size = input.shape().num_samples();

    if (output_type_ == DALI_NO_TYPE) {
      output_desc[0].type = input.type();
    } else {
      output_desc[0].type = dali::TypeTable::GetTypeInfo(output_type_);
    }

    output_desc[0].shape = input.shape();

    if (axes_.size() == 0) {
      axes_.resize(input.shape().sample_dim());
      std::iota(axes_.begin(), axes_.end(), 0);
    }

    TensorListShape<> output_shape;
    kernels::reduce_impl::CalculateReducedShape(
      output_shape,
      input.shape(),
      make_span(axes_),
      keep_dims_,
      false);
    output_desc[0].shape = output_shape;

    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    auto& reduce_impl = static_cast<ImplType<ReductionType, Backend>&>(*this);
    reduce_impl.RunImplImpl(ws);
  }

  template <typename OutputType, typename InputType>
  void RunTyped(HostWorkspace &ws) {
    auto& in = ws.InputRef<CPUBackend>(0);
    auto in_view = view<const InputType>(in);

    auto &out = ws.OutputRef<CPUBackend>(0);
    auto out_view = view<OutputType>(out);

    auto &thread_pool = ws.GetThreadPool();
    int num_threads = thread_pool.size();

    using Kernel = ReductionType<OutputType, InputType>;
    kmgr_.template Resize<Kernel>(num_threads, num_threads);

    for (int sample = 0; sample < in_view.num_samples(); sample++) {
      int64_t priority = volume(in_view.shape.tensor_shape_span(sample));
      thread_pool.AddWork(
        [&, sample](int thread_id) {
          auto in_sample_view = in_view[sample];
          auto out_sample_view = out_view[sample];
          kernels::KernelContext ctx;

          kmgr_.Setup<Kernel>(
            thread_id, ctx, out_sample_view, in_sample_view, make_cspan(axes_));
          kmgr_.Run<Kernel>(thread_id, thread_id, ctx);
        },
        priority);
    }
    thread_pool.RunAll();
  }

  template <typename OutputType, typename InputType>
  void RunTyped(DeviceWorkspace &ws) {
    auto& in = ws.InputRef<GPUBackend>(0);
    auto in_view = view<const InputType>(in);

    auto &out = ws.OutputRef<GPUBackend>(0);
    auto out_view = view<OutputType>(out);

    using Kernel = ReductionType<OutputType, InputType>;
    kmgr_.template Resize<Kernel>(1, 1);

    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();

    kmgr_.Setup<Kernel>(
      0,
      ctx,
      in_view.shape,
      make_cspan(axes_),
      keep_dims_,
      false);
    kmgr_.Run<Kernel>(0, 0, ctx, out_view, in_view);
  }

  DALIDataType OutputType() const { return output_type_; }

 private:
  USE_OPERATOR_MEMBERS();

  vector<int> axes_;
  bool keep_dims_;
  kernels::KernelManager kmgr_;
  DALIDataType output_type_ = DALI_NO_TYPE;
};


template <template <typename T, typename R> class ReductionType, typename Backend>
class ReduceOp : public Reduce<ReductionType, Backend, ReduceOp> {
 public:
  explicit inline ReduceOp(const OpSpec &spec) :  Reduce<ReductionType, Backend, ReduceOp>(spec) {}

  void RunImplImpl(workspace_t<Backend> &ws) {
    auto& in = ws.template InputRef<Backend>(0);
    DALIDataType input_type = in.type().id();

    TYPE_SWITCH(input_type, type2id, DataType, REDUCE_TYPES, (
      auto& base = static_cast<Reduce<ReductionType, Backend, ReduceOp>&>(*this);
      base.template RunTyped<DataType, DataType>(ws);),
      DALI_FAIL(make_string("Unsupported input type: ", input_type)))
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_REDUCE_REDUCE_H_
