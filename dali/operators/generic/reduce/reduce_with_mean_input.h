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

#ifndef DALI_OPERATORS_GENERIC_REDUCE_WITH_MEAN_INPUT_H__
#define DALI_OPERATORS_GENERIC_REDUCE_WITH_MEAN_INPUT_H__

#include <vector>
#include <algorithm>

#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/reduce/reductions.h"
#include "dali/kernels/reduce/reduce_cpu.h"
#include "dali/kernels/reduce/reduce_gpu.h"
#include "dali/kernels/reduce/reduce_setup_utils.h"

#define REDUCE_WITH_MEAN_TYPES_MAP ( \
    ((uint8_t), (uint8_t, float)), \
    ((int8_t), (int8_t, float)), \
    ((uint16_t), (uint16_t, float)), \
    ((int16_t), (int16_t, float)), \
    ((uint32_t), (uint32_t, float)), \
    ((int32_t), (int32_t, float)), \
    ((uint64_t), (uint64_t, float)), \
    ((int64_t), (int64_t, float)), \
    ((float), (float)))

namespace dali {
template <
  template <typename T, typename R, typename S> class ReductionType,
  typename Backend>
class ReduceWithMeanInput : public Operator<Backend> {
 public:
  explicit inline ReduceWithMeanInput(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<DALIDataType>("dtype")),
    axes_(spec.GetRepeatedArgument<int>("axes")),
    keep_dims_(spec.GetArgument<bool>("keep_dims")),
    ddof_(spec.GetArgument<int>("ddof")) {
  }

  bool CanInferOutputs() const override { return true; }

  inline ~ReduceWithMeanInput() override = default;

  bool SetupImpl(
    std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    output_desc.resize(1);
    auto &input = ws.template InputRef<Backend>(0);

    output_desc[0].type = dali::TypeTable::GetTypeInfo(OutputType(input.type().id()));
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
    auto& in = ws.template InputRef<Backend>(0);
    DALIDataType input_type = in.type().id();
    DALIDataType output_type = this->OutputType(input_type);

    TYPE_MAP(
      input_type,
      output_type,
      type2id,
      InputType,
      OutputType,
      REDUCE_WITH_MEAN_TYPES_MAP,
      (this->template RunTyped<OutputType, InputType>(ws);),
      (DALI_FAIL(make_string("Unsupported input type: ", input_type));),
      (DALI_FAIL(make_string("Unsupported types: ", input_type, ", ", output_type));))
    
  }

  template <typename OutputType, typename InputType>
  void RunTyped(HostWorkspace &ws) {
    auto& in = ws.InputRef<CPUBackend>(0);
    auto in_view = view<const InputType>(in);

    auto& mean = ws.InputRef<CPUBackend>(1);
    auto mean_view = view<const OutputType>(mean);

    auto &out = ws.OutputRef<CPUBackend>(0);
    auto out_view = view<OutputType>(out);

    auto &thread_pool = ws.GetThreadPool();
    int num_threads = thread_pool.size();

    using Kernel = ReductionType<OutputType, InputType, OutputType>;
    kmgr_.template Resize<Kernel>(num_threads, num_threads);

    for (int sample = 0; sample < in_view.num_samples(); sample++) {
      int64_t priority = volume(in_view.shape.tensor_shape_span(sample));
      thread_pool.AddWork(
        [&, sample](int thread_id) {
          auto in_sample_view = in_view[sample];
          auto mean_sample_view = mean_view[sample];
          auto out_sample_view = out_view[sample];
          kernels::KernelContext ctx;

          kmgr_.Setup<Kernel>(
            thread_id, ctx, out_sample_view, in_sample_view, make_cspan(axes_), mean_sample_view, ddof_);
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

    auto& mean = ws.InputRef<GPUBackend>(1);
    auto mean_view = view<const OutputType>(mean);

    auto &out = ws.OutputRef<GPUBackend>(0);
    auto out_view = view<OutputType>(out);

    using Kernel = ReductionType<OutputType, InputType, OutputType>;
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
    kmgr_.Run<Kernel>(0, 0, ctx, out_view, in_view, mean_view, ddof_);
  }

  DALIDataType OutputType(DALIDataType input_type) const { 
    if (this->output_type_ != DALI_NO_TYPE) {
      return this->output_type_;
    }

    return DALI_FLOAT;
  }

  DALIDataType output_type_ = DALI_NO_TYPE;

 private:
  USE_OPERATOR_MEMBERS();

  vector<int> axes_;
  bool keep_dims_;
  int ddof_;
  kernels::KernelManager kmgr_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_REDUCE_WITH_MEAN_INPUT_H_
