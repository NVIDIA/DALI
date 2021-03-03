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

#ifndef DALI_OPERATORS_GENERIC_REDUCE_REDUCE_WITH_MEAN_INPUT_H__
#define DALI_OPERATORS_GENERIC_REDUCE_REDUCE_WITH_MEAN_INPUT_H__

#include <vector>
#include <algorithm>

#include "dali/pipeline/operator/operator.h"
#include "dali/operators/generic/reduce/reduce.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/reduce/reductions.h"
#include "dali/kernels/reduce/reduce_cpu.h"
#include "dali/kernels/reduce/reduce_gpu.h"
#include "dali/kernels/reduce/reduce_setup_utils.h"

#define REDUCE_WITH_MEAN_INPUT_TYPES ( \
  uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float)

namespace dali {
template <
  template <typename T, typename R, typename S> class ReductionType,
  typename Backend>
class ReduceWithMeanInput : public Operator<Backend>, detail::AxesHelper {
 public:
  explicit inline ReduceWithMeanInput(const OpSpec &spec) :
    Operator<Backend>(spec),
    AxesHelper(spec),
    keep_dims_(spec.GetArgument<bool>("keep_dims")),
    ddof_(spec.GetArgument<int>("ddof")) {
  }

  bool CanInferOutputs() const override { return true; }

  inline ~ReduceWithMeanInput() override = default;

  bool SetupImpl(
    std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    output_desc.resize(1);
    auto &input = ws.template InputRef<Backend>(0);

    output_desc[0].type = dali::TypeTable::GetTypeInfoFromStatic<float>();
    output_desc[0].shape = input.shape();

    PrepareAxes(input.GetLayout(), input.shape().sample_dim());

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
    DALIDataType output_type = DALI_FLOAT;

    TYPE_SWITCH(
      input_type,
      type2id,
      InputType,
      REDUCE_WITH_MEAN_INPUT_TYPES,
      (this->template RunTyped<float, InputType>(ws);),
      (DALI_FAIL(make_string("Unsupported input type: ", input_type));))
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
    int num_threads = thread_pool.NumThreads();

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
            thread_id,
            ctx,
            out_sample_view,
            in_sample_view,
            make_cspan(axes_),
            mean_sample_view,
            ddof_);
          if (!has_empty_axes_arg_) {
            kmgr_.Run<Kernel>(thread_id, thread_id, ctx);
          } else {
            OutputType *data = out_sample_view.data;
            std::fill(data, data + out_sample_view.num_elements(), 0);
          }
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
    if (!has_empty_axes_arg_) {
      kmgr_.Run<Kernel>(0, 0, ctx, out_view, in_view, mean_view, ddof_);
    } else {
      for (int i = 0; i < out_view.num_samples(); ++i) {
        auto out_sample_view = out_view[i];
        CUDA_CALL(cudaMemsetAsync(
          out_sample_view.data,
          0,
          out_sample_view.num_elements()*sizeof(OutputType),
          ws.stream()));
      }
    }
  }

 private:
  USE_OPERATOR_MEMBERS();

  bool keep_dims_;
  int ddof_;
  kernels::KernelManager kmgr_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_REDUCE_REDUCE_WITH_MEAN_INPUT_H_
