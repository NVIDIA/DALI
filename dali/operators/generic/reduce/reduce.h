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

#ifndef DALI_OPERATORS_GENERIC_REDUCE_H_
#define DALI_OPERATORS_GENERIC_REDUCE_H_

#include <vector>
#include <algorithm>

#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/reduce/reductions.h"
#include "dali/kernels/reduce/reduce_cpu.h"

namespace dali {
#define REDUCE_TYPES (int16_t, int32_t, float)

template <template <typename T, typename R> class ReductionType>
class ReduceCPU : public Operator<CPUBackend> {
 public:
  explicit inline ReduceCPU(const OpSpec &spec) :
    Operator<CPUBackend>(spec),
    axes_(spec.GetRepeatedArgument<int>("axes")),
    keep_dims_(spec.GetArgument<bool>("keep_dims")) { }

  bool CanInferOutputs() const override { return true; }

  inline ~ReduceCPU() override = default;

  DISABLE_COPY_MOVE_ASSIGN(ReduceCPU);

 protected:
  bool SetupImpl(
    std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override {
    output_desc.resize(1);
    auto &input = ws.template InputRef<CPUBackend>(0);

    int batch_size = input.shape().num_samples();

    output_desc[0].type =  input.type();
    output_desc[0].shape = input.shape();

    if (axes_.size() == 0) {
      axes_.resize(input.shape().sample_dim());
      std::iota(axes_.begin(), axes_.end(), 0);
    }

    for (int sample = 0; sample < batch_size; ++sample) {
      for (int axis : axes_) {
        output_desc[0].shape.tensor_shape_span(sample)[axis] = 1;
      }
    }

    if (keep_dims_ == false) {
      vector<TensorShape<>> new_output_shape(batch_size);
      for (int sample = 0; sample < batch_size; ++sample) {
        TensorShape<> new_sample_shape;
        for (auto dim : output_desc[0].shape.tensor_shape_span(sample)) {
          if (dim != 1) {
            new_sample_shape.shape.push_back(dim);
          }
        }
        if (new_sample_shape.shape.size() == 0) {
          new_sample_shape.shape.push_back(1);
        }
        new_output_shape[sample] = new_sample_shape;
      }
      output_desc[0].shape = new_output_shape;
    }
    return true;
  }

  void RunImpl(workspace_t<CPUBackend> &ws) override {
    auto& in = ws.InputRef<CPUBackend>(0);
    DALIDataType data_type = in.type().id();

    TYPE_SWITCH(
      data_type, 
      type2id, 
      DataType,
      REDUCE_TYPES,
      ( RunTyped<DataType>(ws); ),
      ( DALI_FAIL(make_string("Unsupported input type: ", data_type)); )
    )
  }

 private:
  USE_OPERATOR_MEMBERS();

  vector<int> axes_;
  bool keep_dims_;

  template <typename DataType>
  void RunTyped(workspace_t<CPUBackend> &ws) {
    auto& in = ws.InputRef<CPUBackend>(0);
    auto in_view = view<const DataType>(in);

    auto &out = ws.OutputRef<CPUBackend>(0);
    auto out_view = view<DataType>(out);

    auto &thread_pool = ws.GetThreadPool();
    int num_threads = thread_pool.size();

    using Kernel = ReductionType<DataType, DataType>;
    vector<Kernel> kernels(num_threads);

    for (int sample = 0; sample < in_view.num_samples(); sample++) {      
      thread_pool.AddWork(
        [&, sample](int thread_id) {
          auto in_sample_view = in_view[sample];
          auto out_sample_view = out_view[sample];
 
          kernels[thread_id].Setup(
            out_sample_view,
            in_sample_view,
            make_span(axes_));
          kernels[thread_id].Run();
        });  
    }
    thread_pool.RunAll();
  }
};

class SumCPU : public ReduceCPU<kernels::SumCPU> {
 public:
  explicit inline SumCPU(const OpSpec &spec) :
    ReduceCPU<kernels::SumCPU>(spec) {}
};

class MinCPU : public ReduceCPU<kernels::MinCPU> {
 public:
  explicit inline MinCPU(const OpSpec &spec) :
    ReduceCPU<kernels::MinCPU>(spec) {}
};

class MaxCPU : public ReduceCPU<kernels::MaxCPU> {
 public:
  explicit inline MaxCPU(const OpSpec &spec) :
    ReduceCPU<kernels::MaxCPU>(spec) {}
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_REDUCE_H_
