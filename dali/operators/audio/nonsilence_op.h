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

#ifndef DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_
#define DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_

#include <utility>
#include <vector>
#include <dali/kernels/kernel_manager.h>
#include <gtest/gtest_prod.h>
#include <dali/pipeline/data/views.h>
#include <dali/kernels/signal/decibel/to_decibels_cpu.h>
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/signal/moving_mean_square.h"
#include "dali/operators/audio/nonsilence_op_impl.h"

namespace dali {
namespace detail {

const std::string kCutoff = "cutoff_value";  // NOLINT
const int kNumOutputs = 2;
using OutputType = int;
static_assert(std::is_integral<OutputType>::value,
              "Operator return indices, thus OutputType shall be integral");

}  // namespace detail

template<typename Backend>
class NonsilenceOperator : public Operator<Backend> {
 public:
  ~NonsilenceOperator() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperator);


 protected:
  explicit NonsilenceOperator(const OpSpec &spec) :
          Operator<Backend>(spec) {}


  bool CanInferOutputs() const override {
    return true;
  }


  USE_OPERATOR_MEMBERS();
};


class NonsilenceOperatorCpu : public NonsilenceOperator<CPUBackend> {
 public:
  explicit NonsilenceOperatorCpu(const OpSpec &spec) : NonsilenceOperator<CPUBackend>(spec),
                                                       impl_(std::make_unique<Impl>(this)) {}


  ~NonsilenceOperatorCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperatorCpu);

  class Impl;

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<CPUBackend> &ws) override;


  void RunImpl(workspace_t<CPUBackend> &ws) override;


 private:

  std::unique_ptr<Impl> impl_;
};


class DLL_PUBLIC NonsilenceOperatorCpu::Impl {
 private:
  using MmsKernel = kernels::signal::MovingMeanSquareCpu<float>;
  using ToDbKernel = kernels::signal::ToDecibelsCpu<float>;
  using MmsArgs = kernels::signal::MovingMeanSquareArgs;
  using DbArgs = kernels::signal::ToDecibelsArgs<float>;

 public:

  Impl() = default;


  explicit Impl(const NonsilenceOperatorCpu *super) : batch_size_(super->batch_size_) {}


  template<typename InputType, int ndims = 1>
  struct Args {
    TensorView<StorageCPU, const InputType, ndims> input;
    float cutoff_db;
    float reference_db;
    bool reference_max;
    int window_length;
    int reset_inteval;
  };


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) {
    const auto &input = ws.template InputRef<CPUBackend>(0);
    nthreads_ = ws.GetThreadPool().size();;
    nsamples_ = input.size();

    TypeInfo output_type;
    output_type.SetType<detail::OutputType>(TypeTable::GetTypeID<detail::OutputType>());
    TensorShape<> scalar_shape = {1};

    output_desc.resize(detail::kNumOutputs);
    for (int i = 0; i < detail::kNumOutputs; i++) {
      output_desc[i].shape = uniform_list_shape(batch_size_, scalar_shape);
      output_desc[i].type = output_type;
    }
    return true;
  }


  template<typename InputType>
  void RunImplTyped(workspace_t<CPUBackend> &ws) {

    const auto &input = ws.template InputRef<CPUBackend>(0);
    auto &output_begin = ws.OutputRef<CPUBackend>(0);
    auto &output_length = ws.OutputRef<CPUBackend>(1);
    auto &tp = ws.GetThreadPool();
//    auto nsamples = input.size();
//    auto nthreads = ws.GetThreadPool().size();

  for (int sample_id = 0; sample_id < batch_size_; sample_id++) {
    tp.DoWorkWithID(
            [&, sample_id](int thread_id) {
                Args<InputType> args;
                args.input = view<const InputType, 1>(input[sample_id]);
                args.cutoff_db = -20;
                args.reference_db = 1;
                args.reference_max = true;
                args.window_length = 2048;
                args.reset_inteval = -1;
                auto res = DetectNonsilenceRegion(thread_id, sample_id, args);
                auto beg_ptr = output_begin[sample_id].mutable_data<detail::OutputType>();
                auto len_ptr = output_length[sample_id].mutable_data<detail::OutputType>();
                *beg_ptr = res.first;
                *len_ptr = res.second;
            });
  }

  tp.WaitForWork();


  }


  /**
   * Performs nonsilent region detection for single sample
   * @return (begin_idx, length)
   */
  template<typename InputType>
  std::pair<int, int>
  DetectNonsilenceRegion(int thread_id, int sample_id, const Args<InputType> &args) {
    SetupKernels(nthreads_, nsamples_);
    DbArgs db_args;
    db_args.s_ref = args.reference_db;
    db_args.ref_max = args.reference_max;
    RunKernels(thread_id, sample_id, args.input, {args.window_length, args.reset_inteval}, db_args);
    auto dbs = view_as_tensor<float>(to_db_kernel_.outputs_[sample_id]);
    return LeadTrailThresh(make_cspan(dbs.data, dbs.num_elements()), args.cutoff_db);
  }


  /**
   * @brief Performs leading and trailing thresholding.
   *
   * Returns index of a first value above the threshold in the buffer and
   * the length up to the point, where last value above the threshold appears in the buffer.
   *
   * Example:
   * buffer: [0, 0, 0, 0, 50, 50, 0, 0]
   * cutoff: 20
   * return: {4, 2}
   *
   * @return (begin_idx, length)
   */
  template<typename T>
  static std::pair<int, int> LeadTrailThresh(span<const T> buffer, T cutoff) {
    assert(buffer.size() > 0);
    int begin = -1;
    int end = buffer.size();
    while (begin < end && buffer[++begin] < cutoff);  // NOLINT
    if (begin == end) return {-1, 0};
    while (buffer[--end] < cutoff);  // NOLINT
    return {begin, end - begin + 1};
  }


  void SetupKernels(int nthreads, int nsamples) {
    mms_kernel_.Setup(nthreads, nsamples);
    to_db_kernel_.Setup(nthreads, nsamples);
  }


  template<typename InputType>
  void RunKernels(int thread_id, int sample_id, TensorView<StorageCPU, const InputType, 1> in,
                  const MmsArgs &mms_args, const DbArgs &db_args) {
    mms_kernel_.Run(thread_id, sample_id, in, mms_args);
    auto db_in = view_as_tensor<const float>(mms_kernel_.outputs_[sample_id]).to_static<1>();
    to_db_kernel_.Run(thread_id, sample_id, db_in, db_args);
  }


  int nthreads_ = -1, nsamples_ = -1, batch_size_ = -1;


  /**
   * Wrapper for KernelManagers' boilerplate code, made-to-measure for Nonsilence operator
   */
  template<typename Kernel>
  struct YetAnotherKernelManager {
    kernels::KernelManager manager_;
    std::vector<Tensor<CPUBackend>> outputs_;


    void Setup(int nthreads, int nsamples) {
      manager_.Resize<Kernel>(nthreads, nsamples);
      outputs_.resize(nsamples);
    }


    template<typename InputType, typename Args>
    void Run(int thread_id, int sample_id, TensorView<StorageCPU, const InputType, 1> in,
             const Args &args) {
      kernels::KernelContext kctx;
      auto reqs = manager_.Setup<Kernel>(sample_id, kctx, in, args);
      outputs_[sample_id].Resize(reqs.output_shapes[0][sample_id]);
      auto out = view_as_tensor<float>(outputs_[sample_id]);
      manager_.Run<Kernel>(thread_id, sample_id, kctx, out.template to_static<1>(), in, args);
    }
  };

  YetAnotherKernelManager<MmsKernel> mms_kernel_;
  YetAnotherKernelManager<ToDbKernel> to_db_kernel_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_
