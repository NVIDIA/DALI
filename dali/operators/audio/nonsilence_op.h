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

#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "dali/core/convert.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/signal/decibel/decibel_calculator.h"
#include "dali/kernels/signal/moving_mean_square.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace detail {

const int kNumOutputs = 2;

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


  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws) {
    this->GetPerSampleArgument(cutoff_db_, "cutoff_db", ws);
    this->GetPerSampleArgument(reference_db_, "reference_db", ws);
    this->GetPerSampleArgument(reference_max_, "reference_max", ws);
    this->GetPerSampleArgument(window_length_, "window_length", ws);
    this->GetPerSampleArgument(reset_interval_, "reset_interval", ws);
    auto input_type = ws.template InputRef<Backend>(0).type().id();
    if (!IsFloatingPoint(input_type)) {
      // If input type is integral, no need for reset interval
      reset_interval_.assign(reset_interval_.size(), -1);
    }
  }


  std::vector<float> cutoff_db_;
  std::vector<float> reference_db_;
  std::vector<bool> reference_max_;
  std::vector<int> window_length_;
  std::vector<int> reset_interval_;

  USE_OPERATOR_MEMBERS();
};


class NonsilenceOperatorCpu : public NonsilenceOperator<CPUBackend> {
 public:
  explicit NonsilenceOperatorCpu(const OpSpec &spec) :
          NonsilenceOperator<CPUBackend>(spec),
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
  using MmsArgs = kernels::signal::MovingMeanSquareArgs;

 public:
  Impl() = default;


  explicit Impl(const NonsilenceOperatorCpu *enclosing) : enclosing_(enclosing),
                                                          batch_size_(enclosing->batch_size_) {}


  template<typename InputType>
  struct Args {
    TensorView<StorageCPU, const InputType, 1> input;
    float cutoff_db;
    float reference_db;
    bool reference_max;
    int window_length;
    int reset_interval;
  };


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) {
    const auto &input = ws.template InputRef<CPUBackend>(0);
    nsamples_ = input.size();

    TypeInfo output_type;
    output_type.SetType<int>(TypeTable::GetTypeID<int>());
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

    for (int sample_id = 0; sample_id < batch_size_; sample_id++) {
      tp.DoWorkWithID(
              [&, sample_id](int thread_id) {
                  Args<InputType> args;
                  args.input = view<const InputType, 1>(input[sample_id]);
                  args.cutoff_db = -enclosing_->cutoff_db_[sample_id];
                  args.reference_db = enclosing_->reference_db_[sample_id];
                  args.reference_max = enclosing_->reference_max_[sample_id];
                  args.window_length = enclosing_->window_length_[sample_id];
                  args.reset_interval = enclosing_->reset_interval_[sample_id];

                  auto res = DetectNonsilenceRegion(sample_id, args);
                  auto beg_ptr = output_begin[sample_id].mutable_data<int>();
                  auto len_ptr = output_length[sample_id].mutable_data<int>();
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
  DetectNonsilenceRegion(int sample_id, const Args<InputType> &args) {
    SetupKernel();
    RunKernel(sample_id, args.input, {args.window_length, args.reset_interval});
    auto dbs = view_as_tensor<float>(intermediate_buffers_[sample_id]);
    kernels::signal::DecibelCalculator<float> dbc(10.f, args.reference_max ? max(dbs)
                                                                           : args.reference_db);
    return LeadTrailThresh(make_cspan(dbs.data, dbs.num_elements()), dbc.db2signal(args.cutoff_db));
  }


  void SetupKernel() {
    intermediate_buffers_.resize(nsamples_);
  }


  template<typename InputType>
  void
  RunKernel(int sample_id, TensorView<StorageCPU, const InputType, 1> in, const MmsArgs &mms_args) {
    kernels::KernelContext kctx;
    kernels::signal::MovingMeanSquareCpu<InputType> mms;
    auto reqs = mms.Setup(kctx, in, mms_args);
    intermediate_buffers_[sample_id].Resize(reqs.output_shapes[0][sample_id]);
    auto out = view_as_tensor<float>(intermediate_buffers_[sample_id]);
    mms.Run(kctx, out.template to_static<1>(), in, mms_args);
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


  template<typename T>
  T max(TensorView<StorageCPU, T, DynamicDimensions> tv) {
    T max = std::numeric_limits<T>::lowest();
    for (int i = 0; i < tv.num_elements(); i++) {
      max = std::max(max, tv.data[i]);
    }
    return max;
  }


  const NonsilenceOperatorCpu *enclosing_ = nullptr;
  int nsamples_ = -1, batch_size_ = -1;
  std::vector<Tensor<CPUBackend>> intermediate_buffers_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_
