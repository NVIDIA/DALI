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

template<typename InputType>
struct Args {
  TensorView<StorageCPU, const InputType, 1> input;
  float cutoff_db;
  float reference_power;
  bool reference_max;
  int window_length;
  int reset_interval;
};


/**
 * If the buffer is not silent, add window length to the actual result,
 * since we don't know where in the window the non-silent signal is
 */
void extend_nonsilent_range(std::pair<int, int> &thresholding_result, int window_length) {
  if (thresholding_result.second != 0) {
    thresholding_result.second += window_length - 1;
  }
}


template<typename T, int ndims>
T max_element(const TensorView<StorageCPU, const T, ndims> &tv) {
  T max = tv.data[0];
  for (int i = 1; i < tv.num_elements(); i++) {
    max = std::max(max, tv.data[i]);
  }
  return max;
}


/**
 * @brief Performs leading and trailing thresholding.
 *
 * Returns index of a first value above the threshold in the buffer and
 * the length up to the point, where last value above the threshold appears in the buffer.
 *
 * If the length == 0, begin == 0
 *
 * Example:
 * buffer: [0, 0, 0, 0, 50, 50, 0, 0]
 * cutoff: 20
 * return: {4, 2}
 *
 * @return (begin_idx, length)
 */
template<typename T>
std::pair<int, int> LeadTrailThresh(span<const T> buffer, T cutoff) {
  assert(buffer.size() > 0);
  int end = buffer.size();
  int begin = end;
  for (int i = 0; i < end; i++) {
    if (buffer[i] >= cutoff) {
      begin = i;
      break;
    }
  }
  if (begin == end) return {0, 0};  // Rest is silence
  for (int i = end - 1; i >= begin; i--) {
    if (buffer[i] >= cutoff) {
      end = i;
      break;
    }
  }
  return {begin, end - begin + 1};
}


template<typename InputType>
void RunKernel(TensorView<StorageCPU, const InputType, 1> in, Tensor<CPUBackend> &out,
               const kernels::signal::MovingMeanSquareArgs &mms_args) {
  kernels::KernelContext kctx;
  kernels::signal::MovingMeanSquareCpu<InputType> mms;
  auto reqs = mms.Setup(kctx, in, mms_args);
  out.Resize(reqs.output_shapes[0][0]);
  auto tv = view_as_tensor<float>(out);
  mms.Run(kctx, tv.template to_static<1>(), in, mms_args);
}


/**
 * Performs nonsilent region detection for single sample
 * @return (begin_idx, length)
 */
template<typename InputType>
std::pair<int, int>
DetectNonsilenceRegion(Tensor<CPUBackend> &intermediate_buffer, const Args<InputType> &args) {
  RunKernel(args.input, intermediate_buffer, {args.window_length, args.reset_interval});
  auto signal_mms = view_as_tensor<const float>(intermediate_buffer);
  kernels::signal::DecibelToMagnitude<float> db2mag(
      10.f, args.reference_max ? max_element(signal_mms) : args.reference_power);
  auto ret = LeadTrailThresh(make_cspan(signal_mms.data, signal_mms.num_elements()),
                             db2mag(args.cutoff_db));
  extend_nonsilent_range(ret, args.window_length);
  return ret;
}

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
    auto curr_batch_size = ws.GetInputBatchSize(0);
    this->GetPerSampleArgument(cutoff_db_, "cutoff_db", ws, curr_batch_size);
    if (spec.HasArgument("reference_power")) {
      this->GetPerSampleArgument(reference_power_, "reference_power", ws, curr_batch_size);
      for (const auto &val : reference_power_) {
        DALI_ENFORCE(val > 0, make_string("`reference_power` has to be positive. Got: ", val));
      }
    } else {
      reference_max_ = true;
    }
    window_length_ = spec.GetArgument<int>("window_length", &ws);
    auto input_type = ws.template InputRef<Backend>(0).type().id();
    // If input type is not floating point, there's no need for reset interval
    reset_interval_ = IsFloatingPoint(input_type) ? spec.GetArgument<int>("reset_interval", &ws)
                                                  : -1;

    DALI_ENFORCE(reset_interval_ == -1 || reset_interval_ % window_length_ == 0,
                 make_string("`reset_interval` shall be a multiple of `window_length`. "
                             "Got: reset_interval: ", reset_interval_, " vs window_length: ",
                             window_length_));
  }


  std::vector<float> cutoff_db_;
  std::vector<float> reference_power_;
  bool reference_max_ = false;
  int window_length_ = -1;
  int reset_interval_ = -1;

  USE_OPERATOR_MEMBERS();
};


class NonsilenceOperatorCpu : public NonsilenceOperator<CPUBackend> {
 public:
  explicit NonsilenceOperatorCpu(const OpSpec &spec) :
          NonsilenceOperator<CPUBackend>(spec) {
    intermediate_buffers_.resize(num_threads_);
    for (auto &b : intermediate_buffers_) {
      b.set_pinned(false);
    }
  }


  ~NonsilenceOperatorCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperatorCpu);

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<CPUBackend> &ws) override;

  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  template<typename InputType>
  void RunImplTyped(workspace_t<CPUBackend> &ws) {
    const auto &input = ws.template InputRef<CPUBackend>(0);
    auto &output_begin = ws.OutputRef<CPUBackend>(0);
    auto &output_length = ws.OutputRef<CPUBackend>(1);
    auto curr_batch_size = ws.GetInputBatchSize(0);
    auto &tp = ws.GetThreadPool();
    auto in_shape = input.shape();
    for (int sample_id = 0; sample_id < curr_batch_size; sample_id++) {
      tp.AddWork(
              [&, sample_id](int thread_id) {
                  detail::Args<InputType> args;
                  args.input = view<const InputType, 1>(input[sample_id]);
                  args.cutoff_db = cutoff_db_[sample_id];
                  if (!reference_max_) {
                    args.reference_power = reference_power_[sample_id];
                  }
                  args.reference_max = reference_max_;
                  args.window_length = window_length_ < args.input.num_elements() ?
                                                        window_length_ : args.input.num_elements();
                  args.reset_interval = reset_interval_;

                  auto res = DetectNonsilenceRegion(intermediate_buffers_[thread_id], args);
                  auto beg_ptr = output_begin[sample_id].mutable_data<int>();
                  auto len_ptr = output_length[sample_id].mutable_data<int>();
                  *beg_ptr = res.first;
                  *len_ptr = res.second;
              }, in_shape.tensor_size(sample_id));
    }
    tp.RunAll();
  }


  std::vector<Tensor<CPUBackend>> intermediate_buffers_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_
