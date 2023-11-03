// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"

#define NONSILENCE_TYPES uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float  // NOLINT

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
std::pair<int64_t, int64_t> LeadTrailThresh(span<const T> buffer, T cutoff) {
  assert(buffer.size() > 0);
  int64_t end = buffer.size();
  int64_t begin = end;
  for (int64_t i = 0; i < end; i++) {
    if (buffer[i] >= cutoff) {
      begin = i;
      break;
    }
  }
  if (begin == end) return {0, 0};  // Rest is silence
  for (int64_t i = end - 1; i >= begin; i--) {
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
  out.Resize(reqs.output_shapes[0][0], DALI_FLOAT);
  auto tv = view<float>(out);
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
  auto signal_mms = view<const float>(intermediate_buffer);
  kernels::signal::DecibelToMagnitude<float> db2mag(
      10.f, args.reference_max ? max_element(signal_mms) : args.reference_power);

  auto ret = LeadTrailThresh(make_cspan(signal_mms.data, signal_mms.num_elements()),
                             db2mag(args.cutoff_db));
  // If the buffer is not silent, add window length to the actual result,
  // since we don't know where in the window the non-silent signal is
  if (ret.first != 0 && ret.second != 0) {
    int new_start = std::max<int>(ret.first - (args.window_length - 1), 0);
    ret.second += ret.first - new_start;
    ret.first = new_start;
  }
  return ret;
}

}  // namespace detail

template<typename Backend>
class NonsilenceOperator : public StatelessOperator<Backend> {
 public:
  ~NonsilenceOperator() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperator);

 protected:
  explicit NonsilenceOperator(const OpSpec &spec) :
          StatelessOperator<Backend>(spec) {}


  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    AcquireArgs(spec_, ws);
    TensorShape<> scalar_shape = {};
    auto curr_batch_size = ws.GetInputBatchSize(0);

    output_desc.resize(detail::kNumOutputs);
    for (int i = 0; i < detail::kNumOutputs; i++) {
      output_desc[i].shape = uniform_list_shape(curr_batch_size, scalar_shape);
      output_desc[i].type = DALI_INT32;
    }
    return true;
  }

  void AcquireArgs(const OpSpec &spec, const Workspace &ws) {
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
    auto input_type = ws.Input<Backend>(0).type();
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

}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_
