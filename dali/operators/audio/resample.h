// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_AUDIO_RESAMPLE_H_
#define DALI_OPERATORS_AUDIO_RESAMPLE_H_

#include <vector>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/kernels/signal/resampling.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace audio {

#define AUDIO_RESAMPLE_TYPES int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float

template <typename Backend>
class ResampleBase : public Operator<Backend> {
 public:
  explicit ResampleBase(const OpSpec &spec) : Operator<Backend>(spec) {
    DALI_ENFORCE(in_rate_.HasValue() == out_rate_.HasValue(),
      "The parameters ``in_rate`` and ``out_rate`` must be specified together.");
    if (in_rate_.HasValue() + scale_.HasValue() + out_length_.HasValue() > 1)
      DALI_FAIL("The sampling rates, ``scale`` and ``out_length`` cannot be used together.");
    if (!in_rate_.HasValue() && !scale_.HasValue() && !out_length_.HasValue())
      DALI_FAIL("No resampling factor specified! Please supply either the scale, "
        "the output length or the input and output sampling rates.");
    quality_ = spec_.template GetArgument<float>("quality");
    DALI_ENFORCE(quality_ >= 0 && quality_ <= 100, make_string("``quality`` out of range: ",
      quality_, "\nValid range is [0..100]."));
    if (spec_.TryGetArgument(dtype_, "dtype")) {
      // silence useless warning -----------------------------vvvvvvvvvvvvvvvv
      TYPE_SWITCH(dtype_, type2id, T, (AUDIO_RESAMPLE_TYPES), (T x; (void)x;),
      (DALI_FAIL(make_string("Unsupported output type: ", dtype_,
                           "\nSupported types are : ", ListTypeNames<AUDIO_RESAMPLE_TYPES>()))));
    }
  }

  bool CanInferOutputs() const override {
    return true;
  }

  virtual bool SetupImpl(std::vector<OutputDesc> &outputs, const workspace_t<Backend> &ws) {
    outputs.resize(1);
    if (dtype_ == DALI_NO_TYPE)
      dtype_ = ws.template Input<Backend>(0).type();

    outputs[0].type = dtype_;
    CalculateScaleAndShape(outputs[0].shape, ws);

    return true;
  }

  void CalculateScaleAndShape(TensorListShape<> &out_shape, const workspace_t<Backend> &ws) {
    const auto &input = ws.template Input<Backend>(0);
    const TensorListShape<> &shape = input.shape();
    DALI_ENFORCE(shape.sample_dim() == 1 || shape.sample_dim() == 2,
      "Audio resampling supports only time series data, with an optional innermost "
      "channel dimension.");
    out_shape = shape;
    int N = shape.num_samples();
    scales_.resize(N);
    if (in_rate_.HasValue()) {
      assert(out_rate_.HasValue());
      in_rate_.Acquire(spec_, ws, N);
      out_rate_.Acquire(spec_, ws, N);
      for (int s = 0; s < N; s++) {
        double in_rate = in_rate_[s].data[0];
        double out_rate = out_rate_[s].data[0];
        scales_[s] = out_rate / in_rate;
        int64_t in_length = shape.tensor_shape_span(s)[0];
        int64_t out_length =
          kernels::signal::resampling::resampled_length(in_length, in_rate, out_rate);
        out_shape.tensor_shape_span(s)[0] = out_length;
      }
    } else if (scale_.HasValue()) {
      scale_.Acquire(spec_, ws, N);
      for (int s = 0; s < N; s++) {
        double scale = scale_[s].data[0];
        scales_[s] = scale;
        int64_t in_length = shape.tensor_shape_span(s)[0];
        int64_t out_length =
          kernels::signal::resampling::resampled_length(in_length, 1, scale);
        out_shape.tensor_shape_span(s)[0] = out_length;
      }
    } else if (out_length_.HasValue()) {
      out_length_.Acquire(spec_, ws, N);
      for (int s = 0; s < N; s++) {
        int64_t in_length = shape.tensor_shape_span(s)[0];
        int64_t out_length = out_length_[s].data[0];
        scales_[s] = 1.0 * out_length / in_length;
        out_shape.tensor_shape_span(s)[0] = out_length;
      }
    } else {
      assert(!"Unreachable code - the constructor should have thrown.");
    }
  }

 protected:
  USE_OPERATOR_MEMBERS();

  DALIDataType dtype_ = DALI_NO_TYPE;
  float quality_ = 50;

  ArgValue<float> in_rate_{"in_rate", spec_};
  ArgValue<float> out_rate_{"out_rate", spec_};
  ArgValue<float> scale_{"scale", spec_};
  ArgValue<int64_t> out_length_{"out_length", spec_};

  std::vector<double> scales_;
};

}  // namespace audio
}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_RESAMPLE_H_
