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

#include <map>
#include <vector>
#include "dali/operators/audio/resample.h"
#include "dali/operators/audio/resampling_params.h"
#include "dali/kernels/kernel_params.h"

namespace dali {

DALI_SCHEMA(experimental__AudioResample)
  .DocStr(R"(Resamples a signal with a different sampling rate.

The resampling is achieved by applying a sinc filter with Hann window with an extent
controlled by the ``quality`` argument.

The resampling ratio can specified either directly or as a ratio of target to source sampling rate
or calculated from the ratio of requested output length to input length.
)")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg<float>("in_rate", R"(Input sampling rate.

The sampling rate of the input sample. This parameter must be specified together with ``out_rate``.
The value is relative to ``out_rate`` and doesn't need to use any specific unit as long as the
units of input and output rates match.

The ``in_rate`` and ``out_rate`` parameters cannot be specified together with ``scale`` or
``out_length``.)",
    nullptr, true)
  .AddOptionalArg<float>("out_rate", R"(Input sampling rate.

The sampling rate of the input sample. This parameter must be specified together with ``out_rate``.
The value is relative to ``out_rate`` and doesn't need to use any specific unit as long as the
units of input and output rates match.

The ``in_rate`` and ``out_rate`` parameters cannot be specified together with ``scale`` or
``out_length``.)",
    nullptr, true)
  .AddOptionalArg<float>("scale", R"(The scaling factor.

The scaling factor is the ratio of the target sampling rate to source sampling rate. For example,
a ``scale=2`` will produce an output with twice as many samples as the input.

This parameter cannot be specified together with ``in_rate`` and ``out_rate`` or ``out_length``.)",
    nullptr, true)
  .AddOptionalArg<int64_t>("out_length", R"(The requested output length, in samples.

The scaling factor is the ratio of this output length to the input length. This parameter
cannot be specified together with ``in_rate``, ``out_rate`` or ``scale``.)",
    nullptr, true)
  .AddOptionalArg("quality", R"(Resampling quality, where 0 is the lowest, and 100 is
the highest.

0 gives 3 lobes of the sinc filter, 50 gives 16 lobes, and 100 gives 64 lobes.)",
    50.0f, false)
  .AddOptionalArg<DALIDataType>("dtype", R"(The ouput type.

If not specified, the output type is the same as the input type. When the type is changed,
the values are normalized to fill the dynamic range of the target data type. When converting
floating point inputs to integer types, the values are assumed to be in -1..1 range. When converting
between signed and unsigned types, 0 translates to half-range of the unsigned type.)",
    nullptr, false);

namespace audio {

using kernels::InTensorCPU;
using kernels::OutTensorCPU;

class ResampleCPU : public ResampleBase<CPUBackend> {
 public:
  using Base = ResampleBase<CPUBackend>;
  explicit ResampleCPU(const OpSpec &spec) : Base(spec) {
    auto params = ResamplingParams::FromQuality(quality_);
    R.Initialize(params.lobes, params.lookup_size);
  }

  void RunImpl(HostWorkspace &ws) override {
    auto &out = ws.template Output<CPUBackend>(0);
    const auto &in = ws.template Input<CPUBackend>(0);

    const auto &out_shape = out.shape();
    const auto &in_shape = in.shape();
    out.SetLayout(in.GetLayout());
    int N = in.num_samples();
    assert(N == static_cast<int>(scales_.size()));
    assert(out.type() == dtype_);

    auto &tp = ws.GetThreadPool();
    in_fp32.resize(tp.NumThreads());
    for (int s = 0; s < N; s++) {
      tp.AddWork([&, this, s](int thread_idx) {
        InTensorCPU<float> in_view;
        TYPE_SWITCH(in.type(), type2id, T, (AUDIO_RESAMPLE_TYPES),
          (in_view = ConvertInput(in_fp32[thread_idx], view<const T>(in[s]));),
          (DALI_FAIL(
              make_string("Unsupported output type: ", dtype_,
                          "\nSupported types are : ", ListTypeNames<AUDIO_RESAMPLE_TYPES>()));));
        TYPE_SWITCH(dtype_, type2id, T, (AUDIO_RESAMPLE_TYPES),
          (ResampleTyped<T>(view<T>(out[s]), in_view, scales_[s]);),
          (assert(!"Unreachable code.")));
      });
    }
    tp.RunAll();
  }

  template <typename T>
  void ResampleTyped(const OutTensorCPU<T> &out, const InTensorCPU<float> &in, double scale) {
    int ch = out.shape.sample_dim() > 1 ? out.shape[1] : 1;
    R.Resample(out.data, 0, out.shape[0], scale, in.data, in.shape[0], 1, ch);
  }

  template <typename T>
  InTensorCPU<float> ConvertInput(std::vector<float> &tmp, const InTensorCPU<T> &in) {
    tmp.resize(in.num_elements());
    if (IsUnsigned(dtype_) && std::is_signed_v<T>) {
      // squeeze to upper half of the range
      for (size_t i = 0; i < tmp.size(); i++) {
        tmp[i] = (ConvertSatNorm<float>(in.data[i]) + 1.0f) * 0.5f;
      }
    } else if (IsSigned(dtype_) && std::is_unsigned_v<T>) {
      // treat half-range as 0
      for (size_t i = 0; i < tmp.size(); i++) {
        tmp[i] = ConvertSatNorm<float>(in.data[i]) * 2.0f - 1.0f;
      }
    } else {
      for (size_t i = 0; i < tmp.size(); i++) {
        tmp[i] = ConvertSatNorm<float>(in.data[i]);  // just normalize
      }
    }
    return make_tensor_cpu(tmp.data(), in.shape);
  }

  InTensorCPU<float> ConvertInput(std::vector<float> &tmp, const InTensorCPU<float> &in) {
    if (IsSigned(dtype_))
      return in;  // short-circuit

    tmp.resize(in.num_elements());
    for (size_t i = 0; i < tmp.size(); i++) {
      // squeeze to 0..1 range
      tmp[i] = (ConvertSatNorm<float>(in.data[i]) + 1.0f) * 0.5f;
    }
    return make_tensor_cpu(tmp.data(), in.shape);
  }

 private:
  kernels::signal::resampling::Resampler R;
  std::vector<std::vector<float>> in_fp32;
};


}  // namespace audio

DALI_REGISTER_OPERATOR(experimental__AudioResample, audio::ResampleCPU, CPU);

}  // namespace dali
