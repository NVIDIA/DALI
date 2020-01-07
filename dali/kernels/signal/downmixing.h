// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_DOWNMIXING_H_
#define DALI_KERNELS_SIGNAL_DOWNMIXING_H_

#include <cassert>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace signal {

/**
 * @brief Downmix interleaved signals to a single channel.
 *
 * @param out               output buffer (single channel)
 * @param in                input buffer (interleaved multiple channels)
 * @param num_samples       number of samples in each channel
 * @param channels          number of input channels
 * @param weights           weights used for downmixing
 * @param normalize_weights if true, the weights are normalized so their sum is 1
 * @tparam Out output sample type - if integral, the intermediate floating point representation
 *         is stretched so that 0..1 or -1..1 range occupies the whole Out range.
 * @tparam In input sample type - if integral, it's normalized to 0..1 or -1..1 range
 * @tparam static_channels compile-time number of channels
 *
 * Downmix interleaved signals to a single channel, using the weights provided.
 * If `normalize_weights` is true, the weights are copied into intermediate buffer
 * and divided by their sum.
 *
 * @remarks The operation can be done in place if output and input are of the same type.
 */
template <int static_channels = -1, typename Out, typename In>
void DownmixChannels(
    Out *out, const In *in, int64_t samples, int channels,
    const float *weights, bool normalize_weights = false) {
  SmallVector<float, 8> normalized_weights;  // 8 channels should be enough for 7.1 audio
  static_assert(static_channels != 0, "Number of channels cannot be zero."
                                      "Use negative values to use run-time value");
  int actual_channels = static_channels < 0 ? channels : static_channels;
  assert(actual_channels == channels);
  assert(actual_channels > 0);
  if (normalize_weights) {
    double sum = 0;
    for (int i = 0; i < channels; i++)
      sum += weights[i];
    normalized_weights.resize(channels);
    for (int i = 0; i < channels; i++) {
      normalized_weights[i] = weights[i] / sum;
    }
    weights = normalized_weights.data();  // use this pointer now
  }
  for (int64_t o = 0, i = 0; o < samples; o++, i += channels) {
    float sum = ConvertNorm<float>(in[i]) * weights[0];
    for (int c = 1; c < channels; c++) {
      sum += ConvertNorm<float>(in[i + c]) * weights[c];
    }
    out[o] = ConvertSatNorm<Out>(sum);
  }
}

/**
 * @brief Downmix data to a single channel.
 *
 * @param out               output buffer (single channel)
 * @param in                input buffer (interleaved multiple channels)
 * @param num_samples       number of samples in each channel
 * @param channels          number of input channels
 * @param weights           weights used for downmixing
 * @param normalize_weights if true, the weights are normalized so their sum is 1
 * @tparam Out output sample type - if integral, the intermediate floating point representation
 *         is stretched so that 0..1 or -1..1 range occupies the whole Out range.
 * @tparam In input sample type - if integral, it's normalized to 0..1 or -1..1 range
 *
 * Downmix interleaved signals to a single channel, using the weights provided.
 * If `normalize_weights` is true, the weights are copied into intermediate buffer
 * and divided by their sum.
 *
 * @remarks The operation can be done in place if output and input are of the same type.
 */
template <typename Out, typename In>
void Downmix(
    Out *out, const In *in, int64_t samples, int channels,
    const float *weights, bool normalize_weights = false) {
  VALUE_SWITCH(channels, static_channels, (1, 2, 3, 4, 5, 6, 7, 8),
    (DownmixChannels<static_channels>(out, in, samples, static_channels,
                                      weights, normalize_weights);),
    (DownmixChannels(out, in, samples, channels, weights, normalize_weights);)
  );  // NOLINT
}

template <typename Out, typename In>
void Downmix(Out *out, const In *in, int64_t num_samples, int num_channels) {
  SmallVector<float, 8> weights;
  weights.resize(num_channels, 1.0f / num_channels);
  Downmix(out, in, num_samples, num_channels, weights.data());
}


template <typename Out, typename In>
void Downmix(span<Out> out, span<const In> in,
             const std::vector<float> &weights, bool normalize_weights = false) {
  int num_channels = weights.size();
  assert(in.size() % num_channels == 0);
  Downmix(out.data(), in.data(), in.size() / num_channels, weights, normalize_weights);
}


template <typename Out, typename In>
void Downmix(span<Out> out, span<const In> in, int num_channels) {
  assert(in.size() % num_channels == 0);
  Downmix(out.data(), in.data(), in.size() / num_channels, num_channels);
}

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DOWNMIXING_H_
