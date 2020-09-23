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

#include "dali/operators/decoder/audio/audio_decoder_impl.h"
#include "dali/kernels/signal/downmixing.h"

namespace dali {

template <typename T>
span<char> as_raw_span(T *buffer, ptrdiff_t length) {
  return make_span(reinterpret_cast<char*>(buffer), length*sizeof(T));
}

TensorShape<> DecodedAudioShape(const AudioMetadata &meta, float target_sample_rate,
                                bool downmix) {
  bool should_resample = target_sample_rate > 0 && meta.sample_rate != target_sample_rate;
  bool should_downmix = meta.channels > 1 && downmix;
  int channels = meta.channels == 1 || downmix ? 1 : meta.channels;
  int64_t len = should_resample ? kernels::signal::resampling::resampled_length(
                                      meta.length, meta.sample_rate, target_sample_rate)
                                : meta.length;
  return downmix ? TensorShape<>{len} : TensorShape<>{len, channels};
}

template <typename T, typename DecoderType = int16_t>
void DecodeAudio(TensorView<StorageCPU, T, DynamicDimensions> audio, AudioDecoderBase &decoder,
                 const AudioMetadata &meta, kernels::signal::resampling::Resampler &resampler,
                 span<DecoderType> decode_scratch_mem,
                 span<float> resample_scratch_mem,
                 float target_sample_rate, bool downmix,
                 const char *audio_filepath) {  // audio_filepath for debug purposes
  DALI_ENFORCE(meta.sample_rate > 0, "Invalid sampling rate");
  bool should_resample = target_sample_rate > 0 && meta.sample_rate != target_sample_rate;
  bool should_downmix = meta.channels > 1 && downmix;
  int64_t num_samples = meta.length * meta.channels;

  if (!should_resample && !should_downmix && std::is_same<T, DecoderType>::value) {
    int64_t ret = decoder.Decode(as_raw_span(audio.data, num_samples));
    DALI_ENFORCE(ret == num_samples, make_string("Error decoding audio file ", audio_filepath));
    return;
  }

  DALI_ENFORCE(decode_scratch_mem.size() >= num_samples,
               make_string("Decoder scratch memory provided is not big enough. Got: ",
                           decode_scratch_mem.size(), ", need: ", num_samples));

  const int64_t out_channels = should_downmix ? 1 : meta.channels;
  if (should_resample) {
    int64_t req_resample_scratch = meta.length * out_channels;
    DALI_ENFORCE(resample_scratch_mem.size() >= req_resample_scratch,
                 make_string("Resample scratch memory provided is not big enough. Got: ",
                             resample_scratch_mem.size(), ", need: ", req_resample_scratch));
  }

  int64_t ret = decoder.Decode(as_raw_span(decode_scratch_mem.data(), num_samples));
  DALI_ENFORCE(ret == num_samples, make_string("Error decoding audio file ", audio_filepath));

  const int64_t in_len = meta.length * meta.channels;
  if (should_resample) {
    float *resample_in = resample_scratch_mem.data();
    if (should_downmix) {
      assert(resample_scratch_mem.size() == meta.length);
      kernels::signal::Downmix(resample_in, decode_scratch_mem.data(), meta.length, meta.channels);
    } else {  // just cast (resampling kernel expects float)
      assert(resample_scratch_mem.size() == in_len);
      for (int64_t ofs = 0; ofs < in_len; ofs++) {
        resample_in[ofs] = ConvertSatNorm<float>(decode_scratch_mem[ofs]);
      }
    }
    resampler.Resample(audio.data, 0, audio.shape[0], target_sample_rate, resample_in, meta.length,
                       meta.sample_rate, out_channels);
  } else if (should_downmix) {  // downmix only
    kernels::signal::Downmix(audio.data, decode_scratch_mem.data(), meta.length, meta.channels);
  } else {
    // convert or copy only
    for (int64_t ofs = 0; ofs < in_len; ofs++) {
      audio.data[ofs] = ConvertSatNorm<T>(decode_scratch_mem[ofs]);
    }
  }
}

#define DECLARE_IMPL(OutType, DecoderType)                                                  \
  template void DecodeAudio<OutType, DecoderType>(                                          \
      TensorView<StorageCPU, OutType, DynamicDimensions> audio, AudioDecoderBase & decoder, \
      const AudioMetadata &meta, kernels::signal::resampling::Resampler &resampler,         \
      span<DecoderType> decode_scratch_mem, span<float> resample_scratch_mem,               \
      float target_sample_rate, bool downmix, const char *audio_filepath)

DECLARE_IMPL(float, int16_t);
DECLARE_IMPL(int16_t, int16_t);
DECLARE_IMPL(int32_t, int16_t);

#undef DECLARE_IMPL

}  // namespace dali
