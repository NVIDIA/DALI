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

std::pair<int64_t, int64_t> ProcessOffsetAndLength(const AudioMetadata &meta,
                                                   double offset_sec, double length_sec) {
  int64_t offset = 0;
  int64_t length = meta.length;
  if (offset_sec >= 0.0) {
    offset = static_cast<int64_t>(offset_sec * meta.sample_rate);
  }

  if (length_sec >= 0.0) {
    length = static_cast<int64_t>(length_sec * meta.sample_rate);
  }

  // Limit the duration to the bounds of the input
  if ((offset + length) > meta.length) {
    length = meta.length - offset;
  }
  return {offset, length};
}

TensorShape<> DecodedAudioShape(const AudioMetadata &meta, float target_sample_rate, bool downmix) {
  bool should_resample = target_sample_rate > 0 && meta.sample_rate != target_sample_rate;
  bool should_downmix = meta.channels > 1 && downmix;
  int channels = meta.channels == 1 || downmix ? 1 : meta.channels;
  int64_t len = should_resample ? kernels::signal::resampling::resampled_length(
                                      meta.length, meta.sample_rate, target_sample_rate)
                                : meta.length;
  return downmix ? TensorShape<>{len} : TensorShape<>{len, channels};
}

template <typename T>
void DecodeAudio(TensorView<StorageCPU, T, DynamicDimensions> audio, AudioDecoderBase &decoder,
                 const AudioMetadata &meta, kernels::signal::resampling::Resampler &resampler,
                 span<float> decode_scratch_mem,
                 span<float> resample_scratch_mem,
                 float target_sample_rate, bool downmix,
                 const char *audio_filepath) {  // audio_filepath for debug purposes
  assert(meta.sample_rate > 0 && "Invalid sampling rate");
  bool should_resample = target_sample_rate > 0 && meta.sample_rate != target_sample_rate;
  bool should_downmix = meta.channels > 1 && downmix;
  assert(audio.data != nullptr);
  if (volume(audio.shape) <= 0)
    return;

  if (!should_resample && !should_downmix) {
    assert(audio.shape[0] <= meta.length && "Requested to decode more data than available.");
    assert(meta.channels == (audio.shape.size() == 1 ? 1 : audio.shape[1]) &&
           "Number of channels should match the metadata.");
    int64_t ret = decoder.DecodeFrames(audio.data, audio.shape[0]);
    DALI_ENFORCE(ret == audio.shape[0],
      make_string("Error decoding audio file ", audio_filepath, ". Requested ",
                  audio.shape[0], " samples but got ", ret, " samples."));
    return;
  }

  assert(decode_scratch_mem.size() > 0 &&
         "Dowmixing or resampling is required but decoder scratch memory is empty.");
  assert(decode_scratch_mem.size() % meta.channels == 0 &&
         "Expected to decode full audio frames only.");
  assert(decode_scratch_mem.size() <= meta.length * meta.channels &&
         "Requested to decode more data than available.");
  int64_t decoded_audio_len = decode_scratch_mem.size() / meta.channels;
  if (should_resample && should_downmix) {
    // When downmixing, we need an extra buffer for the input of resampling
    assert(resample_scratch_mem.size() == decoded_audio_len &&
           "Downmixing and resampling is required but resampler scratch is either empty or doesn't "
           "have the expected size");
  }

  int64_t ret = decoder.DecodeFrames(decode_scratch_mem.data(), decoded_audio_len);
  DALI_ENFORCE(ret == decoded_audio_len, make_string("Error decoding audio file ", audio_filepath));

  if (should_resample && should_downmix) {
    kernels::signal::Downmix(resample_scratch_mem.data(), decode_scratch_mem.data(),
                             decoded_audio_len, meta.channels);
    resampler.Resample(audio.data, 0, audio.shape[0], target_sample_rate,
                       resample_scratch_mem.data(), decoded_audio_len, meta.sample_rate, 1);
  } else if (should_resample) {  // No downmix
    resampler.Resample(audio.data, 0, audio.shape[0], target_sample_rate, decode_scratch_mem.data(),
                       decoded_audio_len, meta.sample_rate, meta.channels);
  } else if (should_downmix) {  // downmix only
    kernels::signal::Downmix(audio.data, decode_scratch_mem.data(), decoded_audio_len,
                             meta.channels);
  } else {
    assert(false && "Logic error. This should never happen.");
  }
}

#define DECLARE_IMPL(OutType)                                                                     \
  template void DecodeAudio<OutType>(                                                             \
      TensorView<StorageCPU, OutType, DynamicDimensions> audio, AudioDecoderBase & decoder,       \
      const AudioMetadata &meta, kernels::signal::resampling::Resampler &resampler,               \
      span<float> decode_scratch_mem, span<float> resample_scratch_mem,                           \
      float target_sample_rate, bool downmix, const char *audio_filepath);

DECLARE_IMPL(float);
DECLARE_IMPL(int16_t);
DECLARE_IMPL(int32_t);

#undef DECLARE_IMPL

}  // namespace dali
