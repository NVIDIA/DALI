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

#ifndef DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_IMPL_H_
#define DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_IMPL_H_

#include <utility>
#include "dali/operators/decoder/audio/audio_decoder.h"
#include "dali/operators/decoder/audio/generic_decoder.h"
#include "dali/pipeline/data/backend.h"
#include "dali/kernels/signal/resampling.h"
#include "dali/core/tensor_view.h"

namespace dali {

/**
 * @brief Converts offset and length in seconds to offset and lenght in number of samples
 *        according with the audio metadata
 * @param meta Audio metadata
 * @param offset_sec offset, in seconds (optional)
 * @param length_sec length, in seconds. If a negative value is provided, whole buffer is assumed
 * @returns pair containing offset and length in number of samples
 */
DLL_PUBLIC std::pair<int64_t, int64_t> ProcessOffsetAndLength(const AudioMetadata &meta,
                                                              double offset_sec = 0,
                                                              double length_sec = -1);

/**
 * @brief Returns the shape of the resulting decoded audio
 * @param meta Audio metadata
 * @param target_sample_rate If a positive number is provided, it represent the target sampling rate
 *                           (the audio data is expected to be resampled if its original sampling rate differs)
 * @param downmix If set to true, the audio channels are expected to be downmixed, resulting in a shape with 1
 *                dimension ({nsamples,}), instead of 2 ({nsamples, nchannels})       
 */
DLL_PUBLIC TensorShape<> DecodedAudioShape(const AudioMetadata &meta, float target_sample_rate = -1,
                                           bool downmix = true);

/**
 * @brief Decodes audio data, with optional downmixing and resampling
 * @param audio Destination buffer. The function will decode as many audio samples as the shape of this argument
 * @param decoder Decoder object.
 * @param meta Audio metadata.
 * @param resampler Resampler instance used if resampling is required
 * @param decode_scratch_mem Scratch memory used for decoding, when decoding can't be done directly to the output buffer.
 *                           If downmixing or resampling is required, this buffer should have a positive length, representing
 *                           decoded audio length at the original sampling rate: ``length * nchannels``
 * @param resample_scratch_mem Scratch memory used for the input of resampling.
 *                             If resampling is required, the buffer should have a positive length, representing the 
 *                             decoded audio length, ``length`` if downmixing is enabled, or the decoded audio length including
 *                             channels, ``length * nchannels``, otherwise.
 * @param target_sample_rate If a positive value is provided, the signal will be resampled except when its original sampling rate
 *                           is equal to the target.
 * @param downmix If true, the audio channes will be downmixed to a single one
 * @param audio_filepath Path to the audio file being decoded, only used for debugging purposes 
 */
template <typename T>
DLL_PUBLIC void DecodeAudio(TensorView<StorageCPU, T, DynamicDimensions> audio,
                            AudioDecoderBase &decoder, const AudioMetadata &meta,
                            kernels::signal::resampling::Resampler &resampler,
                            span<float> decode_scratch_mem, span<float> resample_scratch_mem,
                            float target_sample_rate, bool downmix, const char *audio_filepath);

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_IMPL_H_
