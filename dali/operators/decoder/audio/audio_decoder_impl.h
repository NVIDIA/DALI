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

#include "dali/operators/decoder/audio/audio_decoder.h"
#include "dali/operators/decoder/audio/generic_decoder.h"
#include "dali/pipeline/data/backend.h"
#include "dali/kernels/signal/resampling.h"
#include "dali/core/tensor_view.h"

namespace dali {

TensorShape<> DecodedAudioShape(const AudioMetadata &meta, float target_sample_rate = -1,
                                bool downmix = true);

template <typename T, typename DecoderType>
void DecodeAudio(TensorView<StorageCPU, T, DynamicDimensions> audio, AudioDecoderBase &decoder,
                 const AudioMetadata &meta, kernels::signal::resampling::Resampler &resampler,
                 span<DecoderType> decode_scratch_mem, span<float> resample_scratch_mem,
                 float target_sample_rate, bool downmix, const char *audio_filepath);

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_IMPL_H_
