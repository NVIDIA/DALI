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

#ifndef DALI_OPERATORS_DECODER_AUDIO_GENERIC_DECODER_H_
#define DALI_OPERATORS_DECODER_AUDIO_GENERIC_DECODER_H_

#include <sndfile.h>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>
#include "dali/core/format.h"
#include "dali/core/error_handling.h"
#include "dali/operators/decoder/audio/audio_decoder.h"


namespace dali {

/**
 * Generic decoder, that serves as a fallback to most of the formats we need.
 */
template<typename SampleType>
class DLL_PUBLIC GenericAudioDecoder : public TypedAudioDecoderBase<SampleType> {
 public:
  DLL_PUBLIC GenericAudioDecoder();

  DLL_PUBLIC ptrdiff_t DecodeTyped(span<SampleType> output) override;

  DLL_PUBLIC ~GenericAudioDecoder() override;

 private:
  AudioMetadata OpenImpl(span<const char> encoded) override;

  void CloseImpl() override;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_GENERIC_DECODER_H_
