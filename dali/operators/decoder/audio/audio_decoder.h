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

#ifndef DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_H_
#define DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_H_

#include <memory>
#include <string>
#include "dali/core/span.h"

namespace dali {

struct AudioMetadata {
  /// @brief Length, in (multi-channel) samples, of the recording
  int64_t length;
  /// @brief Sampling rate, in Hz
  int sample_rate;
  int channels;
  bool channels_interleaved = true;
};

class AudioDecoderBase {
 public:
  AudioMetadata Open(span<const char> encoded) {
    Close();
    return OpenImpl(encoded);
  }

  AudioMetadata OpenFromFile(const std::string &filepath) {
    Close();
    return OpenFromFileImpl(filepath);
  }

  void Close() {
    CloseImpl();
  }

  /**
   * @brief Seeks full frames, or multichannel samples, much like lseek in unistd.h
   * @param nframes Number of full frames (1 frame is equivalent to nchannel samples)
   * @param whence Like in lseek, SEEK_SET. SEEK_CUR, SEEK_END
   * @returns offset in frames from the start of the audio data or -1 if an error occured
   */
  int64_t SeekFrames(int64_t nframes, int whence = SEEK_CUR) {
    return SeekFramesImpl(nframes, whence);
  }

  /**
   * @brief Decode audio samples.
   * @remarks output length should include the number of channels
   *          (audio_length * num_channels)
   * @return Number of samples read
   */
  template <typename T>
  ptrdiff_t Decode(span<T> output) {
    return DecodeImpl(output);
  }

  /**
   * @brief Decode audio frames (1 frame is equivalent to nchannel samples)
   * @return Number of frames read
   */
  template <typename T>
  ptrdiff_t DecodeFrames(T* output, int64_t nframes) {
    return DecodeFramesImpl(output, nframes);
  }

  virtual ~AudioDecoderBase() = default;

 private:
  virtual int64_t SeekFramesImpl(int64_t nframes, int whence) = 0;

  virtual ptrdiff_t DecodeImpl(span<float> output) = 0;
  virtual ptrdiff_t DecodeImpl(span<int16_t> output) = 0;
  virtual ptrdiff_t DecodeImpl(span<int32_t> output) = 0;

  virtual ptrdiff_t DecodeFramesImpl(float* output, int64_t nframes) = 0;
  virtual ptrdiff_t DecodeFramesImpl(int16_t* output, int64_t nframes) = 0;
  virtual ptrdiff_t DecodeFramesImpl(int32_t* output, int64_t nframes) = 0;

  virtual AudioMetadata OpenImpl(span<const char> encoded) = 0;
  virtual AudioMetadata OpenFromFileImpl(const std::string &filepath) = 0;
  virtual void CloseImpl() = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_H_
