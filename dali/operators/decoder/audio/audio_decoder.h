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

#ifndef DALI_AUDIO_DECODER_H
#define DALI_AUDIO_DECODER_H

#include <memory>
#include <dali/core/span.h>

namespace dali {

template<typename T>
struct AudioData {
  std::shared_ptr<T> data;
  int length;
  int sample_rate;  /// [Hz]
  int channels;
  bool channels_interleaved;
};

template<typename T>
struct AudioDecoder {

  // assert proper size of memory
  virtual AudioData<T> Decode(span<const char> encoded) = 0;

  virtual ~AudioDecoder() = default;
};

template<typename T>
struct AllocatingDecoder {
};

template<typename T>
struct NonallocatingDecoder {
  explicit NonallocatingDecoder(std::shared_ptr<T> destination) : destination_(destination) {}

  /**
   * Peeks the encoded buffer and returns, how much memory (in bytes)
   * the decoder requires at `destination`
   */
  virtual size_t memsize(span<const char> encoded) const = 0;

  virtual ~NonallocatingDecoder() = default;

 private:
  std::shared_ptr<T> destination_;
};

template<typename T>
struct WavDecoder : public AudioDecoder<T> {
  ~WavDecoder() override = default;
};

}

#endif //DALI_AUDIO_DECODER_H
