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

namespace detail {

template<typename SampleType>
size_t read_samples(SNDFILE *snd_file, span<SampleType> output) {
  DALI_FAIL("Can't find function for given type. You may want to write your own specialization.")
}


template<>
size_t read_samples<short>(SNDFILE *snd_file, span<short> output) {  // NOLINT
  return sf_read_short(snd_file, output.data(), output.size());
}


template<>
size_t read_samples<int>(SNDFILE *snd_file, span<int> output) {
  return sf_read_int(snd_file, output.data(), output.size());
}


template<>
size_t read_samples<float>(SNDFILE *snd_file, span<float> output) {
  return sf_read_float(snd_file, output.data(), output.size());
}


template<>
size_t read_samples<double>(SNDFILE *snd_file, span<double> output) {
  return sf_read_double(snd_file, output.data(), output.size());
}


class MemoryStream {
 public:
  MemoryStream() : length_(0), curr_(0), input_(nullptr) {}


  MemoryStream(const void *input, int length /* bytes */) :
          length_(length), curr_(0), input_(static_cast<const char *>(input)) {}


  size_t length() {
    return length_;
  }


  size_t seek(sf_count_t offset, int whence) {
    switch (whence) {
      case SEEK_SET:
        curr_ = offset;
        break;
      case SEEK_CUR:
        curr_ += offset;
        break;
      case SEEK_END:
        curr_ = length_ + offset;
        break;
      default:
        DALI_FAIL("Incorrect `whence` argument")
    }
    return curr_;
  }


  size_t read(void *dst, sf_count_t num) {
    num = std::min<sf_count_t>(num, length_ - curr_);
    memcpy(dst, input_ + curr_, num);
    curr_ += num;
    return num;
  }


  size_t tell() {
    return curr_;
  }


 private:
  size_t length_, curr_;
  const char *input_;
};

/*
 * Functions: `GetFileLen`, `Seek`, `Read`, `Tell` are the callbacks required
 * by `sf_open_fd` interface. Descriptions of these functions are copy-pasted
 * from `libsnd` docs.
 */

/**
 * The virtual file context must return the length of the virtual file in bytes.
 */
sf_count_t GetFileLen(void *This) {
  return static_cast<MemoryStream *>(This)->length();
}


/**
 *  The virtual file context must seek to offset using the seek mode provided by whence
 *  which is one of
 *   SEEK_CUR
 *   SEEK_SET
 *   SEEK_END
 * The return value must contain the new offset in the file.
 */
sf_count_t Seek(sf_count_t offset, int whence, void *This) {
  return static_cast<MemoryStream *>(This)->seek(offset, whence);
}


/**
 * The virtual file context must copy ("read") "num" bytes into the buffer provided by dst
 * and return the count of actually copied bytes.
 */
sf_count_t Read(void *dst, sf_count_t num, void *This) {
  return static_cast<MemoryStream *>(This)->read(dst, num);
}


/**
 * Return the current position of the virtual file context.
 */
sf_count_t Tell(void *This) {
  return static_cast<MemoryStream *>(This)->tell();
}

}  // namespace detail

/**
 * Generic decoder, that serves as a fallback to most of the formats we need.
 * It uses `libsnd` for decoding.
 * @tparam SampleType
 */
template<typename SampleType>
class GenericAudioDecoder : public TypedAudioDecoderBase<SampleType> {
 public:
  void DecodeTyped(span <SampleType> output) override {
    detail::read_samples(snd_file, output);
  }


 private:
  AudioMetadata OpenImpl(span<const char> encoded) override {
    assert(!encoded.empty());
    assert(encoded.data());
    AudioMetadata ret;
    sf_info = {};
    sf_info.format = 0;
    mem_stream = detail::MemoryStream(encoded.data(), encoded.size());
    SF_VIRTUAL_IO sf_virtual_io = {
            &detail::GetFileLen,
            &detail::Seek,
            &detail::Read,
            nullptr,  // No writing
            &detail::Tell
    };
    snd_file = sf_open_virtual(&sf_virtual_io, SFM_READ, &sf_info, &mem_stream);
    DALI_ENFORCE(snd_file, make_string("Failed to open encoded data: ", sf_strerror(snd_file)));

    ret.length = sf_info.frames * sf_info.channels;
    ret.channels = sf_info.channels;
    ret.sample_rate = sf_info.samplerate;
    ret.channels_interleaved = false;
    return ret;
  }


  void CloseImpl() override {
    if (snd_file) {
      auto err = sf_close(snd_file);
      DALI_ENFORCE(err == 0, make_string("Failed to close SNDFILE: ", sf_error_number(err)));
      snd_file = nullptr;
    }
    mem_stream = {};
  }


  SNDFILE *snd_file = nullptr;
  SF_INFO sf_info = {};
  detail::MemoryStream mem_stream = {};
};
}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_GENERIC_DECODER_H_
