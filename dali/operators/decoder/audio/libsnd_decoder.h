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

#ifndef DALI_LIBSND_DECODER_H
#define DALI_LIBSND_DECODER_H

#include <cstring>
#include <cassert>
#include <vector>
#include "dali/core/format.h"
#include "dali/core/error_handling.h"
#include "dynlink_snd.h"
#include "audio_decoder.h"


namespace dali {

namespace detail {

class VirtualInputManager {
 public:
  VirtualInputManager(const void *input, int length /* bytes */) :
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


/**
 * The virtual file context must return the length of the virtual file in bytes.
 */
sf_count_t GetFileLen(void *This) {
  return static_cast<VirtualInputManager *>(This)->length();
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
  return static_cast<VirtualInputManager *>(This)->seek(offset, whence);
}


/**
 * The virtual file context must copy ("read") "num" bytes into the buffer provided by dst
 * and return the count of actually copied bytes.
 */
sf_count_t Read(void *dst, sf_count_t num, void *This) {
  return static_cast<VirtualInputManager *>(This)->read(dst, num);
}


/**
 * Return the current position of the virtual file context.
 */
sf_count_t Tell(void *This) {
  return static_cast<VirtualInputManager *>(This)->tell();
}

}  // namespace detail

template<typename T>
struct LibsndWavDecoder : public WavDecoder<T>, public AllocatingDecoder<T> {
  LibsndWavDecoder() {
//    DALI_ENFORCE(LibsndInitChecked(), "Failed to open Libsnd");
    snd::init_snd();
  }


  AudioData<T> Decode(span<const char> encoded) override {
    assert(!encoded.empty());
    assert(encoded.data());
    AudioData<T> ret;
    SF_INFO sf_info;
    sf_info.format = 0;

    SF_VIRTUAL_IO sf_virtual_io = {
            &detail::GetFileLen,
            &detail::Seek,
            &detail::Read,
            nullptr, // No writing
            &detail::Tell
    };

    detail::VirtualInputManager vim(encoded.data(), encoded.size() * sizeof(T));

    auto sound = snd::sf_open_virtual(&sf_virtual_io, SFM_READ, &sf_info, &vim);
    ret.data = std::shared_ptr<T>(new T[sf_info.frames]);

    DALI_ENFORCE(sound, make_string("Failed to open encoded data: ", snd::sf_strerror(sound)));
    auto cnt = snd::sf_readf_short(sound, ret.data.get(), sf_info.frames);

    ret.length = sf_info.frames * sf_info.channels;
    ret.channels = sf_info.channels;
    ret.sample_rate = sf_info.samplerate;
    ret.channels_interleaved = false;
    return ret;
  }
};
}

#endif //DALI_LIBSND_DECODER_H
