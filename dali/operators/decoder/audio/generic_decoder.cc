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

#include "dali/operators/decoder/audio/generic_decoder.h"

namespace dali {

namespace {

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

}  // namespace


template<typename SampleType>
void GenericAudioDecoder<SampleType>::DecodeTyped(span<SampleType> output) {
  impl_->DecodeTyped(output);
}


template<typename SampleType>
AudioMetadata GenericAudioDecoder<SampleType>::OpenImpl(span<const char> encoded) {
  return impl_->OpenImpl(encoded);
}


template<typename SampleType>
void GenericAudioDecoder<SampleType>::CloseImpl() {
  impl_->CloseImpl();
}


template<typename SampleType>
GenericAudioDecoder<SampleType>::~GenericAudioDecoder() = default;


template<typename SampleType>
GenericAudioDecoder<SampleType>::GenericAudioDecoder() :
        impl_(std::make_unique<Impl>()) {
}


template<typename SampleType>
struct GenericAudioDecoder<SampleType>::Impl {
  void DecodeTyped(span<SampleType> output) {
    read_samples(snd_file_, output);
  }


  AudioMetadata OpenImpl(span<const char> encoded) {
    assert(!encoded.empty());
    assert(encoded.data());
    AudioMetadata ret;
    sf_info_ = {};
    sf_info_.format = 0;
    mem_stream_ = {encoded.data(), static_cast<int>(encoded.size())};
    SF_VIRTUAL_IO sf_virtual_io = {
            &GetFileLen,
            &Seek,
            &Read,
            nullptr,  // No writing
            &Tell
    };
    snd_file_ = sf_open_virtual(&sf_virtual_io, SFM_READ, &sf_info_, &mem_stream_);
    DALI_ENFORCE(snd_file_, make_string("Failed to open encoded data: ", sf_strerror(snd_file_)));

    ret.length = sf_info_.frames * sf_info_.channels;
    ret.channels = sf_info_.channels;
    ret.sample_rate = sf_info_.samplerate;
    ret.channels_interleaved = false;
    return ret;
  }


  void CloseImpl() {
    if (snd_file_) {
      auto err = sf_close(snd_file_);
      DALI_ENFORCE(err == 0, make_string("Failed to close SNDFILE: ", sf_error_number(err)));
      snd_file_ = nullptr;
    }
    mem_stream_ = {};
  }


  SNDFILE *snd_file_ = nullptr;
  SF_INFO sf_info_ = {};
  MemoryStream mem_stream_ = {};
};

/*
 * Force instantiation only for given types
 */
template class GenericAudioDecoder<short>;  // NOLINT
template class GenericAudioDecoder<int>;    // NOLINT
template class GenericAudioDecoder<float>;  // NOLINT


}  // namespace dali
