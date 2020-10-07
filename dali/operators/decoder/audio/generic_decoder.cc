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
#include <sndfile.h>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/operators/decoder/audio/audio_decoder.h"

namespace dali {

namespace {

class MemoryStream {
 public:
  MemoryStream(const void *input, int length /* bytes */) :
          length_(length), curr_(0), input_(static_cast<const char *>(input)) {}


  MemoryStream() : MemoryStream(nullptr, 0) {}


  size_t Length() {
    return length_;
  }


  size_t Seek(sf_count_t offset, int whence) {
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


  size_t Read(void *dst, sf_count_t num) {
    num = std::min<sf_count_t>(num, length_ - curr_);
    memcpy(dst, input_ + curr_, num);
    curr_ += num;
    return num;
  }


  size_t Tell() {
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
sf_count_t GetFileLen(void *self) {
  return static_cast<MemoryStream *>(self)->Length();
}


/**
 *  The virtual file context must seek to offset using the seek mode provided by whence
 *  which is one of
 *   SEEK_CUR
 *   SEEK_SET
 *   SEEK_END
 * The return value must contain the new offset in the file.
 */
sf_count_t Seek(sf_count_t offset, int whence, void *self) {
  return static_cast<MemoryStream *>(self)->Seek(offset, whence);
}


/**
 * The virtual file context must copy ("read") "num" bytes into the buffer provided by dst
 * and return the count of actually copied bytes.
 */
sf_count_t Read(void *dst, sf_count_t num, void *self) {
  return static_cast<MemoryStream *>(self)->Read(dst, num);
}


/**
 * Return the current position of the virtual file context.
 */
sf_count_t Tell(void *self) {
  return static_cast<MemoryStream *>(self)->Tell();
}

/**
 * @brief Produce audio metadata instace from SF_INFO
 */
AudioMetadata GetAudioMetadata(const SF_INFO &sf_info) {
  AudioMetadata ret;
  ret.length = sf_info.frames;
  ret.channels = sf_info.channels;
  ret.sample_rate = sf_info.samplerate;
  ret.channels_interleaved = true;
  return ret;
}

}  // namespace


class GenericAudioDecoder : public AudioDecoderBase {
 public:
  GenericAudioDecoder() = default;
  GenericAudioDecoder(GenericAudioDecoder&&) = default;
  GenericAudioDecoder& operator=(GenericAudioDecoder&&) = default;

  ~GenericAudioDecoder() override {
    CloseImpl();
  }

  bool CanSeekFrames() const override {
    return true;
  };

  int64_t SeekFrames(int64_t nframes, int whence) override {
    return sf_seek(sndfile_handle_, nframes, whence);
  }

  ptrdiff_t Decode(span<int16_t> output) override {
    return sf_read_short(sndfile_handle_, output.data(), output.size());
  }

  ptrdiff_t Decode(span<int32_t> output) override {
    return sf_read_int(sndfile_handle_, output.data(), output.size());
  }

  ptrdiff_t Decode(span<float> output) override {
    return sf_read_float(sndfile_handle_, output.data(), output.size());
  }

  ptrdiff_t DecodeFrames(int16_t* output, int64_t nframes) override {
    return sf_readf_short(sndfile_handle_, output, nframes);
  }

  ptrdiff_t DecodeFrames(int32_t* output, int64_t nframes) override {
    return sf_readf_int(sndfile_handle_, output, nframes);
  }

  ptrdiff_t DecodeFrames(float* output, int64_t nframes) override {
    return sf_readf_float(sndfile_handle_, output, nframes);
  }

 private:
  AudioMetadata OpenImpl(span<const char> encoded) override;
  AudioMetadata OpenFromFileImpl(const std::string &filepath) override;
  void CloseImpl() override;

  SNDFILE *sndfile_handle_ = nullptr;
  SF_INFO sf_info_ = {};
  MemoryStream mem_stream_ = {};
};


AudioMetadata GenericAudioDecoder::OpenImpl(span<const char> encoded) {
  assert(!encoded.empty());
  sf_info_ = {};
  mem_stream_ = {encoded.data(), static_cast<int>(encoded.size())};
  SF_VIRTUAL_IO sf_virtual_io = {
          &GetFileLen,
          &Seek,
          &Read,
          nullptr,  // No writing
          &Tell
  };
  sndfile_handle_ = sf_open_virtual(&sf_virtual_io, SFM_READ, &sf_info_, &mem_stream_);
  if (!sndfile_handle_) {
    throw DALIException(make_string("Failed to open encoded data: ", sf_strerror(sndfile_handle_)));
  }
  return GetAudioMetadata(sf_info_);
}

AudioMetadata GenericAudioDecoder::OpenFromFileImpl(const std::string &filepath) {
  DALI_ENFORCE(!filepath.empty(), "filepath is empty");
  sf_info_ = {};
  sndfile_handle_ = sf_open(filepath.c_str(), SFM_READ, &sf_info_);
  if (!sndfile_handle_) {
    throw DALIException(make_string("Failed to open encoded data: ", sf_strerror(sndfile_handle_),
                                    ", filepath: ", filepath));
  }
  return GetAudioMetadata(sf_info_);
}

void GenericAudioDecoder::CloseImpl() {
  if (sndfile_handle_) {
    auto err = sf_close(sndfile_handle_);
    DALI_ENFORCE(err == 0, make_string("Failed to close SNDFILE: ", sf_error_number(err)));
    sndfile_handle_ = nullptr;
  }
  mem_stream_ = {};
}

std::unique_ptr<AudioDecoderBase> make_generic_audio_decoder() {
  return std::make_unique<GenericAudioDecoder>();
}

}  // namespace dali
