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

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <cstring>
#include "dali/core/format.h"
#include "dali/operators/decoder/audio/generic_decoder.h"
#include "dali/test/dali_test_config.h"
#include "dali/core/span.h"

namespace dali {
namespace {

std::string audio_data_root = make_string(testing::dali_extra_path(), "/db/audio/wav/");  // NOLINT

/**
 * Reads file and saves it to vector. Saves the file as plain numbers.
 */
template<typename T>
std::vector<T> ReadTxt(const std::string &filepath) {
  std::ifstream file(filepath.c_str());
  std::istream_iterator<T> begin(file);
  std::istream_iterator<T> end;
  return {begin, end};
}

/**
 * Reads file as a byte stream and saves it to vector.
 */
std::vector<char> ReadBytes(const std::string &filepath) {
  std::vector<char> ret;
  std::ifstream infile(filepath);
  infile.seekg(0, std::ios::end);
  size_t length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  ret.resize(length);
  infile.read(ret.data(), length);
  return ret;
}


template<typename T>
bool CheckBuffers(const T *buf1, const T *buf2, int size) {
  return !std::memcmp(buf1, buf2, sizeof(T) * size);
}

}  // namespace

TEST(AudioDecoderTest, WavDecoderTest) {
  using DataType = short;  // NOLINT
  auto decoder = make_generic_audio_decoder();

  // Contains wav file to be decoded
  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  // Contains raw PCM data decoded offline
  std::string decoded_path = make_string(audio_data_root, "dziendobry.txt");

  constexpr int expected_frequency = 44100;
  constexpr int expected_nchannels = 2;
  std::vector<DataType> vec;
  std::vector<char> bytes;
  try {
    vec = ReadTxt<DataType>(decoded_path);
    bytes = ReadBytes(wav_path);
  } catch (const std::bad_alloc &e) {
    FAIL() << "Test data hasn't been provided: Expected `" << wav_path << "` and `" << decoded_path
           << "` to exist";
  }

  {
    auto meta = decoder->Open(make_cspan(bytes));
    EXPECT_EQ(meta.channels, expected_nchannels);
    EXPECT_EQ(meta.channels_interleaved, true);
    EXPECT_EQ(meta.sample_rate, expected_frequency);
    decoder->Close();
  }

  {
    auto meta = decoder->Open(make_cspan(bytes));
    std::vector<DataType> output(meta.length * meta.channels);
    decoder->Decode(make_span(output));
    EXPECT_PRED3(CheckBuffers<DataType>, output.data(), vec.data(), vec.size());
  }

  {
    auto meta = decoder->Open(make_cspan(bytes));
    std::vector<DataType> output(meta.length * meta.channels);
    decoder->DecodeFrames(output.data(), meta.length);
    EXPECT_PRED3(CheckBuffers<DataType>, output.data(), vec.data(), vec.size());
  }

  {
    auto meta = decoder->Open(make_cspan(bytes));
    int64_t offset = meta.length / 2;
    int64_t length = meta.length - offset;
    // allocating a bigger buffer in purpose
    std::vector<DataType> output(meta.length * meta.channels, 0xBE);
    decoder->SeekFrames(offset, SEEK_CUR);
    decoder->DecodeFrames(output.data(), length);
    EXPECT_PRED3(CheckBuffers<DataType>, output.data(), vec.data() + offset * meta.channels,
                 length * meta.channels);
    // Verifying that we didn't read more than we should
    for (size_t i = length * meta.channels; i < output.size(); i++) {
      ASSERT_EQ(0xBE, output[i]);
    }
  }
}

}  // namespace dali
