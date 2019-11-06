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

#include "audio_decoder.h"
#include "libsnd_decoder.h"
#include <gtest/gtest.h>
#include <dali/test/dali_test_config.h>
#include <dali/core/format.h>
#include <fstream>

using namespace std;

namespace dali {
namespace {
std::string audio_data_root = make_string(testing::dali_extra_path(), "/db/audio/");


template<typename T>
std::vector<T> file_to_vector(const std::string &filepath) {
  std::ifstream file(filepath.c_str());
  std::istream_iterator<T> begin(file);
  std::istream_iterator<T> end;
  return {begin, end};
}


std::vector<char> file_to_bytes(const std::string &filepath) {
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
bool check_buffers(const T *buf1, const T *buf2, int size) {
  for (int i = 0; i < size; i++) {
    if (buf1[i] != buf2[i]) return false;
  }
  return true;
}
}

TEST(AudioDecoderTest, WavDecoderTest) {
  using DataType = short;
  LibsndWavDecoder<DataType> decoder;
  std::string wav_path = make_string(audio_data_root, "three.wav");
  std::string decoded_path = make_string(audio_data_root, "three.txt");
  int frequency = 16000;
  int channels = 1;
  auto vec = file_to_vector<DataType>(decoded_path);
  auto bytes = file_to_bytes(wav_path);

  auto decoded_data = decoder.Decode(make_cspan(bytes));
  EXPECT_EQ(decoded_data.channels, channels);
  EXPECT_EQ(decoded_data.channels_interleaved, false);
  EXPECT_EQ(decoded_data.sample_rate, frequency);
  EXPECT_PRED3(check_buffers<DataType>,
               decoded_data.data.get(), vec.data(), vec.size());

}

}