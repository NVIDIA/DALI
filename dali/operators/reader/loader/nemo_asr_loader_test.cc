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

#include <gtest/gtest.h>
#include <sstream>
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/operators/reader/loader/nemo_asr_loader.h"

namespace dali {

std::string audio_data_root = make_string(testing::dali_extra_path(), "/db/audio/wav/");  // NOLINT

TEST(NemoAsrLoaderTest, ParseManifest) {
  std::stringstream ss;
  ss << "[";
  ss << R"code({"audio_filepath": "path/to/audio1.wav", "duration": 1.45, "text": "A B CD"}, )code";
  ss << R"code({"audio_filepath": "path/to/audio2.wav", "duration": 2.45, "offset": 1.03, "text": "C DA B"}, )code";
  ss << R"code({"audio_filepath": "path/to/audio3.wav", "duration": 3.45})code";
  ss << "]";
  auto s = ss.str();
  std::vector<NemoAsrEntry> entries;
  detail::ParseManifest(entries, ss.str());
  ASSERT_EQ(3, entries.size());

  EXPECT_EQ("path/to/audio1.wav", entries[0].audio_filepath);
  EXPECT_NEAR(1.45, entries[0].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[0].offset, 1e-7);
  EXPECT_EQ("A B CD", entries[0].text);

  EXPECT_EQ("path/to/audio2.wav", entries[1].audio_filepath);
  EXPECT_NEAR(2.45, entries[1].duration, 1e-7);
  EXPECT_NEAR(1.03, entries[1].offset, 1e-7);
  EXPECT_EQ("C DA B", entries[1].text);

  EXPECT_EQ("path/to/audio3.wav", entries[2].audio_filepath);
  EXPECT_NEAR(3.45, entries[2].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[2].offset, 1e-7);
  EXPECT_EQ("", entries[2].text);
}

TEST(NemoAsrLoaderTest, WrongManifestPath) {
  auto spec = OpSpec("NemoAsrReader")
                  .AddArg("manifest_filepath", "./wrong/file.txt")
                  .AddArg("batch_size", 32)
                  .AddArg("device_id", -1);
  NemoAsrLoader loader(spec);
  ASSERT_THROW(loader.PrepareMetadata(), std::runtime_error);
}

void tempfile(std::string& filename, std::string content = "") {
  int fd = mkstemp(&filename[0]);
  ASSERT_NE(-1, fd);
  if (!content.empty())
    write(fd, content.c_str(), content.size());
  close(fd);
}

TEST(NemoAsrLoaderTest, ParseManifestContent) {
  std::string manifest_filepath = "/tmp/nemo_asr_manifest_XXXXXX";  // XXXXXX is replaced in tempfile() 
  tempfile(manifest_filepath, "{ broken_json ]");

  auto spec = OpSpec("NemoAsrReader")
                  .AddArg("manifest_filepath", manifest_filepath)
                  .AddArg("batch_size", 32)
                  .AddArg("device_id", -1);
  
  {
    NemoAsrLoader loader(spec);
    ASSERT_THROW(loader.PrepareMetadata(), std::runtime_error);
  }

  {
    std::ofstream f(manifest_filepath);
    f << "[{}, {}, {}]";
    f.close();

    NemoAsrLoader loader(spec);
    loader.PrepareMetadata();
    EXPECT_EQ(3, loader.Size());
  }

  {
    std::ofstream f(manifest_filepath);
    f << R"code([{"audio_filepath": "/audio/filepath.wav", "text": "this is an example", "duration": 0.32}])code";
    f.close();

    NemoAsrLoader loader(spec);
    loader.PrepareMetadata();
    EXPECT_EQ(1, loader.Size());
  }

  ASSERT_EQ(0, remove(manifest_filepath.c_str()));
}

TEST(NemoAsrLoaderTest, ReadSample) {
  std::string manifest_filepath = "/tmp/nemo_asr_manifest_XXXXXX";  // XXXXXX is replaced in tempfile()
  
  tempfile(manifest_filepath);
  std::ofstream f(manifest_filepath);
  f << "[{\"audio_filepath\": \"" << make_string(audio_data_root, "dziendobry.wav") << "\""
    << ", \"text\": \"dzien dobry\""
    << ", \"duration\": 3.0"
    << "}]";
  f.close();

  // Contains wav file to be decoded
  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  // Contains raw PCM data decoded offline
  std::string decoded_path = make_string(audio_data_root, "dziendobry.txt");

  std::ifstream file(decoded_path.c_str());
  std::vector<short> ref_data{std::istream_iterator<short>(file), std::istream_iterator<short>()};  
  int64_t ref_sz = ref_data.size();
  int64_t ref_samples = ref_sz/2;
  int64_t ref_channels = 2;

  {
    auto spec = OpSpec("NemoAsrReader")
                    .AddArg("manifest_filepath", manifest_filepath)
                    .AddArg("downmix", false)
                    .AddArg("dtype", DALI_INT16)
                    .AddArg("batch_size", 32)
                    .AddArg("device_id", -1);

    NemoAsrLoader loader(spec);
    loader.PrepareMetadata();
    AsrSample sample;
    loader.ReadSample(sample);

    TensorView<StorageCPU, short, 2> ref(ref_data.data(), {ref_samples, 2});
    Check(ref, view<short, 2>(sample.audio));
  }

  {
    auto spec = OpSpec("NemoAsrReader")
                    .AddArg("manifest_filepath", manifest_filepath)
                    .AddArg("downmix", true)
                    .AddArg("dtype", DALI_FLOAT)
                    .AddArg("batch_size", 32)
                    .AddArg("device_id", -1);

    NemoAsrLoader loader(spec);
    loader.PrepareMetadata();
    AsrSample sample;
    loader.ReadSample(sample);

    std::vector<float> downmixed(ref_samples, 0.0f);
    for (int i = 0; i < ref_samples; i++) {
      double l = ref_data[2*i];
      double r = ref_data[2*i+1];
      downmixed[i] = (l + r) / 2;
    }
    TensorView<StorageCPU, float, 1> ref(downmixed.data(), {ref_samples});
    Check(ref, view<float, 1>(sample.audio));
  }
}

}  // namespace dali

