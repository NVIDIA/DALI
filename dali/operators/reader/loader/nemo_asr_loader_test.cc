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
#include "dali/kernels/signal/downmixing.h"

namespace dali {

std::string audio_data_root = make_string(testing::dali_extra_path(), "/db/audio/wav/");  // NOLINT

TEST(NemoAsrLoaderTest, ParseManifest) {
  std::stringstream ss;
  ss << R"code({"audio_filepath": "path/to/audio1.wav", "duration": 1.45, "text": "     A ab B C D   "})code" << std::endl;
  ss << R"code({"audio_filepath": "path/to/audio2.wav", "duration": 2.45, "offset": 1.03, "text": "C DA B"})code" << std::endl;
  ss << R"code({"audio_filepath": "path/to/audio3.wav", "duration": 3.45})code" << std::endl;
  std::vector<NemoAsrEntry> entries;
  detail::ParseManifest(entries, ss);
  ASSERT_EQ(3, entries.size());

  EXPECT_EQ("path/to/audio1.wav", entries[0].audio_filepath);
  EXPECT_NEAR(1.45, entries[0].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[0].offset, 1e-7);
  EXPECT_EQ("     A ab B C D   ", entries[0].text);

  EXPECT_EQ("path/to/audio2.wav", entries[1].audio_filepath);
  EXPECT_NEAR(2.45, entries[1].duration, 1e-7);
  EXPECT_NEAR(1.03, entries[1].offset, 1e-7);
  EXPECT_EQ("C DA B", entries[1].text);

  EXPECT_EQ("path/to/audio3.wav", entries[2].audio_filepath);
  EXPECT_NEAR(3.45, entries[2].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[2].offset, 1e-7);
  EXPECT_EQ("", entries[2].text);

  entries.clear();
  ss.clear();
  ss.seekg(0);

  detail::ParseManifest(entries, ss, 0.0f, 0.0f, true);
  ASSERT_EQ(3, entries.size());

  EXPECT_EQ("path/to/audio1.wav", entries[0].audio_filepath);
  EXPECT_NEAR(1.45, entries[0].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[0].offset, 1e-7);
  EXPECT_EQ("a ab b c d", entries[0].text);

  EXPECT_EQ("path/to/audio2.wav", entries[1].audio_filepath);
  EXPECT_NEAR(2.45, entries[1].duration, 1e-7);
  EXPECT_NEAR(1.03, entries[1].offset, 1e-7);
  EXPECT_EQ("c da b", entries[1].text);

  EXPECT_EQ("path/to/audio3.wav", entries[2].audio_filepath);
  EXPECT_NEAR(3.45, entries[2].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[2].offset, 1e-7);
  EXPECT_EQ("", entries[2].text);

  entries.clear();
  ss.clear();
  ss.seekg(0);

  detail::ParseManifest(entries, ss, 2.0f, 3.0f);  // first and third sample should be ignored
  ASSERT_EQ(1, entries.size());
  EXPECT_EQ("path/to/audio2.wav", entries[0].audio_filepath);

  entries.clear();
  ss.clear();
  ss.seekg(0);
  detail::ParseManifest(entries, ss, 0.5f, 2.45f);  // second sample has a duration of exactly 2.45s
  ASSERT_EQ(2, entries.size());
  EXPECT_EQ("path/to/audio1.wav", entries[0].audio_filepath);
  EXPECT_EQ("path/to/audio2.wav", entries[1].audio_filepath);

  entries.clear();
  ss.clear();
  ss.seekg(0);
  detail::ParseManifest(entries, ss, 0.0, 2.44999f);
  ASSERT_EQ(1, entries.size());
  EXPECT_EQ("path/to/audio1.wav", entries[0].audio_filepath);
}

TEST(NemoAsrLoaderTest, WrongManifestPath) {
  auto spec = OpSpec("NemoAsrReader")
                  .AddArg("manifest_filepaths", std::vector<std::string>{"./wrong/file.txt"})
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
  std::string manifest_filepath =
      "/tmp/nemo_asr_manifest_XXXXXX";  // XXXXXX is replaced in tempfile()
  tempfile(manifest_filepath, "{ broken_json ]");

  auto spec = OpSpec("NemoAsrReader")
                  .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                  .AddArg("batch_size", 32)
                  .AddArg("device_id", -1);

  {
    NemoAsrLoader loader(spec);
    ASSERT_THROW(loader.PrepareMetadata(), std::runtime_error);
    EXPECT_EQ(0, loader.Size());
  }

  {
    std::ofstream f(manifest_filepath);
    f << "{}\n{}\n{}";
    f.close();

    NemoAsrLoader loader(spec);
    ASSERT_THROW(loader.PrepareMetadata(), std::runtime_error);
    EXPECT_EQ(0, loader.Size());
  }

  {
    std::ofstream f(manifest_filepath);
    f << "bla bla bla";
    f.close();

    NemoAsrLoader loader(spec);
    ASSERT_THROW(loader.PrepareMetadata(), std::runtime_error);
    EXPECT_EQ(0, loader.Size());
  }

  {
    std::ofstream f(manifest_filepath);
    f << R"code({"audio_filepath": "/audio/filepath.wav", "text": "this is an example", "duration": 0.32})code";
    f.close();

    NemoAsrLoader loader(spec);
    loader.PrepareMetadata();
    EXPECT_EQ(1, loader.Size());
  }

  ASSERT_EQ(0, remove(manifest_filepath.c_str()));
}

TEST(NemoAsrLoaderTest, ReadSample) {
  std::string manifest_filepath =
      "/tmp/nemo_asr_manifest_XXXXXX";  // XXXXXX is replaced in tempfile()

  tempfile(manifest_filepath);
  std::ofstream f(manifest_filepath);
  f << "{\"audio_filepath\": \"" << make_string(audio_data_root, "dziendobry.wav") << "\""
    << ", \"text\": \"dzien dobry\""
    << ", \"duration\": 3.0"
    << "}";
  f.close();

  // Contains wav file to be decoded
  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  // Contains raw PCM data decoded offline
  std::string decoded_path = make_string(audio_data_root, "dziendobry.txt");

  std::ifstream file(decoded_path.c_str());
  std::vector<int16_t> ref_data{std::istream_iterator<int16_t>(file),
                                std::istream_iterator<int16_t>()};
  int64_t ref_sz = ref_data.size();
  int64_t ref_samples = ref_sz/2;
  int64_t ref_channels = 2;

  {
    auto spec = OpSpec("NemoAsrReader")
                    .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                    .AddArg("downmix", false)
                    .AddArg("dtype", DALI_INT16)
                    .AddArg("num_threads", 4)
                    .AddArg("batch_size", 32)
                    .AddArg("device_id", -1);

    NemoAsrLoader loader(spec);
    loader.PrepareMetadata();
    AsrSample sample;
    loader.ReadSample(sample);
    TensorView<StorageCPU, int16_t> ref(ref_data.data(), {ref_samples, 2});
    Check(ref, view<const int16_t>(sample.audio()));
  }

  std::vector<float> downmixed(ref_samples, 0.0f);
  for (int i = 0; i < ref_samples; i++) {
    double l = ConvertSatNorm<float>(ref_data[2*i]);
    double r = ConvertSatNorm<float>(ref_data[2*i+1]);
    downmixed[i] = (l + r) / 2;
  }
  {
    auto spec = OpSpec("NemoAsrReader")
                    .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                    .AddArg("downmix", true)
                    .AddArg("dtype", DALI_FLOAT)
                    .AddArg("num_threads", 4)
                    .AddArg("batch_size", 32)
                    .AddArg("device_id", -1);

    NemoAsrLoader loader(spec);
    loader.PrepareMetadata();
    AsrSample sample;
    loader.ReadSample(sample);
    TensorView<StorageCPU, float> ref(downmixed.data(), {ref_samples});
    Check(ref, view<const float>(sample.audio()));
  }

  {
    float sr_in = 44100.0f;
    float sr_out = 22050.0f;

    AsrSample sample;
    {
      auto spec = OpSpec("NemoAsrReader")
                      .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                      .AddArg("downmix", true)
                      .AddArg("sample_rate", sr_out)
                      .AddArg("dtype", DALI_FLOAT)
                      .AddArg("num_threads", 4)
                      .AddArg("batch_size", 32)
                      .AddArg("device_id", -1);
      NemoAsrLoader loader(spec);
      loader.PrepareMetadata();
      loader.ReadSample(sample);
    }

    int64_t downsampled_len =
        kernels::signal::resampling::resampled_length(ref_samples, sr_in, sr_out);
    std::vector<float> downsampled(downsampled_len, 0.0f);
    constexpr double q = 50.0;
    int lobes = std::round(0.007 * q * q - 0.09 * q + 3);
    kernels::signal::resampling::Resampler resampler;
    resampler.Initialize(lobes, lobes * 64 + 1);
    resampler.Resample(downsampled.data(), 0, downsampled_len, sr_out, downmixed.data(),
                       downmixed.size(), sr_in, 1);

    TensorView<StorageCPU, float> ref(downsampled.data(), {downsampled_len});
    Check(ref, view<const float>(sample.audio()), EqualEpsRel(1e-6, 1e-6));

    AsrSample sample_int16;
    {
      auto spec = OpSpec("NemoAsrReader")
                    .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                    .AddArg("downmix", true)
                    .AddArg("sample_rate", static_cast<float>(sr_out))
                    .AddArg("dtype", DALI_INT16)
                    .AddArg("num_threads", 4)
                    .AddArg("batch_size", 32)
                    .AddArg("device_id", -1);
      NemoAsrLoader loader(spec);
      loader.PrepareMetadata();
      loader.ReadSample(sample_int16);
    }

    ASSERT_EQ(volume(sample.audio().shape()), volume(sample_int16.audio().shape()));
    std::vector<float> converted(downsampled_len, 0.0f);
    for (size_t i = 0; i < converted.size(); i++)
      converted[i] = ConvertSatNorm<float>(sample_int16.audio().data<int16_t>()[i]);
    TensorView<StorageCPU, float> converted_from_int16(converted.data(), {downsampled_len});
    Check(ref, converted_from_int16, EqualEpsRel(1e-6, 1e-6));
  }
}

}  // namespace dali

