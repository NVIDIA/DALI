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
#include <cstdio>
#include <utility>
#include <sstream>
#include <string>
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/operators/reader/loader/nemo_asr_loader.h"
#include "dali/kernels/signal/downmixing.h"

namespace dali {

float original_sample_rate = 44100.0f;
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

TEST(NemoAsrLoaderTest, ParseNonAsciiTransript) {
  {
    std::stringstream ss;
    ss << R"code({"audio_filepath": "path/to/audio1.wav", "duration": 1.45, "text": "это проверка"})code" << std::endl;
    std::vector<NemoAsrEntry> entries;
    detail::ParseManifest(entries, ss);
    ASSERT_EQ(1, entries.size());
    EXPECT_EQ(std::string(u8"это проверка"), entries[0].text);
  }

  {
    std::stringstream ss;
    ss << R"code({"audio_filepath": "path/to/audio1.wav", "duration": 1.45, "text": "这是一个测试"})code" << std::endl;
    std::vector<NemoAsrEntry> entries;
    detail::ParseManifest(entries, ss);
    ASSERT_EQ(1, entries.size());
    EXPECT_EQ(std::string(u8"这是一个测试"), entries[0].text);
  }
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
  if (!content.empty()) {
    ASSERT_EQ(write(fd, content.c_str(), content.size()), content.size());
  }
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

  ASSERT_EQ(0, std::remove(manifest_filepath.c_str()));
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
    Tensor<CPUBackend> sample_audio;
    sample_audio.set_type(TypeTable::GetTypeInfo(DALI_INT16));
    loader.ReadSample(sample);
    sample_audio.Resize(sample.shape());
    sample.decode_audio(sample_audio, 0);
    TensorView<StorageCPU, int16_t> ref(ref_data.data(), {ref_samples, 2});
    Check(ref, view<const int16_t>(sample_audio));
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
    Tensor<CPUBackend> sample_audio;
    sample_audio.set_type(TypeTable::GetTypeInfo(DALI_FLOAT));
    loader.ReadSample(sample);
    sample_audio.Resize(sample.shape());
    sample.decode_audio(sample_audio, 0);
    TensorView<StorageCPU, float> ref(downmixed.data(), {ref_samples});
    Check(ref, view<const float>(sample_audio), EqualEpsRel(1e-5, 1e-5));
  }

  {
    float sr_in = original_sample_rate;
    float sr_out = 22050.0f;

    AsrSample sample;
    Tensor<CPUBackend> sample_audio;
    sample_audio.set_type(TypeTable::GetTypeInfo(DALI_FLOAT));
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
      sample_audio.Resize(sample.shape());
      sample.decode_audio(sample_audio, 0);
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
    Check(ref, view<const float>(sample_audio), EqualEpsRel(1e-5, 1e-5));

    AsrSample sample_int16;
    Tensor<CPUBackend> sample_int16_audio;
    sample_int16_audio.set_type(TypeTable::GetTypeInfo(DALI_INT16));
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
      sample_int16_audio.Resize(sample_int16.shape());
      sample_int16.decode_audio(sample_int16_audio, 0);
    }

    ASSERT_EQ(volume(sample_audio.shape()), volume(sample_int16_audio.shape()));
    std::vector<float> converted(downsampled_len, 0.0f);
    for (size_t i = 0; i < converted.size(); i++)
      converted[i] = ConvertSatNorm<float>(sample_int16_audio.data<int16_t>()[i]);
    TensorView<StorageCPU, float> converted_from_int16(converted.data(), {downsampled_len});
    Check(ref, converted_from_int16, EqualEpsRel(1e-5, 1e-5));
  }

  ASSERT_EQ(0, std::remove(manifest_filepath.c_str()));
}


TEST(NemoAsrLoaderTest, ReadSample_OffsetAndDuration) {
  // Contains wav file to be decoded
  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  // Contains raw PCM data decoded offline
  std::string decoded_path = make_string(audio_data_root, "dziendobry.txt");
  std::ifstream file(decoded_path.c_str());
  std::vector<int16_t> ref_data{std::istream_iterator<int16_t>(file),
                                std::istream_iterator<int16_t>()};
  int64_t ref_sz = ref_data.size();
  int64_t ref_frames = ref_sz / 2;
  int64_t ref_channels = 2;

  std::string manifest_filepath = "/tmp/nemo_asr_manifest_XXXXXX";
  tempfile(manifest_filepath);

  std::vector<std::pair<double, double>> offset_and_duration = {
      std::make_pair(0.0, -1.0),
      std::make_pair(0.5, -1.0),
      std::make_pair(0.05, 0.2),
      std::make_pair(0.05, 100.2)
  };
  for (const auto &entry : offset_and_duration) {
    double offset_sec, duration_sec;
    std::tie(offset_sec, duration_sec) = entry;
    std::ofstream f(manifest_filepath);
    f << "{\"audio_filepath\": \"" << wav_path << "\"";
    if (offset_sec > 0)
      f << ", \"offset\": " << offset_sec;
    if (duration_sec > 0)
      f << ", \"duration\": " << duration_sec;
    f << "}";
    f.close();

    auto spec = OpSpec("NemoAsrReader")
          .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
          .AddArg("downmix", false)
          .AddArg("dtype", DALI_INT16)
          .AddArg("num_threads", 4)
          .AddArg("batch_size", 32)
          .AddArg("device_id", -1);

    NemoAsrLoader loader(spec);
    loader.PrepareMetadata();

    int64_t offset = 0;
    int64_t length = ref_frames;
    if (offset_sec > 0) {
      offset = static_cast<int64_t>(offset_sec * original_sample_rate);
    }
    if (duration_sec > 0) {
      length = static_cast<int64_t>(duration_sec * original_sample_rate);
    }
    if (offset + length > ref_frames) {
      length = ref_frames - offset;
    }

    AsrSample sample;
    Tensor<CPUBackend> sample_audio;
    sample_audio.set_type(TypeTable::GetTypeInfo(DALI_INT16));

    loader.ReadSample(sample);

    TensorShape<> expected_sh{length, 2};
    ASSERT_EQ(expected_sh, sample.shape());
    sample_audio.Resize(sample.shape());
    sample.decode_audio(sample_audio, 0);

    TensorView<StorageCPU, int16_t> ref(ref_data.data() + offset * 2, expected_sh);
    Check(ref, view<const int16_t>(sample_audio));
  }
  ASSERT_EQ(0, std::remove(manifest_filepath.c_str()));
}


}  // namespace dali

