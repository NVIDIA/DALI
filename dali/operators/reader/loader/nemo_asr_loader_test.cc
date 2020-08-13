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
#include "dali/operators/reader/loader/nemo_asr_loader.h"

namespace dali {

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

}  // namespace dali
