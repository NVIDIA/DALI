// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/imgcodec/image_source.h"

#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

TEST(image_source, open_rewind) {
  const char data[] = {'a', 'b', 'c', 'd'};
  MemInputStream mis(data, 4);
  auto src = ImageSource::FromStream(&mis);
  auto stream = src.Open();
  EXPECT_EQ('a', (stream->ReadOne<char>()));
  EXPECT_EQ('b', (stream->ReadOne<char>()));
  auto stream2 = src.Open();
  EXPECT_EQ('a', (stream2->ReadOne<char>()));
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
