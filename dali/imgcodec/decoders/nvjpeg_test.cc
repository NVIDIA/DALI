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

#include <gtest/gtest.h>
#include "dali/imgcodec/decoders/nvjpeg.h"
#include "dali/test/dali_test.h"

namespace dali {
namespace imgcodec {
namespace test {

TEST(NvJpegDecoderTest, Factory) {
  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));

  NvJpegDecoder decoder;
  EXPECT_TRUE(decoder.IsSupported(device_id));
  auto props = decoder.GetProperties();
  EXPECT_TRUE(static_cast<bool>(props.supported_input_kinds & InputKind::HostMemory));;
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::Filename));;
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::DeviceMemory));;
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::Stream));

  ThreadPool tp(4, device_id, false, "nvjpeg decoder test");
  auto instance = decoder.Create(device_id, tp);
  EXPECT_NE(instance, nullptr);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
