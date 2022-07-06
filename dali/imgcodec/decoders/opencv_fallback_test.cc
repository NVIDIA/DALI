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
#include <string>
#include <vector>
#include "dali/imgcodec/decoders/opencv_fallback.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {
namespace imgcodec {
namespace test {

TEST(OpenCVFallback, Factory) {
    OpenCVDecoder decoder;
    EXPECT_TRUE(decoder.IsSupported(CPU_ONLY_DEVICE_ID));
    auto props = decoder.GetProperties();
    EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::HostMemory));
    EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::Filename));
    EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::DeviceMemory));
    EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Stream));

    ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "dummy");
    auto instance = decoder.Create(CPU_ONLY_DEVICE_ID, tp);
    EXPECT_NE(instance, nullptr);
}

TEST(OpenCVFallback, Decode) {
    ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "dummy");
    OpenCVDecoder decoder;
    auto instance = decoder.Create(CPU_ONLY_DEVICE_ID, tp);
    ASSERT_NE(instance, nullptr);
    //EXPECT_TRUE(instance->CanDecode(img));
    //ImageFormat
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
