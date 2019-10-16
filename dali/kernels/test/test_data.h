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

#ifndef DALI_KERNELS_TEST_TEST_DATA_H_
#define DALI_KERNELS_TEST_TEST_DATA_H_

#include <opencv2/core.hpp>
#include "dali/core/span.h"
#include "dali/test/mat2tensor.h"

namespace dali {
namespace testing {

namespace data {
  span<uint8_t> file(const char *name);
  const cv::Mat &image(const char *name, bool color = true);
}  // namespace data

}  // namespace testing
}  // namespace dali

#endif  // DALI_KERNELS_TEST_TEST_DATA_H_
