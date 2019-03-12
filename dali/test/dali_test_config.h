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

#ifndef DALI_TEST_DALI_TEST_CONFIG_H_
#define DALI_TEST_DALI_TEST_CONFIG_H_

#include <string>

namespace dali {
namespace testing {

/**
 * Returns path to DALI_extra repository
 */
const std::string &dali_extra_path();

}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_CONFIG_H_
