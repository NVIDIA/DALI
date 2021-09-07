// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_DALI_OPERATOR_TEST_UTILS_H_
#define DALI_TEST_DALI_OPERATOR_TEST_UTILS_H_

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "dali/test/dali_operator_test.h"

namespace dali {
namespace testing {

namespace utils {

/// List of all devices used in testing framework
DLL_PUBLIC extern const std::vector<Arguments> kDevices;

}  // namespace utils

inline std::vector<testing::Arguments> cartesian(std::vector<testing::Arguments> args_vec) {
  return args_vec;
}


template<typename... Ts>
inline std::vector<testing::Arguments> cartesian(std::vector<testing::Arguments> args_vec,
                                                 Ts... args_vecs) {
  std::vector<testing::Arguments> result;
  for (auto args : args_vec) {
    auto prod_rest = cartesian(args_vecs...);
    for (auto args_rest : prod_rest) {
      args_rest.insert(args.begin(), args.end());
      result.push_back(std::move(args_rest));
    }
  }
  return result;
}


}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_DALI_OPERATOR_TEST_UTILS_H_
