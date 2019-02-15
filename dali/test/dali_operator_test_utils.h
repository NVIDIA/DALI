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

extern const std::vector<Arguments> kDevices;  /// List of all devices used in testing framework

/**
 * Remove const pointer: const T* -> T
 */
template<typename T>
using remove_cp = typename std::remove_const<typename std::remove_pointer<T>::type>::type;


/**
 * Assigns pointer to raw data of provided TensorList.
 * @param tl
 * @param destination
 */
template<typename T, typename Backend>
void pointer_to_data(const TensorList<Backend> &tl, T &destination) {
  static_assert(std::is_pointer<T>::value, "T is not a pointer");
  static_assert(std::is_fundamental<remove_cp<T>>::value,
                "T is a pointer to non-fundamental type");
  destination = tl.template data<remove_cp<T>>();
}

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
