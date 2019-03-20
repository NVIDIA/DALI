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

#include <mutex>
#include <cstring>
#include <cassert>
#include <iostream>
#include <memory>
#include "dali/test/dali_test_config.h"

namespace dali {
namespace testing {


const std::string &dali_extra_path() {
  // Path to DALI_extra repository
  // XXX: In order to avoid Static Initialization Order Fiasco,
  //      _dali_extra_path is a static pointer to string.
  //      ("Construct on first use" idiom).
  //      There's a "leak" here, but we don't really care about it.
  static std::string *_dali_extra_path = new std::string();

  static std::once_flag noninit_warning;

  std::call_once(noninit_warning, []() {
      auto ptr = std::getenv("DALI_EXTRA_PATH");
      if (!ptr) {
        std::cerr << "DALI_EXTRA_PATH not initialized." << std::endl;
        *_dali_extra_path = ".";
      } else {
        *_dali_extra_path = std::string(ptr);
      }
  });
  return *_dali_extra_path;
}

}  // namespace testing
}  // namespace dali

