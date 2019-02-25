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
#include "dali/test/dali_test_config.h"

namespace dali {
namespace testing {
namespace program_options {

namespace {

/// Path to Dali_extra repository
std::string _dali_extra_path;  // NOLINT

std::once_flag noninit_warning;

}  // namespace


const std::string &dali_extra_path() {
  if (_dali_extra_path.empty()) {
    std::call_once(noninit_warning,
                   []() { std::cerr << "dali_extra_path not initialized. Using current path."; });
  }
  return _dali_extra_path;
}

void parse_program_options(int argc, const char **argv) {
  // TODO(mszolucha): in case more args appear, use better solution (e.g. boost::program_options)
  const char key[] = "--dali_extra_path";
  for (int i = 1; i < argc; i++) {
    if (0 == std::strncmp(argv[i], key, sizeof(key)-1)) {
      _dali_extra_path = std::string{&argv[i][sizeof(key)]};
      break;
    }
  }
}

}  // namespace program_options
}  // namespace testing
}  // namespace dali

