// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TEST_ARGUMENT_KEY_H_
#define DALI_TEST_ARGUMENT_KEY_H_

#include <assert.h>
#include <string>
#include <tuple>
#include <utility>


namespace dali {
namespace testing {

class ArgumentKey {
 public:
  ArgumentKey(const char *arg_name) noexcept  // NOLINT (non-explicit ctor)
          : node_name_(), arg_name_(arg_name) {
    assert(!arg_name_.empty());  // Arg name has been set either as an empty string or not set at all NOLINT
  }


  ArgumentKey(std::string node_name, std::string arg_name) noexcept
          : node_name_(std::move(node_name)), arg_name_(std::move(arg_name)) {
    assert(!arg_name_.empty());  // Arg name has been set either as an empty string or not set at all NOLINT
    assert(!node_name_.empty());  // Node name has been set either as an empty string or not set at all NOLINT
  }


  std::string node_name() const noexcept {
    return node_name_;
  }


  std::string arg_name() const noexcept {
    return arg_name_;
  }


  bool operator<(const ArgumentKey &rhs) const noexcept {
    return std::tie(node_name_, arg_name_) < std::tie(rhs.node_name_, rhs.arg_name_);
  }


 private:
  std::string node_name_, arg_name_;
};

inline std::ostream& operator<<(std::ostream& os, const ArgumentKey& ak) {
  if (ak.node_name().empty()) {
    os << ak.arg_name();
  } else {
    os << ak.node_name() << ":" << ak.arg_name();
  }
  return os;
}

}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_ARGUMENT_KEY_H_
