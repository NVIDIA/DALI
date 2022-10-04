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

#ifndef DALI_CORE_MULTI_ERROR_H_
#define DALI_CORE_MULTI_ERROR_H_

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dali {

class MultipleErrors : public std::runtime_error {
 public:
  explicit MultipleErrors(std::vector<std::exception_ptr> errors)
  : runtime_error(""), errors_(std::move(errors)) {
    compose_message();
  }

  const char *what() const noexcept override {
    return message_.c_str();
  }

  const std::vector<std::exception_ptr> &errors() const {
    return errors_;
  }

 private:
  void compose_message() {
    std::stringstream ss;
    ss << "Multiple exceptions:\n";
    for (const auto &e : errors_) {
      try {
        std::rethrow_exception(e);
      } catch (const std::exception &e) {
        ss << typeid(e).name() << ": " << e.what() << "\n";
      } catch (...) {
        ss << "Unknown exception\n";
      }
    }
    message_ = ss.str();
  }

  std::vector<std::exception_ptr> errors_;
  std::string message_;
};

}  // namespace dali

#endif  // DALI_CORE_MULTI_ERROR_H_
