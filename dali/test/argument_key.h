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

#ifndef DALI_OPERATOR_ARGUMENTS_H
#define DALI_OPERATOR_ARGUMENTS_H

#include <utility>
#include <string>


namespace dali {
namespace testing {

class ArgumentKey : public std::pair<std::string, std::string> {
 public:
  using Base = std::pair<std::string, std::string>;


  ArgumentKey(const char* arg_name) : Base({}, arg_name) {}


  ArgumentKey(std::string node_name, std::string arg_name) :
          Base(std::move(node_name), std::move(arg_name)) {

  }
};

}  // namespace testing
}  // namespace dali

#endif  // DALI_OPERATOR_ARGUMENTS_H
