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
#include <vector>

#include "dali/test/dali_operator_test.h"

namespace dali {
namespace testing {
  

std::vector<testing::Arguments> cartesian(std::vector<testing::Arguments> args_vec) {
 return args_vec;
}

template <typename... Ts>
std::vector<testing::Arguments> cartesian(std::vector<testing::Arguments> args_vec, Ts... args_vecs) {
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

std::ostream& operator<<(std::ostream& os, const ArgumentKey& ak) {
  if (ak.node_name().empty()) {
    os << ak.arg_name();
  }
  else {
    os << ak.node_name() << ":" << ak.arg_name();
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const TestOpArg& op_arg) {
  os << op_arg.to_string();
  return os;
}

// JSON-like output
std::ostream& operator<<(std::ostream& os, const Arguments& args)
{
  std::string separator("");
  std::string indent("");
  if (args.size() == 0) {
    os << "{";
  }
  else if (args.size() == 1) {
    os << "{ ";
  }
  else {
    os << "{\n";
    separator = ",\n";
    indent = "    ";
  }

  for (const auto arg: args) {
    os << indent << "\"" << arg.first << "\" : " << arg.second << separator;
  }

  if (args.size() == 0) {
    os << "}";
  }
  else if (args.size() == 1) {
    os << " }";
  }
  else {
    os << "}\n";
  }

  return os;
}


// Force GTest to write our way
void PrintTo(const Arguments& args, std::ostream* os) {
  *os << args;
}


}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_DALI_OPERATOR_TEST_UTILS_H_