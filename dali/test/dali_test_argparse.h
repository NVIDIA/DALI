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


#ifndef DALI_TEST_DALI_TEST_ARGPARSE_H_
#define DALI_TEST_DALI_TEST_ARGPARSE_H_

#include <boost/program_options.hpp>
#include <string>

namespace dali {
namespace testing {


namespace detail {

template<typename T>
void fill_argument(const boost::program_options::variables_map &vm,
                   const std::string &key, T &variable) {
  variable = vm[key].as<T>();
}


std::string kDaliExtraPath;  // NOLINT

}  // namespace detail

namespace program_options {

std::string dali_extra_path() {
  return dali::testing::detail::kDaliExtraPath;
}

}  // namespace program_options

void parse_args(int argc, char **argv) {
  namespace po = boost::program_options;

  po::options_description desc("");
  desc.add_options()
          ("dali_extra_path", po::value<std::string>(), "set path to dali_extra");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  detail::fill_argument(vm, "dali_extra_path", detail::kDaliExtraPath);
}

}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_ARGPARSE_H_
