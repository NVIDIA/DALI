// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_FILESYSTEM_H_
#define DALI_OPERATORS_READER_LOADER_FILESYSTEM_H_

#include <string>
#include <utility>
#include "dali/core/common.h"

namespace dali {
namespace filesystem {

/**
 * @brief Prepends dir to a relative path and keeps absolute path unchanged.
 */
DLL_PUBLIC string join_path(const std::string &dir, const std::string &path);

#ifdef WINVER
constexpr char dir_sep = '\\';
#else
constexpr char dir_sep = '/';
#endif

}  // namespace filesystem
}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FILESYSTEM_H_
