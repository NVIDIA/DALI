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

#include "dali/operators/reader/loader/filesystem.h"

namespace dali {
namespace filesystem {

std::string join_path(const std::string &dir, const std::string &path) {
  if (dir.empty())
    return path;
  if (path.empty())
    return dir;

  char separator = dir_sep;
#ifdef WINVER
  // If an URI, use slash as a separator
  if (URI::Parse(dir).valid)
    separator = '/';
#endif

  if (path[0] == separator)  // absolute path
    return path;
#ifdef WINVER
  if (path[1] == ':')
    return path;
#endif
  if (dir[dir.length() - 1] == separator)
    return dir + path;
  else
    return dir + separator + path;
}

}  // namespace filesystem
}  // namespace dali
