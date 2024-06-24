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
#include <cstring>
#include <string>
#include <filesystem>
#include "dali/util/uri.h"

namespace dali {
namespace filesystem {

std::string join_path(const std::string &dir, const std::string &path) {
  if (dir.empty())
    return path;
  if (path.empty())
    return dir;

  auto uri = URI::Parse(dir, URI::ParseOpts::AllowNonEscaped);
  if (uri.valid()) {
    const char *separators = "/";
    // TODO(janton): In case we ever support Windows
#ifdef _WINVER
    if (uri.scheme() == "file")
      separators = "/\\";
#endif

    if (strchr(separators, path[0]))  // absolute path
      return std::string(uri.scheme_authority()) + path;
    else if (strchr(separators, dir[dir.length() - 1]))  // dir ends with a separator
      return dir + path;
    else  // basic case
      return dir + '/' + path;
  }
  return std::filesystem::path(dir) / std::filesystem::path(path);
}

}  // namespace filesystem
}  // namespace dali
