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

#include <algorithm>
#include <iostream>
#include <sstream>
#include "dali/operators/reader/loader/utils.h"

namespace dali {

namespace {

static const std::vector<std::string> kSkipExtensions = {"..", "."};


inline std::string ListExtensions(const std::vector<std::string> &extensions) {
  std::stringstream ss;
  for (const auto &ext : extensions) {
    ss << ext << ", ";
  }
  return ss.str();
}

}  // namespace

bool HasExtension(std::string filepath, const std::vector<std::string> &extensions) {
  std::transform(filepath.begin(), filepath.end(), filepath.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  for (const auto &ext : kSkipExtensions) {
    if (ext == filepath) {
      return false;
    }
  }

  for (const auto &ext : extensions) {
    size_t pos = filepath.rfind(ext);
    if (pos != std::string::npos && pos + ext.length() == filepath.length()) {
      return true;
    }
  }

  std::cerr << "[Warning]: File " << filepath
            << " has extension that is not supported by the decoder. Supported extensions: "
            << ListExtensions(extensions) << std::endl;
  return false;
}


bool HasKnownExtension(const std::string &filepath) {
  std::vector<std::string> extensions;

  extensions.insert(extensions.end(), kKnownAudioExtensions.begin(), kKnownAudioExtensions.end());
  extensions.insert(extensions.end(), kKnownImageExtensions.begin(), kKnownImageExtensions.end());

  return HasExtension(filepath, extensions);
}

}  // namespace dali
