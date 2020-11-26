// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <string>

#include "dali/util/cufile.h"
#include "dali/util/std_cufile.h"

namespace dali {

std::unique_ptr<CUFileStream> CUFileStream::Open(const std::string& uri, bool read_ahead,
                                                 bool use_mmap) {
  std::string processed_uri;

  const char prefix[] = "file://";
  if (!strncmp(uri.c_str(), prefix, sizeof(prefix) - 1)) {
    processed_uri = uri.substr(sizeof(prefix) - 1);
  } else {
    processed_uri = uri;
  }

  DALI_ENFORCE(!use_mmap, "mmap not implemented with cuFile yet.");
  return std::make_unique<StdCUFileStream>(processed_uri);
}

}  // namespace dali
