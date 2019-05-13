// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/common.h"

namespace dali {

std::vector<std::string> string_split(const std::string &s, const char delim) {
    std::vector<std::string> ret;
    size_t pos = 0;
    while (pos != std::string::npos) {
        size_t newpos = s.find(delim, pos);
        ret.push_back(s.substr(pos, newpos - pos));
        if (newpos != std::string::npos) {
            pos = newpos + 1;
        } else {
            pos = newpos;
        }
    }
    return ret;
}

}  // namespace dali
