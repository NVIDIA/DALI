// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef DALI_CORE_STRING_MAP_H_
#define DALI_CORE_STRING_MAP_H_

#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>

namespace dali {

struct string_hash : public std::hash<std::string_view> {
    using is_transparent = void;
};

template <typename Value>
using unordered_string_map = std::unordered_map<std::string, Value, string_hash, std::equal_to<>>;

template <typename Value>
using string_map = std::map<std::string, Value, std::less<>>;

}  // namespace dali

#endif  // DALI_CORE_STRING_MAP_H_
