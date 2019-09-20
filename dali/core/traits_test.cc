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

#include <list>
#include <map>
#include <deque>
#include <set>
#include <queue>
#include <stack>
#include <array>
#include "dali/core/traits.h"

namespace dali {

namespace {

using T = int;

}  // namespace

static_assert(is_container<std::vector<T>>::value, "");
static_assert(is_container<std::list<T>>::value, "");
static_assert(is_container<std::map<T, T>>::value, "");
static_assert(is_container<std::deque<T>>::value, "");
static_assert(is_container<std::set<T>>::value, "");

static_assert(!is_container<int>::value, "");
static_assert(!is_container<void>::value, "");
static_assert(!is_container<std::stack<T>>::value, "");
static_assert(!is_container<std::array<T, 42>>::value,
        "std::array is not regarded as a container, since it doesn't have custom allocator");


}  // namespace dali
