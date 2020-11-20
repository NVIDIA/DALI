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

#ifndef DALI_CORE_MM_MM_TEST_UTILS_H_
#define DALI_CORE_MM_MM_TEST_UTILS_H_

namespace dali {

inline bool is_aligned(void *ptr, size_t alignment) {
  return (reinterpret_cast<size_t>(ptr) & (alignment-1)) == 0;
}

}  // namespace dali

#endif  // DALI_CORE_MM_MM_TEST_UTILS_H_
