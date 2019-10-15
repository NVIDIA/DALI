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


#ifndef DALI_OPERATORS_UTIL_RANDOMIZER_H_
#define DALI_OPERATORS_UTIL_RANDOMIZER_H_

#include "dali/pipeline/data/backend.h"

namespace dali {

template <typename Backend>
class Randomizer {
 public:
  explicit Randomizer(int seed = 1234, size_t len = 128*32*32);

#if __CUDA_ARCH__
  __device__
#endif
  int rand(int idx);

  void Cleanup();

 private:
    void *states_;
    size_t len_;
    int device_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_RANDOMIZER_H_
