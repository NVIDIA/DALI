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

#ifndef DALI_INPUT_PROVIDER_H
#define DALI_INPUT_PROVIDER_H

#include <functional>
#include <vector>
#include "dali/core/small_vector.h"

namespace dali {

class InputProvider {
 public:
  using BatchSizeObserver = std::function<void(int /* batch_size */)>;
  void Notify(int batch_size) {
    for (auto& obs : batch_size_observers_) {
      obs(batch_size);
    }
  }
  void Register(BatchSizeObserver&& obs) {
    batch_size_observers_.emplace_back(std::forward<BatchSizeObserver>(obs));
  }

 private:
  SmallVector<BatchSizeObserver, 8> batch_size_observers_{};
};

}  // namespace dali

#endif  // DALI_INPUT_PROVIDER_H
