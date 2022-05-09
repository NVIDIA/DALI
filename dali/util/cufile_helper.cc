// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <mutex>
#include <vector>
#include "dali/core/spinlock.h"
#include "dali/util/cufile_helper.h"

using namespace dali;  // NOLINT
namespace cufile {

namespace {

struct CUFileHandleInstance {
  std::weak_ptr<CUFileDriverHandle> wptr;
  std::mutex lock;
  std::shared_ptr<CUFileDriverHandle> get() {
    auto sptr = wptr.lock();
    if (sptr)
      return sptr;
    std::lock_guard g(lock);
    sptr = wptr.lock();  // double check under lock
    if (sptr)
      return sptr;
    sptr = std::make_shared<CUFileDriverHandle>(-1);
    return sptr;
  }
};

}  // namespace

std::shared_ptr<CUFileDriverHandle> CUFileDriverHandle::Get() {
  static CUFileHandleInstance instance;
  instance.get();
}

}  // namespace cufile
