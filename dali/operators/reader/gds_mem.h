// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_GDS_MEM_H_
#define DALI_OPERATORS_READER_GDS_MEM_H_

#include <memory>
#include "dali/core/mm/memory.h"

namespace dali {

class GDSAllocator {
 public:
  explicit GDSAllocator(int device_id = -1);

  mm::memory_resource<mm::memory_kind::device> *resource() const {
    return rsrc_.get();
  }

  static GDSAllocator &instance(int device_id = -1);

 private:
  std::shared_ptr<mm::memory_resource<mm::memory_kind::device>> rsrc_;
};

size_t GetGDSChunkSize();

inline std::shared_ptr<uint8_t> gds_alloc(size_t bytes) {
    return mm::alloc_raw_shared<uint8_t>(GDSAllocator::instance().resource(), bytes);
}

}  // namespace dali

#endif  // DALI_OPERATORS_READER_GDS_MEM_H_
