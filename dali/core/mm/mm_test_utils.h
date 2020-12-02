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

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/detail/util.h"

namespace dali {
namespace mm {
namespace test {

template <bool owning, bool security_check,
          typename Base, typename Upstream, typename... ExtraParams>
class test_resource_wrapper : public Base {
  using sentinel_t = uint32_t;
  void *do_allocate(ExtraParams... extra, size_t bytes, size_t alignment) override {
    if (simulate_oom_)
      throw std::bad_alloc();
    void *ret;
    if (security_check) {
      ret = upstream_->allocate(extra..., bytes + sizeof(sentinel_t), alignment);
      detail::write_sentinel<sentinel_t>(ret, bytes);
    } else {
      ret = upstream_->allocate(extra..., bytes, alignment);
    }
    freed_.erase(ret);  // in case of address reuse
    if (ret) {  // nullptr can be returned on success if 0 bytes were requested
      auto it_ins = allocated_.insert({ret, { bytes, alignment }});
      if (!it_ins.second) {
        ASSERT_TRUE(!security_check && (bytes == 0 || it_ins.first->second.bytes == 0))
          << "The allocator returned the same address twice for non-empty allocations.", ret;
      }
      it_ins.first->second = { bytes, alignment };
    }
    return ret;
  }

  void do_deallocate(ExtraParams... extra, void *ptr, size_t bytes, size_t alignment) override {
    ASSERT_EQ(freed_.count(ptr), 0u) << "Double free of " << ptr;
    auto it = allocated_.find(ptr);
    ASSERT_NE(it, allocated_.end())
      << "The address " << ptr << " was not allocated by this allocator";
    ASSERT_EQ(it->second.bytes, bytes) << "Different size freed than allocated";
    ASSERT_EQ(it->second.alignment, alignment) << "Diffrent alignment freed than allocated";
    if (security_check) {
      ASSERT_TRUE(detail::check_sentinel<sentinel_t>(ptr, bytes)) << "Memory corruption detected";
    }
    freed_.insert(ptr);
    allocated_.erase(ptr);
    if (security_check)
      bytes += sizeof(sentinel_t);
    upstream_->deallocate(extra..., ptr, bytes, alignment);
  }

  struct alloc_params {
    size_t bytes = 0, alignment = 1;
  };
  std::unordered_map<void *, alloc_params> allocated_;
  std::unordered_set<void *> freed_;
  Upstream *upstream_ = nullptr;
  bool simulate_oom_ = false;

  bool do_is_equal(const Base &other) const noexcept override {
    if (auto *oth = dynamic_cast<const test_resource_wrapper*>(&other))
      return *upstream_ == *oth->upstream_;
    else
      return false;
  }

 public:
  test_resource_wrapper() = default;
  explicit test_resource_wrapper(Upstream *upstream) : upstream_(upstream) {}
  test_resource_wrapper(const test_resource_wrapper &) = delete;
  test_resource_wrapper(test_resource_wrapper &&) = delete;

  ~test_resource_wrapper() {
    reset();
  }

  void simulate_out_of_memory(bool enable) {
    simulate_oom_ = enable;
  }

  void reset(Upstream *upstream = nullptr) {
    if (upstream_) {
      for (auto &alloc : allocated_) {
        auto total_bytes = alloc.second.bytes;
        if (security_check)
          total_bytes += sizeof(sentinel_t);
        upstream_->deallocate(alloc.first, total_bytes, alloc.second.alignment);
      }
    }

    if (owning && upstream_)
      delete upstream_;
    upstream_ = upstream;
    allocated_.clear();
    freed_.clear();
  }


  void check_leaks() const {
    if (allocated_.empty())
      return;

    std::stringstream ss;
    ss << "Leaked blocks:\n";
    for (auto &alloc : allocated_) {
      ss << alloc.first << " :  " << alloc.second.bytes
         << " bytes, aligned to " << alloc.second.alignment << "\n";
    }
    GTEST_FAIL() << ss.str();
  }
};

struct test_host_resource
: public test_resource_wrapper<false, true, memory_resource, memory_resource> {
  test_host_resource() : test_resource_wrapper(&malloc_memory_resource::instance()) {}
};

template <typename Upstream>
using test_stream_resource = test_resource_wrapper<
    false, true, stream_memory_resource, Upstream, cudaStream_t>;

template <typename T>
void Fill(void *ptr, size_t bytes, T fill_pattern) {
  memcpy(ptr, &fill_pattern, std::min(bytes, sizeof(T)));
  size_t sz = sizeof(T);
  while (sz < bytes) {
    size_t next_sz = std::min(2 * sz, bytes);
    memcpy(static_cast<char*>(ptr) + sz, ptr, next_sz - sz);
    sz = next_sz;
  }
}

template <typename T>
void CheckFill(const void *ptr, size_t bytes, T fill_pattern) {
  size_t offset = 0;
  for (; offset + sizeof(T) <= bytes; offset += sizeof(T)) {
    ASSERT_EQ(std::memcmp(static_cast<const char*>(ptr) + offset, &fill_pattern, sizeof(T)), 0);
  }
  ASSERT_EQ(std::memcmp(static_cast<const char*>(ptr) + offset, &fill_pattern, bytes - offset), 0);
}

}  // namespace test
}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_MM_TEST_UTILS_H_
