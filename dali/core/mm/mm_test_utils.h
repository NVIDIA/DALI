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
#include <dali/core/mm/mr.h>
#include <dali/core/mm/malloc_resource.h>
#include <dali/core/mm/detail/util.h>
#include <cstdlib>
#include <unordered_map>

namespace dali {
namespace mm {

inline bool is_aligned(void *ptr, size_t alignment) {
  return (reinterpret_cast<size_t>(ptr) & (alignment-1)) == 0;
}

template <bool owning, bool security_check = true>
class test_resource_wrapper : public memory_resource {
  using sentinel_t = uint32_t;
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (simulate_oom_)
      throw std::bad_alloc();
    void *ret;
    if (security_check) {
      ret = upstream_->allocate(bytes + sizeof(sentinel_t), alignment);
      detail::write_sentinel<sentinel_t>(ret, bytes);
    } else {
      ret = upstream_->allocate(bytes, alignment);
    }
    freed_.erase(ret);  // in case of address reuse
    if (ret) {  // nullptr can be returned on success if 0 bytes were requested
      auto it_ins = allocated_.insert({ret, { bytes, alignment }});
      if (!it_ins.second) {
        ASSERT_TRUE(!security_check && (bytes == 0 || it_ins.first->second.bytes == 0))
          << "The allocator returned the same address twice for non-empty allocations.", nullptr;
      }
      it_ins.first->second = { bytes, alignment };
    }
    return ret;
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    ASSERT_EQ(freed_.count(ptr), 0u) << "Double free of " << ptr;
    auto it = allocated_.find(ptr);
    ASSERT_NE(it, allocated_.end())
      << "The address " << ptr << " was not allocated by this allocator";
    ASSERT_EQ(it->second.bytes, bytes) << "Difreent size freed than allocated";
    ASSERT_EQ(it->second.alignment, alignment) << "Difreent alignment freed than allocated";
    if (security_check) {
      ASSERT_TRUE(detail::check_sentinel<sentinel_t>(ptr, bytes)) << "Memory corruption detected";
    }
    freed_.insert(ptr);
    allocated_.erase(ptr);
    if (security_check)
      bytes += sizeof(sentinel_t);
    upstream_->deallocate(ptr, bytes, alignment);
  }

  struct alloc_params {
    size_t bytes = 0, alignment = 1;
  };
  std::unordered_map<void *, alloc_params> allocated_;
  std::set<void *> freed_;
  memory_resource *upstream_ = nullptr;
  bool simulate_oom_ = false;

  bool do_is_equal(const memory_resource &other) const noexcept override {
    if (auto *oth = dynamic_cast<const test_resource_wrapper*>(&other))
      return *upstream_ == *oth->upstream_;
    else
      return false;
  }

 public:
  test_resource_wrapper() = default;
  explicit test_resource_wrapper(memory_resource *upstream) : upstream_(upstream) {}
  test_resource_wrapper(const test_resource_wrapper &) = delete;
  test_resource_wrapper(test_resource_wrapper &&) = delete;

  ~test_resource_wrapper() {
    reset();
  }

  void simulate_out_of_memory(bool enable) {
    simulate_oom_ = enable;
  }

  void reset(memory_resource *upstream = nullptr) {
    if (upstream_) {
      for (auto &alloc : allocated_) {
        deallocate(alloc.first, alloc.second.bytes, alloc.second.alignment);
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

struct test_host_resource : public test_resource_wrapper<false, true> {
  test_host_resource() : test_resource_wrapper(&malloc_memory_resource::instance()) {}
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_MM_TEST_UTILS_H_
