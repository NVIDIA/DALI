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
#include <utility>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/detail/util.h"

namespace dali {
namespace mm {
namespace test {

template <bool owning, bool security_check,
          typename Base, typename Upstream>
class test_resource_wrapper;

template <bool owning, bool security_check, typename Upstream>
class test_resource_wrapper_impl {
 protected:
  using sentinel_t = uint32_t;

  template <typename UpstreamAlloc, typename... Extra>
  void *do_allocate_impl(UpstreamAlloc &&ua, size_t bytes, size_t alignment, Extra&&... extra) {
    if (simulate_oom_)
      throw std::bad_alloc();
    void *ret;
    if (security_check) {
      ret = ua(bytes + sizeof(sentinel_t), alignment, std::forward<Extra>(extra)...);
      detail::write_sentinel<sentinel_t>(ret, bytes);
    } else {
      ret = ua(bytes, alignment, std::forward<Extra>(extra)...);
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

  template <typename UpstreamDealloc, typename... Extra>
  void do_deallocate_impl(UpstreamDealloc &&ud, void *ptr, size_t bytes, size_t alignment,
                          Extra&&... extra) {
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
    ud(ptr, bytes, alignment, std::forward<Extra>(extra)...);
  }

  struct alloc_params {
    size_t bytes = 0, alignment = 1;
  };
  std::unordered_map<void *, alloc_params> allocated_;
  std::unordered_set<void *> freed_;
  Upstream *upstream_ = nullptr;
  bool simulate_oom_ = false;

  test_resource_wrapper_impl() = default;
  explicit test_resource_wrapper_impl(Upstream *upstream) : upstream_(upstream) {}
  test_resource_wrapper_impl(const test_resource_wrapper_impl &) = delete;
  test_resource_wrapper_impl(test_resource_wrapper_impl &&) = delete;

  ~test_resource_wrapper_impl() {
    reset();
  }

 public:
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

template <memory_kind kind, allocation_order order, bool owning, bool security_check,
          typename Upstream>
class test_resource_wrapper<owning, security_check, memory_resource<kind, order>, Upstream>
: public memory_resource<kind, order>
, public test_resource_wrapper_impl<owning, security_check, Upstream> {
  static_assert(!security_check || kind != memory_kind::device,
                "Cannot place a security cookie in device memory");

  using test_resource_wrapper_impl<owning, security_check, Upstream>::test_resource_wrapper_impl;
  bool do_is_equal(const memory_resource<kind, order> &other) const noexcept override {
    if (auto *oth = dynamic_cast<const test_resource_wrapper*>(&other))
      return this->upstream_->is_equal(*oth->upstream_);
    else
      return false;
  }

  void *do_allocate(size_t bytes, size_t alignment) override {
    return this->do_allocate_impl([&](size_t b, size_t a) {
      return this->upstream_->allocate(b, a);
    }, bytes, alignment);
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    return this->do_deallocate_impl([&](void *p, size_t b, size_t a) {
      return this->upstream_->deallocate(p, b, a);
    }, ptr, bytes, alignment);
  }
};


template <memory_kind kind, bool owning, bool security_check,
          typename Upstream>
class test_resource_wrapper<owning, security_check, stream_aware_memory_resource<kind>, Upstream>
: public stream_aware_memory_resource<kind>
, public test_resource_wrapper_impl<owning, security_check, Upstream> {
  static_assert(!security_check || kind != memory_kind::device,
                "Cannot place a security cookie in device memory");

  using test_resource_wrapper_impl<owning, security_check, Upstream>::test_resource_wrapper_impl;
  bool do_is_equal(const memory_resource<kind> &other) const noexcept override {
    if (auto *oth = dynamic_cast<const test_resource_wrapper*>(&other))
      return this->upstream_->is_equal(*oth->upstream_);
    else
      return false;
  }

  void *do_allocate(size_t bytes, size_t alignment) override {
    return this->do_allocate_impl([&](size_t b, size_t a) {
      return this->upstream_->allocate(b, a);
    }, bytes, alignment);
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    return this->do_deallocate_impl([&](void *p, size_t b, size_t a) {
      return this->upstream_->deallocate(p, b, a);
    }, ptr, bytes, alignment);
  }

  void *do_allocate_async(size_t bytes, size_t alignment, stream_view sv) override {
    return this->do_allocate_impl([&](size_t b, size_t a, stream_view sv) {
      return this->upstream_->allocate_async(b, a, sv);
    }, bytes, alignment, sv);
  }

  void do_deallocate_async(void *ptr, size_t bytes, size_t alignment, stream_view sv) override {
    return this->do_deallocate_impl([&](void *p, size_t b, size_t a, stream_view sv) {
      return this->upstream_->deallocate_async(p, b, a, sv);
    }, ptr, bytes, alignment, sv);
  }
};

struct test_host_resource
: public test_resource_wrapper<false, true, host_memory_resource, host_memory_resource> {
  test_host_resource() : test_resource_wrapper(&malloc_memory_resource::instance()) {}
};

struct test_device_resource
: public test_resource_wrapper<false, false,
  memory_resource<memory_kind::device>, memory_resource<memory_kind::device>> {
  test_device_resource() : test_resource_wrapper(&cuda_malloc_memory_resource::instance()) {}
};

template <memory_kind kind, bool owning, typename Upstream = stream_aware_memory_resource<kind>>
using test_stream_resource = test_resource_wrapper<
    owning, detail::is_host_memory(kind), stream_aware_memory_resource<kind>, Upstream>;


class test_dev_pool_resource : public test_stream_resource<memory_kind::device, true> {
 public:
  test_dev_pool_resource() {
    reset(new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(&cuda_mr_));
  }

 private:
  rmm::mr::cuda_memory_resource cuda_mr_;
};


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
