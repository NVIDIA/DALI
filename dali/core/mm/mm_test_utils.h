// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/async_pool.h"
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

  size_t alloc_curr_   = 0;
  size_t alloc_peak_   = 0;
  size_t num_allocs_   = 0;
  size_t num_deallocs_ = 0;

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
      alloc_curr_ += bytes;
      num_allocs_++;
      if (alloc_curr_ > alloc_peak_)
        alloc_peak_ = alloc_curr_;
    }
    return ret;
  }

  template <typename UpstreamDealloc, typename... Extra>
  void do_deallocate_impl(UpstreamDealloc &&ud, void *ptr, size_t bytes, size_t alignment,
                          Extra&&... extra) {
    ASSERT_GE(alloc_curr_, bytes);
    ASSERT_EQ(freed_.count(ptr), 0u) << "Double free of " << ptr;
    auto it = allocated_.find(ptr);
    ASSERT_NE(it, allocated_.end())
      << "The address " << ptr << " was not allocated by this allocator";
    ASSERT_EQ(it->second.bytes, bytes) << "Different size freed than allocated";
    ASSERT_EQ(it->second.alignment, alignment) << "Diffrent alignment freed than allocated";
    if (security_check) {
      ASSERT_TRUE(detail::check_sentinel<sentinel_t>(ptr, bytes)) << "Memory corruption detected";
    }
    num_deallocs_++;
    alloc_curr_ -= bytes;
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
  size_t get_peak_size() const {
    return alloc_peak_;
  }

  size_t get_current_size() const {
    return alloc_curr_;
  }

  size_t get_num_allocs() const {
    return num_allocs_;
  }

  size_t get_num_deallocs() const {
    return num_deallocs_;
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
    alloc_curr_ = 0;
    alloc_peak_ = 0;
    num_allocs_ = 0;
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

template <typename Kind, typename Context, bool owning, bool security_check,
          typename Upstream>
class test_resource_wrapper<owning, security_check, memory_resource<Kind, Context>, Upstream>
: public memory_resource<Kind, Context>
, public test_resource_wrapper_impl<owning, security_check, Upstream> {
  static_assert(!security_check || !std::is_same<Kind, mm::memory_kind::device>::value,
                "Cannot place a security cookie in device memory");

  using test_resource_wrapper_impl<owning, security_check, Upstream>::test_resource_wrapper_impl;
  bool do_is_equal(const memory_resource<Kind, Context> &other) const noexcept override {
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

  Context do_get_context() const noexcept override {
    return this->upstream_->get_context();
  }
};


template <typename Kind, bool owning, bool security_check,
          typename Upstream>
class test_resource_wrapper<owning, security_check, async_memory_resource<Kind>, Upstream>
: public async_memory_resource<Kind>
, public test_resource_wrapper_impl<owning, security_check, Upstream> {
  static_assert(!security_check || !std::is_same<Kind, mm::memory_kind::device>::value,
                "Cannot place a security cookie in device memory");

  using test_resource_wrapper_impl<owning, security_check, Upstream>::test_resource_wrapper_impl;
  bool do_is_equal(const memory_resource<Kind> &other) const noexcept override {
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

  void *do_allocate_async(size_t bytes, size_t alignment, stream_view strm_vw) override {
    return this->do_allocate_impl([&](size_t b, size_t a, stream_view sv) {
      return this->upstream_->allocate_async(b, a, sv);
    }, bytes, alignment, strm_vw);
  }

  void do_deallocate_async(
    void *ptr, size_t bytes, size_t alignment, stream_view strm_vw) override {
    return this->do_deallocate_impl([&](void *p, size_t b, size_t a, stream_view sv) {
      return this->upstream_->deallocate_async(p, b, a, sv);
    }, ptr, bytes, alignment, strm_vw);
  }

 public:
};

struct test_host_resource
: public test_resource_wrapper<false, true, host_memory_resource, host_memory_resource> {
  test_host_resource() : test_resource_wrapper(&malloc_memory_resource::instance()) {}
};

struct test_pinned_resource
: public test_resource_wrapper<false, true, pinned_memory_resource, pinned_memory_resource> {
  test_pinned_resource() : test_resource_wrapper(&upstream_instance()) {}

  static pinned_malloc_memory_resource &upstream_instance() {
    static pinned_malloc_memory_resource inst;
    return inst;
  }
};

struct test_device_resource
: public test_resource_wrapper<false, false,
  memory_resource<memory_kind::device>, memory_resource<memory_kind::device>> {
  test_device_resource() : test_resource_wrapper(&cuda_malloc_memory_resource::instance()) {}
};

template <typename Kind, bool owning, typename Upstream = async_memory_resource<Kind>>
using test_stream_resource = test_resource_wrapper<
    owning, detail::is_host_accessible<Kind>, async_memory_resource<Kind>, Upstream>;


class test_dev_pool_resource : public test_stream_resource<memory_kind::device, true> {
 public:
  test_dev_pool_resource() {
    using resource = async_pool_resource<mm::memory_kind::device>;
    reset(new resource(&cuda_malloc_memory_resource::instance()));
  }
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
