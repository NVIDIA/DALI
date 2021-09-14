// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <type_traits>
#include "dali/core/mm/composite_resource.h"
#include "dali/core/mm/async_pool.h"
#include "dali/core/mm/malloc_resource.h"

namespace dali {
namespace mm {

static_assert(std::is_convertible<
    decltype(make_composite_resource(
        std::make_shared<async_pool_resource<memory_kind::device>>(), 42, "the answer"))*,
    async_memory_resource<memory_kind::device>*>::value,
    "A composite resource made from an async resource should have an async interface");

static_assert(std::is_convertible<
    decltype(make_shared_composite_resource(
        std::make_shared<malloc_memory_resource>(), "the answer", "is", 42).get()),
    memory_resource<memory_kind::host>*>::value,
    "A composite resource made from an sync resource should have a memory_resource interface");

static_assert(!std::is_convertible<
    decltype(make_composite_resource(
        std::make_shared<malloc_memory_resource>(), 42, "the answer"))*,
    async_memory_resource<memory_kind::host>*>::value,
    "A composite resource made from an sync resource should not have an async interface");

namespace test {

namespace {

struct IsAlive {
  IsAlive() {
    alive = true;
  }
  ~IsAlive() {
    alive = false;
  }
  bool alive;
};


template <typename Kind>
struct DummyResource : public async_memory_resource<Kind> {
  int allocate_seq = -1;
  int deallocate_seq = -1;
  int allocate_async_seq = -1;
  int deallocate_async_seq = -1;
  mutable int is_equal_seq = -1;

  mutable int seq = 0;
  IsAlive *alive = nullptr;

  DummyResource() = default;
  explicit DummyResource(IsAlive *alive) : alive(alive) {
  }

  ~DummyResource() {
    if (alive != nullptr) {
      EXPECT_TRUE(alive->alive);
    }
  }

  void *do_allocate(size_t, size_t) override {
    allocate_seq = ++seq;
    return nullptr;
  }

  void *do_allocate_async(size_t bytes, size_t alignment, stream_view stream) override {
    allocate_async_seq = ++seq;
    return nullptr;
  }

  void do_deallocate(void *, size_t, size_t) override {
    deallocate_seq = ++seq;
  }

  void do_deallocate_async(void *, size_t, size_t, stream_view) override {
    deallocate_async_seq = ++seq;
  }

  bool do_is_equal(const memory_resource<Kind> &other) const noexcept override {
    is_equal_seq = ++seq;
    return this == &other;
  }
};

}  // namespace


TEST(MMCompositeResource, InterfacePropagation) {
  auto rsrc = std::make_shared<DummyResource<memory_kind::device>>();
  auto cr = make_composite_resource(rsrc);
  cr.allocate(0, 0);
  cr.deallocate(nullptr, 0, 0);
  cr.allocate_async(0, 0, {});
  cr.deallocate_async(nullptr, 0, 0, {});
  EXPECT_TRUE(cr.is_equal(cr));
  EXPECT_EQ(rsrc->allocate_seq, 1);
  EXPECT_EQ(rsrc->deallocate_seq, 2);
  EXPECT_EQ(rsrc->allocate_async_seq, 3);
  EXPECT_EQ(rsrc->deallocate_async_seq, 4);
  EXPECT_EQ(rsrc->is_equal_seq, 5);
}

TEST(MMCompositeResource, Lifetime) {
  auto alive = std::make_shared<IsAlive>();
  auto rsrc = std::make_shared<DummyResource<memory_kind::host>>(alive.get());
  auto cr = make_shared_composite_resource(std::move(rsrc), std::move(alive));
  cr.reset();  // the GTEST conditions are in the resource destructor
}

}  // namespace test
}  // namespace mm
}  // namespace dali
