// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/core/small_vector.h"

namespace dali {

static_assert(SmallVector<int, 6>::static_size == 6, "Static doesn's match template argument");
static_assert(sizeof(SmallVector<int, 7>) >= 7*sizeof(int) + sizeof(size_t),
  "SmallVector must be large enough to house its data items and size");

namespace {
struct TestObj {
  TestObj() {
    total++;
  }
  TestObj(int value) : value(value) {  // NOLINT
    total++;
  }
  ~TestObj() {
    total--;
    if (zombie) {
      zombies--;
    }
  }
  TestObj(const TestObj &other) {
    value = other.value;
    total++;
    if ((zombie = other.zombie))
      zombies++;
  }
  TestObj(TestObj &&other) {
    value = other.value;
    total++;
    other.zombie = true;
    zombies++;
  }
  TestObj &operator=(const TestObj &other) {
    value = other.value;
    if (other.zombie && !other.zombie)
      zombies++;
    if (zombie && !other.zombie)
      zombies--;
    zombie = other.zombie;
    return *this;
  }
  TestObj &operator=(TestObj &&other) {
    value = other.value;
    if (!zombie)
      zombies++;
    zombie = other.zombie;
    other.zombie = true;
    return *this;
  }

  bool operator==(const TestObj &other) const {
    return value == other.value;
  }

  int value = 0;
  bool zombie = false;
  static size_t total, zombies;
};
size_t TestObj::total = 0;
size_t TestObj::zombies = 0;
}  // namespace

TEST(TestObj, RefCount) {
  EXPECT_EQ(TestObj::total, 0);
  EXPECT_EQ(TestObj::zombies, 0);
  {
    TestObj a, b, c;
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(TestObj::zombies, 0);
    a = b;
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(TestObj::zombies, 0);
    a = std::move(b);
    {
      TestObj d = std::move(c);
      EXPECT_EQ(TestObj::total, 4);
      EXPECT_EQ(TestObj::zombies, 2);
    }
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(TestObj::zombies, 2);
    b = std::move(a);
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(TestObj::zombies, 2);
    b = std::move(a);
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(TestObj::zombies, 3);
    b = TestObj();
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(TestObj::zombies, 2);
    a = c = b;
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(TestObj::zombies, 0);
  }
  EXPECT_EQ(TestObj::total, 0);
  EXPECT_EQ(TestObj::zombies, 0);
}

TEST(SmallVector, Static) {
  {
    SmallVector<TestObj, 5> a;
    EXPECT_EQ(a.capacity(), 5);
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);
    a.push_back(4);
    a.push_back(5);
    EXPECT_EQ(a.size(), 5);
    EXPECT_EQ(a.capacity(), 5);
    EXPECT_FALSE(a.is_dynamic());
    for (int i = 0; i < 5; i++)
      EXPECT_EQ(a[i].value, i+1);
    EXPECT_EQ(TestObj::total, 5);
  }
  EXPECT_EQ(TestObj::total, 0);
}

TEST(SmallVector, Dynamic) {
  {
    SmallVector<TestObj, 3> a;
    EXPECT_EQ(a.capacity(), 3);
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);
    EXPECT_FALSE(a.is_dynamic());
    EXPECT_EQ(TestObj::total, 3);
    a.push_back(4);
    EXPECT_EQ(TestObj::total, 4);
    EXPECT_EQ(a.size(), 4);
    EXPECT_EQ(a.capacity(), 6);
    EXPECT_TRUE(a.is_dynamic());
    a.push_back(5);
    EXPECT_EQ(a.size(), 5);

    EXPECT_EQ(TestObj::total, 5);
  }
  EXPECT_EQ(TestObj::total, 0);
}

TEST(SmallVector, InsertNoRealloc) {
  {
    SmallVector<TestObj, 3> a;
    a.push_back(1);
    a.push_back(3);
    a.insert_at(1, 2);
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(a[0].value, 1);
    EXPECT_EQ(a[1].value, 2);
    EXPECT_EQ(a[2].value, 3);
  }
  EXPECT_EQ(TestObj::total, 0);
}

TEST(SmallVector, InsertRealloc) {
  {
    SmallVector<TestObj, 3> a;
    a.push_back(1);
    a.push_back(3);
    a.push_back(4);
    ASSERT_EQ(a.size(), 3);
    ASSERT_EQ(a.capacity(), 3);
    a.insert_at(1, 2);
    EXPECT_TRUE(a.is_dynamic());
    EXPECT_EQ(TestObj::zombies, 0);
    EXPECT_EQ(TestObj::total, 4);
    EXPECT_EQ(a[0].value, 1);
    EXPECT_EQ(a[1].value, 2);
    EXPECT_EQ(a[2].value, 3);
    EXPECT_EQ(a[3].value, 4);
  }
  EXPECT_EQ(TestObj::total, 0);
}

TEST(SmallVector, MultipleInsert_PoD) {
  dali::SmallVector<int, 3> v;
  EXPECT_EQ(v.capacity(), 3);
  EXPECT_EQ(v.size(), 0);
  v.push_back(1);
  v.push_back(3);
  v.push_back(5);
  EXPECT_FALSE(v.is_dynamic());
  v.push_back(7);
  EXPECT_TRUE(v.is_dynamic());
  v.insert(v.begin() + 1, 2);
  v.insert(v.begin() + 3, 4);
  v.insert(v.begin() + 5, 6);
  v.insert(v.begin() + 7, 8);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
  EXPECT_EQ(v[3], 4);
  EXPECT_EQ(v[4], 5);
  EXPECT_EQ(v[5], 6);
  EXPECT_EQ(v[6], 7);
  EXPECT_EQ(v[7], 8);
  v.erase(v.begin()+2, v.end()-2);
  ASSERT_EQ(v.size(), 4);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 7);
  EXPECT_EQ(v[3], 8);
}

TEST(SmallVector, PopBack) {
  dali::SmallVector<int, 3> v;
  v.push_back(1);
  v.pop_back();
  EXPECT_TRUE(v.empty());
  v.push_back(2);
  EXPECT_EQ(v.back(), 2);
  v.push_back(3);
  v.push_back(4);
  EXPECT_FALSE(v.is_dynamic());
  EXPECT_EQ(v.back(), 4);
  v.pop_back();
  EXPECT_EQ(v.back(), 3);
  v.push_back(5);
  EXPECT_FALSE(v.is_dynamic());
  v.push_back(6);
  EXPECT_EQ(v.back(), 6);
  EXPECT_TRUE(v.is_dynamic());
  v.pop_back();
  EXPECT_EQ(v.back(), 5);
}

template <typename T, typename U>
inline void EXPECT_VEC_EQUAL(const T &a, const U &b) {
  auto it_a = a.begin();
  auto it_b = b.begin();
  for (size_t i = 0; it_a != a.end() && it_b != b.end(); ++it_a, ++it_b, ++i) {
    EXPECT_EQ(*it_a, *it_b) << " difference at index " << i;
  }
  EXPECT_EQ(it_a, a.end()) << "`a` is longer";
  EXPECT_EQ(it_b, b.end()) << "`b` is longer";
}

TEST(SmallVector, MovePoD) {
  SmallVector<int, 4> a, b;
  a.push_back(1);
  a.push_back(2);
  b.push_back(3);
  b = std::move(a);
  EXPECT_EQ(b[0], 1);
  EXPECT_EQ(b[1], 2);
  EXPECT_TRUE(a.empty());
  b.push_back(3);
  b.push_back(4);
  b.push_back(5);
  EXPECT_TRUE(b.is_dynamic());
  auto *ptr = b.data();
  a = std::move(b);
  EXPECT_EQ(a.data(), ptr);
  EXPECT_TRUE(b.empty());
}

TEST(SmallVector, Move) {
  {
    SmallVector<TestObj, 3> a, b;
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);
    b = std::move(a);
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(TestObj::zombies, 0);
    a = std::move(b);
    EXPECT_EQ(TestObj::total, 3);
    EXPECT_EQ(TestObj::zombies, 0);
    EXPECT_TRUE(b.empty());

    a.push_back(4);
    b.push_back(11);
    b.push_back(12);
    EXPECT_EQ(TestObj::total, 6);
    auto *ptr = a.data();
    b = std::move(a);
    EXPECT_EQ(b.data(), ptr) << "Pointer from dynamic vector should have been moved";
    EXPECT_EQ(TestObj::total, 4);
    EXPECT_TRUE(b.is_dynamic());
    EXPECT_FALSE(a.is_dynamic());

    a.push_back(11);
    a.push_back(12);
    EXPECT_FALSE(a.is_dynamic());
    b = std::move(a);
    EXPECT_FALSE(b.is_dynamic());
    a.clear();
    b.clear();
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);
    a.push_back(4);

    SmallVector<TestObj, 16> d = std::move(a);
    EXPECT_FALSE(d.is_dynamic());
    EXPECT_EQ(TestObj::total, 4);
  }
  EXPECT_EQ(TestObj::total, 0);
}


TEST(SmallVector, Copy) {
  {
    SmallVector<TestObj, 3> a, b;
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);
    a.push_back(4);
    b.push_back(11);
    b.push_back(12);
    EXPECT_EQ(TestObj::total, 6);
    EXPECT_TRUE(a.is_dynamic());
    EXPECT_FALSE(b.is_dynamic());
    a = b;
    EXPECT_VEC_EQUAL(a, b);
    EXPECT_EQ(TestObj::total, 4);
    a.push_back(13);
    a.push_back(14);
    EXPECT_EQ(TestObj::total, 6);
    b = a;
    EXPECT_VEC_EQUAL(a, b);
    EXPECT_EQ(TestObj::total, 8);

    SmallVector<TestObj, 16> c = a;
    EXPECT_FALSE(c.is_dynamic());
    EXPECT_VEC_EQUAL(a, c);
  }
  EXPECT_EQ(TestObj::total, 0);
}

TEST(SmallVector, Erase) {
  SmallVector<TestObj, 3> a;
  a.push_back(1);
  a.push_back(2);
  a.push_back(3);
  EXPECT_EQ(TestObj::total, 3);
  auto it = a.erase(a.begin() + 1);
  EXPECT_EQ(a.size(), 2);
  EXPECT_EQ(TestObj::total, 2);
  EXPECT_EQ(*it, 3);
  a.push_back(4);
  EXPECT_FALSE(a.is_dynamic());
  a.push_back(5);
  EXPECT_TRUE(a.is_dynamic());
  a.erase(a.begin() + 1, a.end()-1);
  EXPECT_EQ(a.size(), 2);
  EXPECT_EQ(a[0], 1);
  EXPECT_EQ(a[1], 5);
  a.erase(a.begin(), a.end());
  EXPECT_TRUE(a.empty());
  EXPECT_EQ(TestObj::total, 0);
}

TEST(SmallVector, Resize) {
  SmallVector<TestObj, 4> v;
  v.resize(3, 5);
  ASSERT_EQ(v.size(), 3);
  EXPECT_EQ(v[0], 5);
  EXPECT_EQ(v[1], 5);
  EXPECT_EQ(v[2], 5);
  EXPECT_EQ(TestObj::total, 3);
  v.resize(16, 42);
  ASSERT_EQ(v.size(), 16);
  EXPECT_EQ(TestObj::total, 16);
  EXPECT_EQ(TestObj::zombies, 0);
  for (int i = 3; i < 16; i++)
    EXPECT_EQ(v[i], 42);
  v.resize(6);
  EXPECT_EQ(v.size(), 6);
  EXPECT_EQ(TestObj::total, 6);
  EXPECT_EQ(TestObj::zombies, 0);
}

TEST(SmallVector, FromVector) {
  std::vector<TestObj> vec = { 1, 2, 3, 4 };
  SmallVector<TestObj, 3> v1 = vec;
  SmallVector<TestObj, 5> v2;
  ASSERT_EQ(v1.size(), 4);
  EXPECT_TRUE(v1.is_dynamic());
  EXPECT_EQ(v1[0], 1);
  EXPECT_EQ(v1[1], 2);
  EXPECT_EQ(v1[2], 3);
  EXPECT_EQ(v1[3], 4);
  EXPECT_EQ(TestObj::total, 8);
  EXPECT_EQ(TestObj::zombies, 0);

  v2.push_back(666);
  v2 = vec;
  EXPECT_FALSE(v2.is_dynamic());
  ASSERT_EQ(v2.size(), 4);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2[2], 3);
  EXPECT_EQ(v2[3], 4);
  EXPECT_EQ(TestObj::total, 12);
  EXPECT_EQ(TestObj::zombies, 0);
}

TEST(SmallVector, InitList) {
  SmallVector<TestObj, 3> v = { 1, 2, 3 };
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
  EXPECT_EQ(TestObj::total, 3);
  EXPECT_EQ(TestObj::zombies, 0);
}

TEST(SmallVector, ToVector) {
  SmallVector<TestObj, 3> sv = { 1, 2, 3 };
  std::vector<TestObj> vec = sv.to_vector();
  EXPECT_EQ(vec[0], 1);
  EXPECT_EQ(vec[1], 2);
  EXPECT_EQ(vec[2], 3);
  EXPECT_EQ(TestObj::total, 6);
  EXPECT_EQ(TestObj::zombies, 0);
}

}  // namespace dali
