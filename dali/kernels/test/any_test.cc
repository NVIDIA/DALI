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
#include "dali/core/any.h"

namespace dali {

TEST(Any, AnyCast) {
  any a = 5;
  EXPECT_EQ(any_cast<int>(a), 5);
  EXPECT_THROW(any_cast<int8_t>(a), bad_any_cast);
  any_cast<int&>(a) = 7;
  EXPECT_EQ(any_cast<int>(a), 7);
  EXPECT_EQ(&any_cast<const int&>(a), &any_cast<int&>(a));
  const any *ca = &a;
  EXPECT_EQ(any_cast<int>(&a), any_cast<int>(ca));
  *any_cast<int>(&a) = 9;
  EXPECT_EQ(any_cast<int&>(a), 9);
  EXPECT_EQ(any_cast<const int&>(a), 9);
  EXPECT_EQ(any_cast<int&&>(std::move(a)), 9);
}

struct test_string {
  test_string(size_t *track, std::string str) : track(track), str(std::move(str)) {
    (*track)++;
  }
  test_string(const test_string &other) : track(other.track), str(other.str) {
    (*track)++;
  }
  test_string(test_string &&other) : track(other.track), str(std::move(other.str)) {
    (*track)++;
  }
  ~test_string() {
    (*track)--;
  }

  test_string &operator=(const test_string &other) {
    if (track) (*track)--;
    track = other.track;
    (*track)++;
    str = other.str;
    return *this;
  }

  test_string &operator=(test_string &&other) {
    if (track) (*track)--;
    track = other.track;
    (*track)++;
    str = std::move(other.str);
    return *this;
  }

  size_t *track = nullptr;
  std::string str;
};

TEST(Any, Assign) {
  size_t track = 0;

  {
    any a = 5;
    EXPECT_EQ(any_cast<int>(a), 5);
    any_cast<int&>(a) = 7;
    EXPECT_EQ(any_cast<int>(a), 7);
    float fp = 9;
    a = fp;
    EXPECT_EQ(any_cast<float>(a), 9);

    EXPECT_THROW(any_cast<int>(a), bad_any_cast);

    test_string s(&track, "hello, world");
    EXPECT_EQ(track, 1) << "Missing initial instance count";
    a = s;
    EXPECT_EQ(track, 2) << "'a' should count as an instance";

    any b = a;
    EXPECT_EQ(track, 3) << "instances 's', 'a', 'b' should be counted";

    EXPECT_EQ(any_cast<test_string>(a).str, "hello, world");
    EXPECT_EQ(any_cast<test_string>(b).str, "hello, world");
    const char *ptr = any_cast<test_string&>(b).str.data();
    any c = std::move(b);
    EXPECT_FALSE(b.has_value());
    EXPECT_EQ(track, 3) << "instances 's', 'a', 'c' should be counted; b is cleared";

    EXPECT_EQ(any_cast<test_string>(c).str, "hello, world");
    EXPECT_EQ(any_cast<test_string&>(c).str.data(), ptr) << "Data not moved properly";

    any d;
    EXPECT_FALSE(d.has_value());
    d = c;
    EXPECT_EQ(track, 4) << "valid instances: 's', 'a', 'c', 'd'";
    EXPECT_EQ(any_cast<test_string>(d).str, "hello, world");
    any e;
    d = e;
    EXPECT_FALSE(d.has_value());
    EXPECT_EQ(track, 3) << "valid instances: 's', 'a', 'c'";

    any f = c;
    EXPECT_EQ(track, 4) << "valid instances: 's', 'a', 'c', 'f'";
    f = std::move(a);
    EXPECT_FALSE(a.has_value());
    EXPECT_EQ(track, 3) << "valid instances: 's', 'c', 'f'";

    f = 42;
    EXPECT_EQ(track, 2) << "valid instances: 's', 'c'";
  }
  EXPECT_EQ(track, 0);
}

TEST(Any, MakeAny) {
  auto a = make_any<std::string>("test string");
  EXPECT_EQ(any_cast<std::string>(a), "test string");

  a = make_any<std::vector<int>>({ 3, 4, 5, 6, 7});
  std::vector<int> vec = { 3, 4, 5, 6, 7};
  EXPECT_EQ(any_cast<std::vector<int>>(a), vec);
}

}  // namespace dali

