// Copyright (c) 2019, 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/format.h"

namespace dali {

TEST(MakeStringTest, no_delimiter) {
  auto str = make_string("jeden", 2, 3);
  ASSERT_EQ(str, "jeden23");
}

TEST(MakeStringTest, char_delimiter) {
  auto str = make_string_delim(' ', "jeden", 2, 3);
  ASSERT_EQ(str, "jeden 2 3");
}


TEST(MakeStringTest, string_delimiter) {
  auto str = make_string_delim("a custom delimiter", "jeden", 2, 3);
  ASSERT_EQ(str, "jedena custom delimiter2a custom delimiter3");
}


TEST(MakeStringTest, no_arguments) {
  auto str = make_string();
  ASSERT_EQ(str, "");
}


TEST(MakeStringTest, one_argument) {
  auto str = make_string("d[-_-]b");
  ASSERT_EQ(str, "d[-_-]b");
}


TEST(MakeStringTest, only_delimiter) {
  auto str = make_string_delim(">.<");
  ASSERT_EQ(str, "");
}


TEST(MakeStringTest, delimiter_and_one_argument) {
  auto str = make_string_delim("it really doesn't matter what's in here", "szalpal was here");
  ASSERT_EQ(str, "szalpal was here");
}


namespace {

std::string get_string(const std::ostream &os) {
  std::stringstream ss;
  ss << os.rdbuf();
  return ss.str();
}


bool stream_cmp(const std::ostream &lhs, const std::ostream &rhs) {
  return get_string(lhs) == get_string(rhs);
}

}  // namespace

TEST(PrintDelimTest, multiple_arguments) {
  std::stringstream ref_ss, in_ss;
  ref_ss << "a" << "," << "b";
  print_delim(in_ss, ",", "a", "b");
  ASSERT_PRED2(stream_cmp, ref_ss, in_ss);
}


TEST(PrintDelimTest, one_argument) {
  std::stringstream ref_ss, in_ss;
  ref_ss << "a";
  print_delim(in_ss, ",", "a");
  ASSERT_PRED2(stream_cmp, ref_ss, in_ss);
}


TEST(PrintDelimTest, only_delimiter) {
  std::stringstream ref_ss, in_ss;
  print_delim(in_ss, ",");
  ASSERT_PRED2(stream_cmp, ref_ss, in_ss);
}

TEST(JoinTest, CArray) {
  std::stringstream ss;
  int a1[1] = { 42 };
  join(ss, a1);
  EXPECT_EQ(ss.str(), "42");
  ss = {};
  float a3[]  = { 100, 42, 5 };
  join(ss, a3, no_delimiter());
  EXPECT_EQ(ss.str(), "100425");
  ss = {};
  join(ss, a3);
  EXPECT_EQ(ss.str(), "100, 42, 5");
  ss = {};
  join(ss, a3, "x");
  EXPECT_EQ(ss.str(), "100x42x5");
}

TEST(JoinTest, StdArray) {
  std::stringstream ss;
  std::array<int, 0> a0;
  join(ss, a0);
  EXPECT_EQ(ss.str(), "");
  ss = {};
  std::array<int, 1> a1 = { 42 };
  join(ss, a1);
  EXPECT_EQ(ss.str(), "42");
  ss = {};
  std::array<float, 3> a3  = { 100, 42, 5 };
  join(ss, a3, no_delimiter());
  EXPECT_EQ(ss.str(), "100425");
  ss = {};
  join(ss, a3, "x");
  EXPECT_EQ(ss.str(), "100x42x5");
}

TEST(JoinTest, Vector) {
  std::stringstream ss;
  std::vector<int> v;
  join(ss, v);
  EXPECT_EQ(ss.str(), "");
  ss = {};
  v.push_back(42);
  join(ss, v);
  EXPECT_EQ(ss.str(), "42");
  ss = {};
  v.push_back(100);
  v.push_back(66);
  join(ss, v);
  EXPECT_EQ(ss.str(), "42, 100, 66");
}

TEST(JoinTest, VectorPrint) {
  std::stringstream ss, ref_ss;
  std::vector<float> v = { -5, 1, 36.5 };
  ss << v;
  EXPECT_EQ(ss.str(), "-5, 1, 36.5");
}

}  // namespace dali
