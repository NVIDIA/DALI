// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/util/uri.h"
#include <gtest/gtest.h>

namespace dali {

TEST(URI, Parse_1) {
  auto uri = URI::Parse(
    "https://john.doe@www.example.com:123/forum/questions/?tag=networking&order=newest#top");
  EXPECT_EQ("https", uri.scheme());
  EXPECT_EQ("john.doe@www.example.com:123", uri.authority());
  EXPECT_EQ("/forum/questions/", uri.path());
  EXPECT_EQ("tag=networking&order=newest", uri.query());
  EXPECT_EQ("/forum/questions/?tag=networking&order=newest", uri.path_and_query());
  EXPECT_EQ("top", uri.fragment());
}

TEST(URI, Parse_2) {
  auto uri = URI::Parse(
    "ldap://[2001:db8::7]/c=GB?objectClass?one");
  EXPECT_EQ("ldap", uri.scheme());
  EXPECT_EQ("[2001:db8::7]", uri.authority());
  EXPECT_EQ("/c=GB", uri.path());
  EXPECT_EQ("objectClass?one", uri.query());
  EXPECT_EQ("/c=GB?objectClass?one", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_3) {
  auto uri = URI::Parse(
    "mailto:John.Doe@example.com");
  EXPECT_EQ("mailto", uri.scheme());
  EXPECT_EQ("", uri.authority());
  EXPECT_EQ("John.Doe@example.com", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("John.Doe@example.com", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_4) {
  auto uri = URI::Parse(
    "news:comp.infosystems.www.servers.unix");
  EXPECT_EQ("news", uri.scheme());
  EXPECT_EQ("", uri.authority());
  EXPECT_EQ("comp.infosystems.www.servers.unix", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("comp.infosystems.www.servers.unix", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_5) {
  auto uri = URI::Parse(
    "tel:+1-816-555-1212");
  EXPECT_EQ("tel", uri.scheme());
  EXPECT_EQ("", uri.authority());
  EXPECT_EQ("+1-816-555-1212", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("+1-816-555-1212", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_6) {
  auto uri = URI::Parse(
    "telnet://192.0.2.16:80/");
  EXPECT_EQ("telnet", uri.scheme());
  EXPECT_EQ("192.0.2.16:80", uri.authority());
  EXPECT_EQ("/", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("/", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_7) {
  auto uri = URI::Parse(
    "urn:oasis:names:specification:docbook:dtd:xml:4.1.2");
  EXPECT_EQ("urn", uri.scheme());
  EXPECT_EQ("", uri.authority());
  EXPECT_EQ("oasis:names:specification:docbook:dtd:xml:4.1.2", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("oasis:names:specification:docbook:dtd:xml:4.1.2", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_Error1) {
  auto uri = URI::Parse(
    "telnet://192.  0.2.16:80/");
  EXPECT_FALSE(uri.valid());
}

TEST(URI, Parse_Error2) {
  auto uri = URI::Parse(
    "telnet://192.\n0.2.16:80/");
  EXPECT_FALSE(uri.valid());
}

TEST(URI, Parse_Error3) {
  auto uri = URI::Parse("noscheme");
  EXPECT_FALSE(uri.valid());
}

}  // namespace dali
