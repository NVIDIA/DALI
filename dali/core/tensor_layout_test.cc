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
#include <type_traits>
#include "dali/core/tensor_layout.h"

namespace dali {

static_assert(
  std::is_standard_layout<TensorLayout>::value &&
  std::is_trivially_copy_constructible<TensorLayout>::value &&
  std::is_trivially_copy_assignable<TensorLayout>::value &&
  std::is_trivially_destructible<TensorLayout>::value,
  "TensorLayout must be a POD except for non-trivial construction");

TEST(TensorLayout, Construction) {
  TensorLayout empty;
  EXPECT_EQ(empty.ndim(), 0);
  EXPECT_EQ(empty.str(), string());
  EXPECT_STREQ(empty.c_str(), "");

  TensorLayout from_literal = "NHWC";
  EXPECT_EQ(from_literal.ndim(), 4);
  EXPECT_EQ(from_literal.str(), "NHWC");
  EXPECT_STREQ(from_literal.c_str(), "NHWC");

  const char *cstr = "CHW";
  TensorLayout from_cstr = cstr;
  EXPECT_EQ(from_cstr.ndim(), 3);
  EXPECT_EQ(from_cstr.str(), "CHW");
  EXPECT_STREQ(from_cstr.c_str(), "CHW");

  TensorLayout from_str = std::string("asdfg");
  EXPECT_EQ(from_str.ndim(), 5);
  EXPECT_EQ(from_str.str(), "asdfg");
  EXPECT_STREQ(from_str.c_str(), "asdfg");
}

TEST(TensorLayout, MaxLength) {
  // copy to a local variable to prevent GTest from requiring
  // external linkage on TensorLayout::max_ndim
  constexpr int kMaxN = TensorLayout::max_ndim;
  char buf[kMaxN + 1];
  for (int i = 0; i < kMaxN; i++)
    buf[i] = 'a' + i;
  buf[kMaxN] = 0;

  TensorLayout tl = buf;
  EXPECT_EQ(tl.ndim(), kMaxN);
  EXPECT_STREQ(tl.c_str(), buf);
}

TEST(TensorLayout, Equality) {
  EXPECT_TRUE(TensorLayout("NHWC") == TensorLayout("NHWC"));
  EXPECT_TRUE(TensorLayout("HWC") != TensorLayout("CHW"));
  EXPECT_TRUE(TensorLayout("asdf") == "asdf");                // NOLINT
  EXPECT_TRUE(TensorLayout("asdf") != std::string("asdfg"));
  EXPECT_TRUE(std::string("dsaf") == TensorLayout("dsaf"));
  EXPECT_TRUE("fadd" != TensorLayout("fads"));                // NOLINT

  EXPECT_FALSE(TensorLayout("NHWC") != TensorLayout("NHWC"));
  EXPECT_FALSE(TensorLayout("HWC") == TensorLayout("CHW"));
  EXPECT_FALSE(TensorLayout("asdf") != "asdf");               // NOLINT
  EXPECT_FALSE(TensorLayout("asdf") == std::string("asdfg"));
  EXPECT_FALSE(std::string("dsaf") != TensorLayout("dsaf"));
  EXPECT_FALSE("fadd" == TensorLayout("fads"));               // NOLINT
}


void TestTLCompare(const std::string &s1, const std::string &s2) {
  static const char err_msg[] =
    "Tensor layout comparison should yield same result as string comparison";

  EXPECT_EQ((TensorLayout(s1) < TensorLayout(s2)), (s1 < s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) < s2), (s1 < s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) < s2.c_str()), (s1 < s2)) << err_msg;
  EXPECT_EQ((s1 < TensorLayout(s2)), (s1 < s2)) << err_msg;
  EXPECT_EQ((s1.c_str() < TensorLayout(s2)), (s1 < s2)) << err_msg;

  EXPECT_EQ((TensorLayout(s1) > TensorLayout(s2)), (s1 > s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) > s2), (s1 > s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) > s2.c_str()), (s1 > s2)) << err_msg;
  EXPECT_EQ((s1 > TensorLayout(s2)), (s1 > s2)) << err_msg;
  EXPECT_EQ((s1.c_str() > TensorLayout(s2)), (s1 > s2)) << err_msg;

  EXPECT_EQ((TensorLayout(s1) <= TensorLayout(s2)), (s1 <= s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) <= s2), (s1 <= s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) <= s2.c_str()), (s1 <= s2)) << err_msg;
  EXPECT_EQ((s1 <= TensorLayout(s2)), (s1 <= s2)) << err_msg;
  EXPECT_EQ((s1.c_str() <= TensorLayout(s2)), (s1 <= s2)) << err_msg;

  EXPECT_EQ((TensorLayout(s1) >= TensorLayout(s2)), (s1 >= s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) >= s2), (s1 >= s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) >= s2.c_str()), (s1 >= s2)) << err_msg;
  EXPECT_EQ((s1 >= TensorLayout(s2)), (s1 >= s2)) << err_msg;
  EXPECT_EQ((s1.c_str() >= TensorLayout(s2)), (s1 >= s2)) << err_msg;

  EXPECT_EQ((TensorLayout(s1) == TensorLayout(s2)), (s1 == s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) == s2), (s1 == s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) == s2.c_str()), (s1 == s2)) << err_msg;
  EXPECT_EQ((s1 == TensorLayout(s2)), (s1 == s2)) << err_msg;
  EXPECT_EQ((s1.c_str() == TensorLayout(s2)), (s1 == s2)) << err_msg;

  EXPECT_EQ((TensorLayout(s1) != TensorLayout(s2)), (s1 != s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) != s2), (s1 != s2)) << err_msg;
  EXPECT_EQ((TensorLayout(s1) != s2.c_str()), (s1 != s2)) << err_msg;
  EXPECT_EQ((s1 != TensorLayout(s2)), (s1 != s2)) << err_msg;
  EXPECT_EQ((s1.c_str() != TensorLayout(s2)), (s1 != s2)) << err_msg;
}

TEST(TensorLayout, Comparison) {
  TestTLCompare("str1", "str2");
  TestTLCompare("ASDF", "ADSF");
  TestTLCompare("111", "1111");
  TestTLCompare("4321", "432");
  TestTLCompare("bang", "bang");
  TestTLCompare("str1", "");
  TestTLCompare("", "str2");
  TestTLCompare("", "");
}

TEST(TensorLayout, Find) {
  EXPECT_EQ(TensorLayout("asdfgh").find('a'), 0);
  EXPECT_EQ(TensorLayout("asdfgh").find('s'), 1);
  EXPECT_EQ(TensorLayout("asdfgh").find('h'), 5);
  EXPECT_EQ(TensorLayout("asdfgh").find('S'), -1);
  EXPECT_EQ(TensorLayout("asdfgh").find('\0'), -1);
  EXPECT_EQ(TensorLayout().find('a'), -1);
  EXPECT_EQ(TensorLayout().find('\0'), -1);
}

TEST(TensorLayout, Skip) {
  EXPECT_EQ(TensorLayout("asdfgh").skip('a'), TensorLayout("sdfgh"));
  EXPECT_EQ(TensorLayout("asdfgh").skip('s'), TensorLayout("adfgh"));
  EXPECT_EQ(TensorLayout("asdfgh").skip('h'), TensorLayout("asdfg"));
  EXPECT_EQ(TensorLayout("asdfgh").skip('\0'), TensorLayout("asdfgh"));
  EXPECT_EQ(TensorLayout("HWC").skip('N'), TensorLayout("HWC"));
  EXPECT_EQ(TensorLayout().skip('a'), TensorLayout());
  EXPECT_EQ(TensorLayout("a").skip('a'), TensorLayout());
  EXPECT_EQ(TensorLayout("through").skip('h'), TensorLayout("trough"));
}

TEST(TensorLayout, ImageLayout) {
  TensorLayout nhwc = "NHWC";
  TensorLayout chw = "CHW";
  TensorLayout nchw = "NCHW";
  TensorLayout ncdhw = "NCDHW";
  TensorLayout ndhwc = "NDHWC";
  TensorLayout layouts[] = { nhwc, chw, nchw, ncdhw, ndhwc };

  EXPECT_FALSE(ImageLayoutInfo::IsChannelFirst(nhwc));
  EXPECT_FALSE(ImageLayoutInfo::IsChannelFirst(ndhwc));
  EXPECT_TRUE(ImageLayoutInfo::IsChannelFirst(nchw));
  EXPECT_TRUE(ImageLayoutInfo::IsChannelFirst(ncdhw));
  EXPECT_TRUE(ImageLayoutInfo::IsChannelFirst(chw));

  EXPECT_TRUE(ImageLayoutInfo::IsChannelLast(nhwc));
  EXPECT_TRUE(ImageLayoutInfo::IsChannelLast(ndhwc));
  EXPECT_FALSE(ImageLayoutInfo::IsChannelLast(nchw));
  EXPECT_FALSE(ImageLayoutInfo::IsChannelLast(ncdhw));
  EXPECT_FALSE(ImageLayoutInfo::IsChannelLast(chw));

  EXPECT_TRUE(LayoutInfo::HasSampleDim(nhwc));
  EXPECT_FALSE(LayoutInfo::HasSampleDim(chw));
  EXPECT_TRUE(LayoutInfo::HasSampleDim(nchw));
  EXPECT_TRUE(LayoutInfo::HasSampleDim(ndhwc));
  EXPECT_TRUE(LayoutInfo::HasSampleDim(ncdhw));

  EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(nhwc), 2);
  EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(chw), 2);
  EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(nchw), 2);
  EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(ndhwc), 3);
  EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(ncdhw), 3);

  for (const TensorLayout &tl : layouts) {
    EXPECT_TRUE(ImageLayoutInfo::IsImage(tl));
    EXPECT_EQ(ImageLayoutInfo::Is2D(tl), ImageLayoutInfo::NumSpatialDims(tl) == 2);
    EXPECT_EQ(ImageLayoutInfo::Is3D(tl), ImageLayoutInfo::NumSpatialDims(tl) == 3);
  }
  EXPECT_FALSE(ImageLayoutInfo::IsImage("NC"));
}

TEST(TensorLayout, Sub) {
  EXPECT_EQ(TensorLayout("NHWC").sub(1), "HWC");
  EXPECT_EQ(TensorLayout("asdfgh").sub(2, 3), "dfg");
  EXPECT_EQ(TensorLayout("12345").first(3), "123");
  EXPECT_EQ(TensorLayout("12345").last(4), "2345");
}

TEST(TensorLayout, Concat) {
  EXPECT_EQ(TensorLayout("NH") + TensorLayout("WC"), TensorLayout("NHWC"));
  EXPECT_EQ(TensorLayout("1234") + "56", TensorLayout("123456"));
  EXPECT_EQ("abc" + TensorLayout("def"), TensorLayout("abcdef"));
}

TEST(TensorLayout, SampleLayout) {
  EXPECT_EQ(TensorLayout("NHWC").sample_layout(), TensorLayout("HWC"));
  EXPECT_EQ(TensorLayout().sample_layout(), TensorLayout());
  EXPECT_THROW(TensorLayout("HWC").sample_layout(), std::logic_error);
}

TEST(TensorLayout, VideoLayout) {
  EXPECT_TRUE(VideoLayoutInfo::IsVideo("NFCHW"));
  EXPECT_TRUE(VideoLayoutInfo::IsVideo("FCHW"));
  EXPECT_FALSE(VideoLayoutInfo::IsStillImage("NFCHW"));
  EXPECT_TRUE(VideoLayoutInfo::IsChannelFirst("NFCHW"));
  EXPECT_FALSE(VideoLayoutInfo::IsChannelFirst("NFHWC"));
  EXPECT_EQ(VideoLayoutInfo::FrameDimIndex("NFCHW"), 1);
  EXPECT_FALSE(VideoLayoutInfo::IsSequence("NDCHW"));
  EXPECT_TRUE(VideoLayoutInfo::IsSequence("FDCHW"));
  EXPECT_FALSE(VideoLayoutInfo::IsSequence("DFCHW"));
  EXPECT_TRUE(VideoLayoutInfo::HasSequence("DFCHW"));
  EXPECT_TRUE(VideoLayoutInfo::IsStillImage("NDCHW"));
  EXPECT_EQ(VideoLayoutInfo::GetFrameLayout("FCHW"), TensorLayout("CHW"));
  EXPECT_EQ(VideoLayoutInfo::GetFrameLayout("NFHWC"), TensorLayout("NHWC"));
}

TEST(TensorLayout, IsPermutationOf) {
  EXPECT_TRUE(TensorLayout("asdfg").is_permutation_of("asdfg"));
  EXPECT_TRUE(TensorLayout("asdfg").is_permutation_of("gfdsa"));
  EXPECT_TRUE(TensorLayout("asdfa").is_permutation_of("aasdf"));
  EXPECT_TRUE(TensorLayout("").is_permutation_of(""));
  EXPECT_TRUE(TensorLayout("11211").is_permutation_of("21111"));
  EXPECT_TRUE(TensorLayout("111122").is_permutation_of("211112"));
  EXPECT_TRUE(TensorLayout("453162").is_permutation_of("123456"));
  EXPECT_FALSE(TensorLayout("453162").is_permutation_of("12345"));
  EXPECT_FALSE(TensorLayout("53162").is_permutation_of("123456"));
  EXPECT_FALSE(TensorLayout("11122").is_permutation_of("1112"));
  EXPECT_FALSE(TensorLayout("11122").is_permutation_of("1122"));
  EXPECT_FALSE(TensorLayout("22111").is_permutation_of("1122"));
  EXPECT_FALSE(TensorLayout("asdff").is_permutation_of("aasdf"));
  EXPECT_FALSE(TensorLayout("asdff").is_permutation_of(""));
  EXPECT_FALSE(TensorLayout("").is_permutation_of("asdfdsa"));
}

TEST(TensorLayout, GetLayoutMapping) {
  {
    auto perm = GetLayoutMapping<4>("NHWC", "NCHW");
    std::array<int, 4> ref{{ 0, 3, 1, 2 }};
    EXPECT_EQ(perm, ref);

    perm = GetLayoutMapping<4>("NCHW", "NHWC");
    ref = {{ 0, 2, 3, 1 }};
    EXPECT_EQ(perm, ref);
  }
  {
    auto perm = GetLayoutMapping<5>("01234", "34201");
    std::array<int, 5> ref{{ 3, 4, 2, 0, 1 }};
    EXPECT_EQ(perm, ref);

    perm = GetLayoutMapping<5>("aabba", "baaba");
    ref = {{ 2, 0, 1, 3, 4 }};
    EXPECT_EQ(perm, ref);

    perm = GetLayoutMapping<5>("aaabb", "baaba");
    ref = {{ 3, 0, 1, 4, 2 }};
    EXPECT_EQ(perm, ref);
  }
  {
    EXPECT_THROW(GetLayoutMapping<1>("@", "#"), DALIException);
  }
}

TEST(TensorLayout, Resize) {
  TensorLayout layout;
  EXPECT_EQ(layout, "");

  layout.resize(3);
  EXPECT_EQ(layout, "???");

  layout = "HW";
  EXPECT_EQ(layout, "HW");

  layout.resize(3);
  EXPECT_EQ(layout, "HW?");

  layout.resize(4, '#');
  EXPECT_EQ(layout, "HW?#");
}

}  // namespace dali
