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
#include "dali/pipeline/data/tensor_layout.h"

namespace dali {

TEST(TensorLayout, Construction) {
  TensorLayout empty;
  EXPECT_EQ(empty.ndim(), 0);

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

TEST(TensorLayout, VideoLayout) {
  EXPECT_TRUE(VideoLayoutInfo::IsVideo("NFCHW"));
  EXPECT_FALSE(VideoLayoutInfo::IsStillImage("NFCHW"));
  EXPECT_TRUE(VideoLayoutInfo::IsChannelFirst("NFCHW"));
  EXPECT_FALSE(VideoLayoutInfo::IsChannelFirst("NFHWC"));
  EXPECT_EQ(VideoLayoutInfo::FrameDim("NFCHW"), 1);
  EXPECT_FALSE(VideoLayoutInfo::IsSequence("NDCHW"));
  EXPECT_TRUE(VideoLayoutInfo::IsStillImage("NDCHW"));
}

}  // namespace dali
