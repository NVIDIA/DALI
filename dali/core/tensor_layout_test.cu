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
#include "dali/core/tensor_layout.h"
#include "dali/test/device_test.h"

namespace dali {

__device__ DeviceString dev_to_string(const TensorLayout &tl) {
  return tl.c_str();
}

DEVICE_TEST(TensorLayout_Dev, Construction, 1, 1) {
  TensorLayout empty;
  DEV_EXPECT_EQ(empty.ndim(), 0);
  DEV_EXPECT_EQ(empty.c_str()[0], 0);

  TensorLayout from_literal = "NHWC";
  DEV_EXPECT_EQ(from_literal.ndim(), 4);
  for (int i = 0; i <= 4; i ++)
    DEV_EXPECT_EQ(from_literal.c_str()[i], "NHWC"[i]);

  const char *cstr = "CHW";
  TensorLayout from_cstr = cstr;
  DEV_EXPECT_EQ(from_cstr.ndim(), 3);
  for (int i = 0; i <= 3; i ++)
    DEV_EXPECT_EQ(from_cstr.c_str()[i], "CHW"[i]);
}

DEVICE_TEST(TensorLayout_Dev, MaxLength, 1, 1) {
  // copy to a local variable to prevent GTest from requiring
  // external linkage on TensorLayout::max_ndim
  constexpr int kMaxN = TensorLayout::max_ndim;
  char buf[kMaxN + 1];
  for (int i = 0; i < kMaxN; i++)
    buf[i] = 'a' + i;
  buf[kMaxN] = 0;

  TensorLayout tl = buf;
  DEV_EXPECT_EQ(tl.ndim(), kMaxN);

  // include NULL terminator
  for (int i = 0; i <= kMaxN; i++)
    DEV_EXPECT_EQ(tl.c_str()[i], buf[i]);
}

DEVICE_TEST(TensorLayout_Dev, Equality, 1, 1) {
  DEV_EXPECT_TRUE(TensorLayout("NHWC") == TensorLayout("NHWC"));
  DEV_EXPECT_TRUE(TensorLayout("HWC") != TensorLayout("CHW"));
  DEV_EXPECT_TRUE(TensorLayout("asdf") == "asdf");                // NOLINT
  DEV_EXPECT_TRUE("fadd" != TensorLayout("fads"));                // NOLINT

  DEV_EXPECT_FALSE(TensorLayout("NHWC") != TensorLayout("NHWC"));
  DEV_EXPECT_FALSE(TensorLayout("HWC") == TensorLayout("CHW"));
  DEV_EXPECT_FALSE(TensorLayout("asdf") != "asdf");               // NOLINT
  DEV_EXPECT_FALSE("fadd" == TensorLayout("fads"));               // NOLINT

  DEV_EXPECT_FALSE("" == TensorLayout("asdf"));
  DEV_EXPECT_FALSE("asdf" == TensorLayout(""));
  DEV_EXPECT_TRUE("" != TensorLayout("asdf"));
  DEV_EXPECT_TRUE("asdf" != TensorLayout(""));
  DEV_EXPECT_FALSE("" != TensorLayout(""));
  DEV_EXPECT_TRUE("" == TensorLayout(""));
  DEV_EXPECT_TRUE(TensorLayout("") == "");
}

DEVICE_TEST(TensorLayout_Dev, ImageLayout, 1, 1) {
  TensorLayout nhwc = "NHWC";
  TensorLayout chw = "CHW";
  TensorLayout nchw = "NCHW";
  TensorLayout ncdhw = "NCDHW";
  TensorLayout ndhwc = "NDHWC";
  TensorLayout layouts[] = { nhwc, chw, nchw, ncdhw, ndhwc };

  DEV_EXPECT_FALSE(ImageLayoutInfo::IsChannelFirst(nhwc));
  DEV_EXPECT_FALSE(ImageLayoutInfo::IsChannelFirst(ndhwc));
  DEV_EXPECT_TRUE(ImageLayoutInfo::IsChannelFirst(nchw));
  DEV_EXPECT_TRUE(ImageLayoutInfo::IsChannelFirst(ncdhw));
  DEV_EXPECT_TRUE(ImageLayoutInfo::IsChannelFirst(chw));

  DEV_EXPECT_TRUE(ImageLayoutInfo::IsChannelLast(nhwc));
  DEV_EXPECT_TRUE(ImageLayoutInfo::IsChannelLast(ndhwc));
  DEV_EXPECT_FALSE(ImageLayoutInfo::IsChannelLast(nchw));
  DEV_EXPECT_FALSE(ImageLayoutInfo::IsChannelLast(ncdhw));
  DEV_EXPECT_FALSE(ImageLayoutInfo::IsChannelLast(chw));

  DEV_EXPECT_TRUE(LayoutInfo::HasSampleDim(nhwc));
  DEV_EXPECT_FALSE(LayoutInfo::HasSampleDim(chw));
  DEV_EXPECT_TRUE(LayoutInfo::HasSampleDim(nchw));
  DEV_EXPECT_TRUE(LayoutInfo::HasSampleDim(ndhwc));
  DEV_EXPECT_TRUE(LayoutInfo::HasSampleDim(ncdhw));

  DEV_EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(nhwc), 2);
  DEV_EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(chw), 2);
  DEV_EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(nchw), 2);
  DEV_EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(ndhwc), 3);
  DEV_EXPECT_EQ(ImageLayoutInfo::NumSpatialDims(ncdhw), 3);

  for (const TensorLayout &tl : layouts) {
    DEV_EXPECT_TRUE(ImageLayoutInfo::IsImage(tl));
    DEV_EXPECT_EQ(ImageLayoutInfo::Is2D(tl), ImageLayoutInfo::NumSpatialDims(tl) == 2);
    DEV_EXPECT_EQ(ImageLayoutInfo::Is3D(tl), ImageLayoutInfo::NumSpatialDims(tl) == 3);
  }
  DEV_EXPECT_FALSE(ImageLayoutInfo::IsImage("NC"));
}

DEVICE_TEST(TensorLayout_Dev, Find, 1, 1) {
  DEV_EXPECT_EQ(TensorLayout("asdfgh").find('a'), 0);
  DEV_EXPECT_EQ(TensorLayout("asdfgh").find('s'), 1);
  DEV_EXPECT_EQ(TensorLayout("asdfgh").find('h'), 5);
  DEV_EXPECT_EQ(TensorLayout("asdfgh").find('S'), -1);
  DEV_EXPECT_EQ(TensorLayout("asdfgh").find('\0'), -1);
  DEV_EXPECT_EQ(TensorLayout().find('a'), -1);
  DEV_EXPECT_EQ(TensorLayout().find('\0'), -1);
}

DEVICE_TEST(TensorLayout_Dev, Skip, 1, 1) {
  DEV_EXPECT_EQ(TensorLayout("asdfgh").skip('a'), TensorLayout("sdfgh"));
  DEV_EXPECT_EQ(TensorLayout("asdfgh").skip('s'), TensorLayout("adfgh"));
  DEV_EXPECT_EQ(TensorLayout("asdfgh").skip('h'), TensorLayout("asdfg"));
  DEV_EXPECT_EQ(TensorLayout("asdfgh").skip('\0'), TensorLayout("asdfgh"));
  DEV_EXPECT_EQ(TensorLayout("HWC").skip('N'), TensorLayout("HWC"));
  DEV_EXPECT_EQ(TensorLayout().skip('a'), TensorLayout());
  DEV_EXPECT_EQ(TensorLayout("a").skip('a'), TensorLayout());
  DEV_EXPECT_EQ(TensorLayout("through").skip('h'), TensorLayout("trough"));
}


DEVICE_TEST(TensorLayout_Dev, Sub, 1, 1) {
  DEV_EXPECT_EQ(TensorLayout("NHWC").sub(1), TensorLayout("HWC"));
  DEV_EXPECT_EQ(TensorLayout("asdfgh").sub(2, 3), TensorLayout("dfg"));
  DEV_EXPECT_EQ(TensorLayout("12345").first(3), TensorLayout("123"));
  DEV_EXPECT_EQ(TensorLayout("12345").last(4), TensorLayout("2345"));
}

DEVICE_TEST(TensorLayout_Dev, Concat, 1, 1) {
  DEV_EXPECT_EQ(TensorLayout("NH") + TensorLayout("WC"), TensorLayout("NHWC"));
  DEV_EXPECT_EQ(TensorLayout("1234") + "56", TensorLayout("123456"));
  DEV_EXPECT_EQ("abc" + TensorLayout("def"), TensorLayout("abcdef"));
}

DEVICE_TEST(TensorLayout_Dev, SampleLayout, 1, 1) {
  DEV_EXPECT_EQ(TensorLayout("NHWC").sample_layout(), TensorLayout("HWC"));
  DEV_EXPECT_EQ(TensorLayout().sample_layout(), TensorLayout());
}

DEVICE_TEST(TensorLayout_Dev, VideoLayout, 1, 1) {
  DEV_EXPECT_TRUE(VideoLayoutInfo::IsVideo("NFCHW"));
  DEV_EXPECT_TRUE(VideoLayoutInfo::IsVideo("FCHW"));
  DEV_EXPECT_FALSE(VideoLayoutInfo::IsStillImage("NFCHW"));
  DEV_EXPECT_TRUE(VideoLayoutInfo::IsChannelFirst("NFCHW"));
  DEV_EXPECT_FALSE(VideoLayoutInfo::IsChannelFirst("NFHWC"));
  DEV_EXPECT_EQ(VideoLayoutInfo::FrameDimIndex("NFCHW"), 1);
  DEV_EXPECT_FALSE(VideoLayoutInfo::IsSequence("NDCHW"));
  DEV_EXPECT_TRUE(VideoLayoutInfo::IsSequence("FDCHW"));
  DEV_EXPECT_FALSE(VideoLayoutInfo::IsSequence("DFCHW"));
  DEV_EXPECT_TRUE(VideoLayoutInfo::HasSequence("DFCHW"));
  DEV_EXPECT_TRUE(VideoLayoutInfo::IsStillImage("NDCHW"));
}

}  // namespace dali
