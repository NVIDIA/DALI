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

#include "dali/operators/image/resize/resize_attr.h"
#include <gtest/gtest.h>

namespace dali {

TEST(ResizeAttr, ParseLayout) {
  auto test = [](const TensorLayout &layout, int expected_spatial_ndim, int expected_first_dim) {
    int spatial_ndim = -1, first_spatial_dim = -1;
    EXPECT_NO_THROW(ResizeAttr::ParseLayout(spatial_ndim, first_spatial_dim, layout));
    EXPECT_EQ(spatial_ndim, expected_spatial_ndim);
    EXPECT_EQ(first_spatial_dim, expected_first_dim);
  };
  test("HW", 2, 0);
  test("DHW", 3, 0);

  test("HWC", 2, 0);
  test("CHW", 2, 1);

  test("DHWC", 3, 0);
  test("CDHW", 3, 1);

  test("FHWC", 2, 1);
  test("FCHW", 2, 2);

  test("FDHWC", 3, 1);
  test("FCDHW", 3, 2);

  {
    int spatial_ndim = -1, first_spatial_dim = -1;
    EXPECT_THROW(ResizeAttr::ParseLayout(spatial_ndim, first_spatial_dim, "HCW"),
                 std::runtime_error);
    EXPECT_THROW(ResizeAttr::ParseLayout(spatial_ndim, first_spatial_dim, "FWCH"),
                 std::runtime_error);
  }
}

void CheckResizeParams(const ResizeParams &actual, const ResizeParams &ref) {
  auto D = ref.dst_size.size();
  assert(ref.src_lo.size() == D);
  assert(ref.src_hi.size() == D);
  ASSERT_EQ(actual.dst_size.size(), D);
  ASSERT_EQ(actual.src_lo.size(), D);
  ASSERT_EQ(actual.src_hi.size(), D);
  const float eps = 1.0f/256;
  for (decltype(D) d = 0; d < D; d++) {
    EXPECT_EQ(actual.dst_size[d], ref.dst_size[d]) << "dst_size differs at dimension " << d;
    EXPECT_NEAR(actual.src_lo[d], ref.src_lo[d], eps) << "src_lo differs at dimension " << d;
    EXPECT_NEAR(actual.src_hi[d], ref.src_hi[d], eps) << "src_hi differs at dimension " << d;
  }
}

TEST(ResizeAttr, ResizeSeparate) {
  ArgumentWorkspace ws;
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_x", 480.0f);
    TensorListShape<> shape = {{
      TensorShape<3>{ 768, 1024, 3 },
      TensorShape<3>{ 320, 240, 1 }
    }};
    spec.AddArg("batch_size", shape.num_samples());

    ResizeAttr attr(spec);
    attr.PrepareResizeParams(spec, ws, shape, "HWC");

    CheckResizeParams(attr.params_[0], { { 360, 480 }, { 0, 0 }, { 768, 1024 } });
    CheckResizeParams(attr.params_[1], { { 640, 480 }, { 0, 0 }, { 320, 240 } });
  }
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_y", 480.0f);
    TensorListShape<> shape = {{
      TensorShape<3>{ 768, 1024, 3 },
      TensorShape<3>{ 320, 240, 1 }
    }};
    spec.AddArg("batch_size", shape.num_samples());

    ResizeAttr attr(spec);
    attr.PrepareResizeParams(spec, ws, shape, "HWC");

    CheckResizeParams(attr.params_[0], { { 480, 640 }, { 0, 0 }, { 768, 1024 } });
    CheckResizeParams(attr.params_[1], { { 480, 360 }, { 0, 0 }, { 320, 240 } });
  }
}

}  // namespace dali
