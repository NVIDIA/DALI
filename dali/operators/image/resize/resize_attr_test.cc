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

#include "dali/operators/image/resize/resize_attr.h"  // NOLINT
#include <gtest/gtest.h>
#include <memory>
#include "dali/test/tensor_test_utils.h"

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
}

TEST(ResizeAttr, ParseLayoutError) {
  int spatial_ndim = -1, first_spatial_dim = -1;
  EXPECT_THROW(ResizeAttr::ParseLayout(spatial_ndim, first_spatial_dim, "HCW"),
                std::runtime_error);
  EXPECT_THROW(ResizeAttr::ParseLayout(spatial_ndim, first_spatial_dim, "FWCH"),
                std::runtime_error);
}


inline void PrintTo(const ResizeParams &entry, std::ostream *os) {
  auto print = [os](auto &collection) {
    *os << "[";
    bool first = true;
    for (auto x : collection) {
      if (first)
        first = false;
      else
        *os << ", ";
      *os << x;
    }
    *os << "]";
  };

  *os << "dst = ";
  print(entry.dst_size);
  *os << "; lo = ";
  print(entry.src_lo);
  *os << "; hi = ";
  print(entry.src_hi);
}

inline bool CheckResizeParams(const ResizeParams &a, const ResizeParams &b) {
  if (a.dst_size != b.dst_size)
    return false;
  if (a.src_lo.size() != b.src_lo.size() || a.src_hi.size() != b.src_hi.size())
    return false;
  EqualEpsRel eq(1e-5, 1e-6);
  for (int d = 0, D = a.src_lo.size(); d < D; d++) {
    if (!eq(a.src_lo[d], b.src_lo[d]))
      return false;
  }
  for (int d = 0, D = a.src_hi.size(); d < D; d++) {
    if (!eq(a.src_hi[d], b.src_hi[d]))
      return false;
  }
  return true;
}

// use a macro instead of a function to get meaningful line numbers from GTest
#define CHECK_PARAMS(a, ...) \
  EXPECT_PRED2(CheckResizeParams, ResizeParams(a), ResizeParams(__VA_ARGS__))

TEST(ResizeAttr, ResizeSeparate) {
  ArgumentWorkspace ws;
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_x", 480.0f);
    TensorListShape<> shape = {{
      TensorShape<3>{ 768, 1024, 3 },
      TensorShape<3>{ 320, 240, 1 }
    }};
    spec.AddArg("max_batch_size", shape.num_samples());

    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "HWC");

    CHECK_PARAMS(attr.params_[0], { { 360, 480 }, { 0, 0 }, { 768, 1024 } });
    CHECK_PARAMS(attr.params_[1], { { 640, 480 }, { 0, 0 }, { 320, 240 } });
  }
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_y", 480.0f);
    TensorListShape<> shape = {{
      TensorShape<3>{ 768, 1024, 3 },
      TensorShape<3>{ 320, 240, 1 }
    }};
    spec.AddArg("max_batch_size", shape.num_samples());

    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "HWC");

    CHECK_PARAMS(attr.params_[0], { { 480, 640 }, { 0, 0 }, { 768, 1024 } });
    CHECK_PARAMS(attr.params_[1], { { 480, 360 }, { 0, 0 }, { 320, 240 } });
  }
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_z", 480.0f);
    TensorListShape<> shape = {{
      TensorShape<4>{ 3, 512, 768, 1024 },
      TensorShape<4>{ 1, 400, 320, 240 }
    }};
    spec.AddArg("max_batch_size", shape.num_samples());

    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "CDHW");

    CHECK_PARAMS(attr.params_[0], { { 480, 720, 960 }, { 0, 0, 0 }, { 512, 768, 1024 } });
    CHECK_PARAMS(attr.params_[1], { { 480, 384, 288 }, { 0, 0, 0 }, { 400, 320, 240 } });
  }
}

TEST(ResizeAttr, Resize3DSeparateArgs) {
  ArgumentWorkspace ws;
  auto zcoord = std::make_shared<TensorList<CPUBackend>>();
  zcoord->set_pinned(false);
  zcoord->Resize(TensorListShape<0>(2));
  *zcoord->mutable_tensor<float>(0) = 140;
  *zcoord->mutable_tensor<float>(1) = 150;
  ws.AddArgumentInput("resize_z", zcoord);

  TensorListShape<> shape = {{
    TensorShape<3>{ 123, 234, 345 },
    TensorShape<3>{ 321, 432, 543 }
  }};
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_x", 120.0f);
    spec.AddArg("resize_y", 130.0f);
    spec.AddArg("max_batch_size", shape.num_samples());
    spec.AddArgumentInput("resize_z", "resize_z");

    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "DHW");

    CHECK_PARAMS(attr.params_[0], { { 140, 130, 120 }, { 0, 0, 0 }, { 123, 234, 345 } });
    CHECK_PARAMS(attr.params_[1], { { 150, 130, 120 }, { 0, 0, 0 }, { 321, 432, 543 } });
  }
}

TEST(ResizeAttr, Resize3DSubpixelScale) {
  ArgumentWorkspace ws;
  auto zcoord = std::make_shared<TensorList<CPUBackend>>();
  zcoord->set_pinned(false);
  zcoord->Resize(TensorListShape<0>(2));
  *zcoord->mutable_tensor<float>(0) = 140;
  *zcoord->mutable_tensor<float>(1) = 150;
  ws.AddArgumentInput("resize_z", zcoord);

  TensorListShape<> shape = {{
    TensorShape<3>{ 123, 234, 345 },
    TensorShape<3>{ 321, 432, 543 }
  }};

  // Configure ResizeAttr with missing Y coordinate
  OpSpec spec("Resize");
  spec.AddArg("resize_x", 120.0f);
  spec.AddArg("max_batch_size", shape.num_samples());
  spec.AddArgumentInput("resize_z", "resize_z");

  ResizeAttr attr;
  attr.PrepareResizeParams(spec, ws, shape, "DHW");
  EXPECT_TRUE(attr.subpixel_scale_);

  // Y coordinate is missing and mode is "default" - use geometric mean of the xz scales for the
  // missing coordinate.
  double h0 = 234;
  double h1 = 432;
  // Calculate the ideal, subpixel, outptut shape.
  double subpixel_y0 = h0 * sqrt(140.0/123 * 120 / 345);
  double subpixel_y1 = h1 * sqrt(150.0/321 * 120 / 543);
  // Real shape needs to be integral - let's round it.
  int y0 = std::roundl(subpixel_y0);
  int y1 = std::roundl(subpixel_y1);

  // Now let's adjust the input ROI. It will be scaled proprtionately to the ratio
  // of actual (rounded) size to the ideal size.
  double center0 = h0 * 0.5;
  double center1 = h1 * 0.5;
  float lo0 = center0 - center0 * y0 / subpixel_y0;
  float hi0 = center0 + center0 * y0 / subpixel_y0;
  float lo1 = center1 - center1 * y1 / subpixel_y1;
  float hi1 = center1 + center1 * y1 / subpixel_y1;

  CHECK_PARAMS(attr.params_[0], { { 140, y0, 120 }, { 0, lo0, 0 }, { 123, hi0, 345 } });
  CHECK_PARAMS(attr.params_[1], { { 150, y1, 120 }, { 0, lo1, 0 }, { 321, hi1, 543 } });

  // Let's disable subpixel scaling
  spec.AddArg("subpixel_scale", false);
  attr.PrepareResizeParams(spec, ws, shape, "DHW");
  EXPECT_FALSE(attr.subpixel_scale_);

  // Now the input ROI should not be affected by the ideal vs real size difference
  CHECK_PARAMS(attr.params_[0], { { 140, y0, 120 }, { 0, 0, 0 }, { 123, 234, 345 } });
  CHECK_PARAMS(attr.params_[1], { { 150, y1, 120 }, { 0, 0, 0 }, { 321, 432, 543 } });
}

TEST(ResizeAttr, Resize3DStretch) {
  ArgumentWorkspace ws;
  auto zcoord = std::make_shared<TensorList<CPUBackend>>();
  zcoord->set_pinned(false);
  zcoord->Resize(TensorListShape<0>(2));
  *zcoord->mutable_tensor<float>(0) = 140;
  *zcoord->mutable_tensor<float>(1) = 150;
  ws.AddArgumentInput("resize_z", zcoord);

  TensorListShape<> shape = {{
    TensorShape<3>{ 123, 234, 345 },
    TensorShape<3>{ 321, 432, 543 }
  }};

  // Configure ResizeAttr with missing Y coordinate;
  // the mode is now "stretch", so the missing dimension (H) should be left unscaled.
  OpSpec spec("Resize");
  spec.AddArg("resize_x", 120.0f);
  spec.AddArg("max_batch_size", shape.num_samples());
  spec.AddArg("mode", "stretch");
  spec.AddArgumentInput("resize_z", "resize_z");

  ResizeAttr attr;
  attr.PrepareResizeParams(spec, ws, shape, "DHW");

  CHECK_PARAMS(attr.params_[0], { { 140, 234, 120 }, { 0, 0, 0 }, { 123, 234, 345 } });
  CHECK_PARAMS(attr.params_[1], { { 150, 432, 120 }, { 0, 0, 0 }, { 321, 432, 543 } });
}

TEST(ResizeAttr, Resize3DSizeArg) {
  ArgumentWorkspace ws;
  auto size = std::make_shared<TensorList<CPUBackend>>();
  auto tls = uniform_list_shape<1>(2, TensorShape<1>{3});
  size->set_pinned(false);
  size->Resize(tls);
  size->mutable_tensor<float>(0)[0] = 140;
  size->mutable_tensor<float>(0)[1] = 130;
  size->mutable_tensor<float>(0)[2] = 120;

  size->mutable_tensor<float>(1)[0] = 150;
  size->mutable_tensor<float>(1)[1] = 130;
  size->mutable_tensor<float>(1)[2] = 120;
  ws.AddArgumentInput("size", size);

  TensorListShape<> shape = {{
    TensorShape<3>{ 123, 234, 345 },
    TensorShape<3>{ 321, 432, 543 }
  }};
  OpSpec spec("Resize");
  spec.AddArgumentInput("size", "size");
  spec.AddArg("max_batch_size", shape.num_samples());
  spec.AddArg("subpixel_scale", false);

  {
    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "DHW");

    CHECK_PARAMS(attr.params_[0], { { 140, 130, 120 }, { 0, 0, 0 }, { 123, 234, 345 } });
    CHECK_PARAMS(attr.params_[1], { { 150, 130, 120 }, { 0, 0, 0 }, { 321, 432, 543 } });
  }

  size->mutable_tensor<float>(0)[1] = 0;  // 0 is undefined - as if the dim was absent
  size->mutable_tensor<float>(1)[1] = 0;

  {
    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "DHW");

    // one coordinate is missing and mode is default - use geometric mean of the scales for the
    // missing coordinate (y, in this case)
    int y0 = std::roundl(234 * sqrt(140.0/123 * 120 / 345));
    int y1 = std::roundl(432 * sqrt(150.0/321 * 120 / 543));

    CHECK_PARAMS(attr.params_[0], { { 140, y0, 120 }, { 0, 0, 0 }, { 123, 234, 345 } });
    CHECK_PARAMS(attr.params_[1], { { 150, y1, 120 }, { 0, 0, 0 }, { 321, 432, 543 } });
  }


  size->mutable_tensor<float>(0)[0] = -140;  // flip the volume depthwise
  size->mutable_tensor<float>(1)[2] = -120;  // flip horizontally
  size->mutable_tensor<float>(1)[0] =  150;  // don't flip

  {
    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "DHW");

    // one coordinate is missing and mode is default - use geometric mean of the scales for the
    // missing coordinate (y, in this case)
    int y0 = std::roundl(234 * sqrt(140.0/123 * 120 / 345));
    int y1 = std::roundl(432 * sqrt(150.0/321 * 120 / 543));

    CHECK_PARAMS(attr.params_[0], { { 140, y0, 120 }, { 123, 0, 0 }, { 0, 234, 345 } });
    CHECK_PARAMS(attr.params_[1], { { 150, y1, 120 }, { 0, 0, 543 }, { 321, 432, 0 } });

    spec.AddArg("mode", "stretch");
    // the mode is now "stretch", so the missing dimension (H) should be left unscaled
    attr.PrepareResizeParams(spec, ws, shape, "DHW");

    CHECK_PARAMS(attr.params_[0], { { 140, 234, 120 }, { 123, 0, 0 }, { 0, 234, 345 } });
    CHECK_PARAMS(attr.params_[1], { { 150, 432, 120 }, { 0, 0, 543 }, { 321, 432, 0 } });
  }
}


TEST(ResizeAttr, Resize3DNotLarger) {
  ArgumentWorkspace ws;
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_x", 500.0f);
    spec.AddArg("resize_y", 400.0f);
    spec.AddArg("mode", "not_larger");
    TensorListShape<> shape = {{
      TensorShape<3>{ 1536, 768, 1024 },
      TensorShape<3>{ 32, 320, 240 }
    }};
    spec.AddArg("max_batch_size", shape.num_samples());

    {
      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "DHW");

      CHECK_PARAMS(attr.params_[0], { { 750, 375, 500 }, { 0, 0, 0 }, { 1536, 768, 1024 } });
      CHECK_PARAMS(attr.params_[1], { { 40, 400, 300 }, { 0, 0, 0 }, { 32, 320, 240 } });
    }

    {
      spec.AddArg("resize_z", 600.0f);

      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "DHW");

      CHECK_PARAMS(attr.params_[0], { { 600, 300, 400 }, { 0, 0, 0 }, { 1536, 768, 1024 } });
      CHECK_PARAMS(attr.params_[1], { { 40, 400, 300 }, { 0, 0, 0 }, { 32, 320, 240 } });
    }
  }
}

TEST(ResizeAttr, Resize3DNotSmaller) {
  ArgumentWorkspace ws;
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_x", 600.0f);
    spec.AddArg("resize_y", 480.0f);
    spec.AddArg("mode", "not_smaller");
    TensorListShape<> shape = {{
      TensorShape<3>{ 1536, 768, 1024 },
      TensorShape<3>{ 32, 320, 240 }
    }};
    spec.AddArg("max_batch_size", shape.num_samples());

    {
      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "DHW");

      CHECK_PARAMS(attr.params_[0], { { 960, 480, 640 }, { 0, 0, 0 }, { 1536, 768, 1024 } });
      CHECK_PARAMS(attr.params_[1], { { 80, 800, 600 }, { 0, 0, 0 }, { 32, 320, 240 } });
    }

    {
      spec.AddArg("resize_z", 160.0f);

      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "DHW");

      CHECK_PARAMS(attr.params_[0], { { 960, 480, 640 }, { 0, 0, 0 }, { 1536, 768, 1024 } });
      CHECK_PARAMS(attr.params_[1], { { 160, 1600, 1200 }, { 0, 0, 0 }, { 32, 320, 240 } });
    }


    {
      spec.SetArg("max_size", vector<float>{720.0f});

      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "DHW");

      CHECK_PARAMS(attr.params_[0], { { 720, 360, 480 }, { 0, 0, 0 }, { 1536, 768, 1024 } });
      CHECK_PARAMS(attr.params_[1], { { 72, 720, 540 }, { 0, 0, 0 }, { 32, 320, 240 } });
    }

    {
      spec.SetArg("max_size", vector<float>{720.0f, 384.0f, 400.0f});

      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "DHW");

      CHECK_PARAMS(attr.params_[0], { { 600, 300, 400 }, { 0, 0, 0 }, { 1536, 768, 1024 } });
      CHECK_PARAMS(attr.params_[1], { { 38, 384, 288 },
                                      { 0.166667f, 0, 0 },  // subpixel scale
                                      { 31.83333f, 320, 240 } });
    }
  }
}


TEST(ResizeAttr, ResizeShorter) {
  ArgumentWorkspace ws;
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_shorter", 600.0f);
    TensorListShape<> shape = {{
      TensorShape<4>{ 20, 400, 800, 3 },
      TensorShape<4>{ 30, 500, 250, 3 }
    }};
    spec.AddArg("max_batch_size", shape.num_samples());

    {
      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "FHWC");

      CHECK_PARAMS(attr.params_[0], { { 600, 1200 }, { 0, 0 }, { 400, 800 } });
      CHECK_PARAMS(attr.params_[1], { { 1200, 600 }, { 0, 0 }, { 500, 250 } });
    }

    {
      spec.SetArg("max_size", vector<float>{800.0f, 1000.0f});

      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "FHWC");

      CHECK_PARAMS(attr.params_[0], { { 500, 1000 }, { 0, 0 }, { 400, 800 } });
      CHECK_PARAMS(attr.params_[1], { { 800, 400 }, { 0, 0 }, { 500, 250 } });
    }
  }
}

TEST(ResizeAttr, ResizeLonger) {
  ArgumentWorkspace ws;
  {
    OpSpec spec("Resize");
    spec.AddArg("resize_longer", 600.0f);
    TensorListShape<> shape = {{
      TensorShape<4>{ 20, 3, 400, 800 },
      TensorShape<4>{ 30, 3, 500, 250 }
    }};
    spec.AddArg("max_batch_size", shape.num_samples());

    {
      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "FCHW");

      CHECK_PARAMS(attr.params_[0], { { 300, 600 }, { 0, 0 }, { 400, 800 } });
      CHECK_PARAMS(attr.params_[1], { { 600, 300 }, { 0, 0 }, { 500, 250 } });
    }
  }
}

TEST(ResizeAttr, RoI) {
  ArgumentWorkspace ws;

  auto roi_input = std::make_shared<TensorList<CPUBackend>>();
  auto tls = uniform_list_shape<1>(2, TensorShape<1>{2});
  roi_input->set_pinned(false);
  roi_input->Resize(tls);
  roi_input->mutable_tensor<float>(0)[0] = 330;
  roi_input->mutable_tensor<float>(0)[1] = 40;
  roi_input->mutable_tensor<float>(1)[0] = 80;
  roi_input->mutable_tensor<float>(1)[1] = 230;
  ws.AddArgumentInput("roi_end", roi_input);

  {
    OpSpec spec("Resize");
    spec.AddArg("resize_x", 200.0f);
    spec.AddArg("resize_y", -100.0f);
    spec.AddArg("roi_start", vector<float>({7, 200}));
    spec.AddArgumentInput("roi_end", "roi_end");
    TensorListShape<> shape = {{
      TensorShape<3>{ 3, 400, 800 },
      TensorShape<3>{ 3, 100, 250 }
    }};
    spec.AddArg("max_batch_size", shape.num_samples());

    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "CHW");
    EXPECT_FALSE(attr.roi_relative_);

    // Input ROI is:
    // 0:    7, 200  -- 330,  40
    // 1:    7, 200  --  80, 230
    // Y axis is flipped, so we have:
    // 0:   330, 200 --   7,  40
    // 1:   80,  200 --   7, 230
    CHECK_PARAMS(attr.params_[0], { { 100, 200 }, { 330, 200 }, { 7, 40 } });
    CHECK_PARAMS(attr.params_[1], { { 100, 200 }, { 80, 200 }, { 7, 230 } });
  }

  ws.Clear();

  roi_input->mutable_tensor<float>(0)[0] = 0.1f;
  roi_input->mutable_tensor<float>(0)[1] = 0.9f;
  roi_input->mutable_tensor<float>(1)[0] = 0.8f;
  roi_input->mutable_tensor<float>(1)[1] = 0.9f;
  ws.AddArgumentInput("roi_start", roi_input);

  {
    OpSpec spec("Resize");
    spec.AddArg("resize_x", -200.0f);
    spec.AddArg("resize_y", 100.0f);
    spec.AddArg("roi_end", vector<float>({0.5, 0.5}));
    spec.AddArgumentInput("roi_start", "roi_start");
    spec.AddArg("roi_relative", true);
    TensorListShape<> shape = {{
      TensorShape<3>{ 3, 400, 800 },
      TensorShape<3>{ 3, 100, 250 }
    }};
    spec.AddArg("max_batch_size", shape.num_samples());

    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "CHW");
    EXPECT_TRUE(attr.roi_relative_);

    // Input ROI is:
    // 0:   40, 720 -- 200, 400
    // 1:   80, 225 --  50, 125
    // X axis is flipped, so we have:
    // 0:   40, 400 -- 200, 720
    // 1:   80, 125 --  50, 225
    CHECK_PARAMS(attr.params_[0], { { 100, 200 }, { 40, 400 }, { 200, 720 } });
    CHECK_PARAMS(attr.params_[1], { { 100, 200 }, { 80, 125 }, { 50, 225 } });
  }
}

}  // namespace dali
