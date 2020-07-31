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

inline bool operator==(const ResizeParams &a, const ResizeParams &b) {
  return a.dst_size == b.dst_size && a.src_lo == b.src_lo && a.src_hi == b.src_hi;
}

// use a macro instead of a function to get meaningful line numbers from GTest
#define CHECK_PARAMS(a, ...) EXPECT_EQ(ResizeParams(a), ResizeParams(__VA_ARGS__))

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
    spec.AddArg("batch_size", shape.num_samples());

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
    spec.AddArg("batch_size", shape.num_samples());

    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "CDHW");

    CHECK_PARAMS(attr.params_[0], { { 480, 720, 960 }, { 0, 0, 0 }, { 512, 768, 1024 } });
    CHECK_PARAMS(attr.params_[1], { { 480, 384, 288 }, { 0, 0, 0 }, { 400, 320, 240 } });
  }
}

TEST(ResizeAttr, Resize3DStretchSeparateArgs) {
  ArgumentWorkspace ws;
  auto zcoord = std::make_shared<TensorList<CPUBackend>>();
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
    spec.AddArg("batch_size", shape.num_samples());
    spec.AddArgumentInput("resize_z", "resize_z");

    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "DHW");

    CHECK_PARAMS(attr.params_[0], { { 140, 130, 120 }, { 0, 0, 0 }, { 123, 234, 345 } });
    CHECK_PARAMS(attr.params_[1], { { 150, 130, 120 }, { 0, 0, 0 }, { 321, 432, 543 } });
  }

  {
    OpSpec spec("Resize");
    spec.AddArg("resize_x", 120.0f);
    spec.AddArg("batch_size", shape.num_samples());
    spec.AddArgumentInput("resize_z", "resize_z");

    ResizeAttr attr;
    attr.PrepareResizeParams(spec, ws, shape, "DHW");

    // one coordinate is missing and mode is stretch - use geometric mean of the scales for the
    // missing coordinate (y, in this case)
    int y0 = std::roundl(234 * sqrt(140.0/123 * 120 / 345));
    int y1 = std::roundl(432 * sqrt(150.0/321 * 120 / 543));

    CHECK_PARAMS(attr.params_[0], { { 140, y0, 120 }, { 0, 0, 0 }, { 123, 234, 345 } });
    CHECK_PARAMS(attr.params_[1], { { 150, y1, 120 }, { 0, 0, 0 }, { 321, 432, 543 } });
  }
}

TEST(ResizeAttr, Resize3DStretchSizeArg) {
  ArgumentWorkspace ws;
  auto size = std::make_shared<TensorList<CPUBackend>>();
  auto tls = uniform_list_shape<1>(2, TensorShape<1>{3});
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
  spec.AddArg("batch_size", shape.num_samples());

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

    // one coordinate is missing and mode is stretch - use geometric mean of the scales for the
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

    // one coordinate is missing and mode is stretch - use geometric mean of the scales for the
    // missing coordinate (y, in this case)
    int y0 = std::roundl(234 * sqrt(140.0/123 * 120 / 345));
    int y1 = std::roundl(432 * sqrt(150.0/321 * 120 / 543));

    CHECK_PARAMS(attr.params_[0], { { 140, y0, 120 }, { 123, 0, 0 }, { 0, 234, 345 } });
    CHECK_PARAMS(attr.params_[1], { { 150, y1, 120 }, { 0, 0, 543 }, { 321, 432, 0 } });
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
    spec.AddArg("batch_size", shape.num_samples());

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
    spec.AddArg("batch_size", shape.num_samples());

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
      CHECK_PARAMS(attr.params_[1], { { 38, 384, 288 }, { 0, 0, 0 }, { 32, 320, 240 } });
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
    spec.AddArg("batch_size", shape.num_samples());

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
    spec.AddArg("batch_size", shape.num_samples());

    {
      ResizeAttr attr;
      attr.PrepareResizeParams(spec, ws, shape, "FCHW");

      CHECK_PARAMS(attr.params_[0], { { 300, 600 }, { 0, 0 }, { 400, 800 } });
      CHECK_PARAMS(attr.params_[1], { { 600, 300 }, { 0, 0 }, { 500, 250 } });
    }
  }
}

}  // namespace dali
