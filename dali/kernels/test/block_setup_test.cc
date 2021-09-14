// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/common/block_setup.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/int_literals.h"

namespace dali {
namespace kernels {

TEST(TensorShape, skip_dim) {
  TensorShape<5> in = { 1, 2, 3, 4, 5 };
  EXPECT_EQ(skip_dim<-1>(in), in);
  EXPECT_EQ(skip_dim<0>(in), TensorShape<4>(   2, 3, 4, 5));  // NOLINT
  EXPECT_EQ(skip_dim<1>(in), TensorShape<4>(1,    3, 4, 5));
  EXPECT_EQ(skip_dim<2>(in), TensorShape<4>(1, 2,    4, 5));
  EXPECT_EQ(skip_dim<3>(in), TensorShape<4>(1, 2, 3,    5));
  EXPECT_EQ(skip_dim<4>(in), TensorShape<4>(1, 2, 3, 4   ));  // NOLINT
}

TEST(BlockSetup, shape2vec) {
  TensorShape<1> in1 = { 123 };
  EXPECT_EQ(shape2vec(in1), ivec1(123));
  TensorShape<3> in3 = { 2, 3, 4 };
  EXPECT_EQ(shape2vec(in3), ivec3(4, 3, 2));
}

template <int ndim, int channel_dim>
void PerturbAndValidate(
    BlockSetup<ndim, channel_dim> &setup,
    TensorListShape<ndim + (channel_dim >= 0)> shape,
    const TensorListShape<ndim + (channel_dim >= 0)> in_shape,
    span<TensorShape<ndim>> sizes) {
  constexpr int tensor_ndim = BlockSetup<ndim, channel_dim>::tensor_ndim;
  EXPECT_NO_THROW(setup.ValidateOutputShape(shape, in_shape, sizes));
  for (int i = 0; i < shape.num_samples(); i++) {
    for (int d = 0; d < tensor_ndim; d++) {
      shape.tensor_shape_span(i)[d]++;
      EXPECT_THROW(setup.ValidateOutputShape(shape, in_shape, sizes), std::exception);
      shape.tensor_shape_span(i)[d]--;
      EXPECT_NO_THROW(setup.ValidateOutputShape(shape, in_shape, sizes));
    }
  }
}

TEST(BlockSetup, GetOutputShape_DHWC) {
  BlockSetup<3, 3> setup;  // DHWC
  TensorListShape<4> in_shape = {{
    { 100, 200, 300, 3 },
    { 12, 23, 34, 5 }
  }};
  std::vector<TensorShape<3>> out_sizes = {
    { 20, 30, 40 },
    { 10, 15, 25 }
  };
  TensorListShape<4> out_shape = setup.GetOutputShape(in_shape, make_span(out_sizes));
  TensorListShape<4> ref_shape = {{
    { 20, 30, 40, 3},
    { 10, 15, 25, 5}
  }};
  EXPECT_EQ(out_shape, ref_shape);
  PerturbAndValidate(setup, out_shape, in_shape, make_span(out_sizes));
}

TEST(BlockSetup, GetOutputShape_DHW) {
  BlockSetup<3, -1> setup;
  TensorListShape<3> in_shape = {{
    { 100, 200, 300 },
    { 12, 23, 34 }
  }};
  std::vector<TensorShape<3>> out_sizes = {
    { 20, 30, 40 },
    { 10, 15, 25 }
  };
  TensorListShape<3> out_shape = setup.GetOutputShape(in_shape, make_span(out_sizes));
  TensorListShape<3> ref_shape = {{
    { 20, 30, 40 },
    { 10, 15, 25 }
  }};
  EXPECT_EQ(out_shape, ref_shape);
  PerturbAndValidate(setup, out_shape, in_shape, make_span(out_sizes));
}

TEST(BlockSetup, GetOutputShape_CHW) {
  BlockSetup<2, 0> setup;
  TensorListShape<3> in_shape = {{
    { 5, 768, 1024 },
    { 3, 576, 720 },
    { 4, 224, 224 }
}};
  std::vector<TensorShape<2>> out_sizes = {
    { 300, 400 },
    { 240, 320 },
    { 256, 256 }
  };
  TensorListShape<3> out_shape = setup.GetOutputShape(in_shape, make_span(out_sizes));
  TensorListShape<3> ref_shape = {{
    { 5, 300, 400 },
    { 3, 240, 320 },
    { 4, 256, 256 }
  }};
  EXPECT_EQ(out_shape, ref_shape);
  PerturbAndValidate(setup, out_shape, in_shape, make_span(out_sizes));
}


namespace {

template <int dim>
struct BlockMap {
  int64_t end;
  std::map<int64_t, BlockMap<dim-1>> inner;
};

template <>
struct BlockMap<0> {
  int64_t end;
  // dummy - to avoid specialization
  const bool inner = false;
};

template <int dim>
bool operator==(const BlockMap<dim> &a, const BlockMap<dim> &b) {
  return a.end == b.end && a.inner == b.inner;
}

inline void ValidateBlockMap(const BlockMap<0> &map, const TensorShape<0> &shape) {}

/**
 * @brief Check that the output shape is covered with rectangular grid.
 *
 * The grid cells must be aligned between rows/slices, but don't have to be uniform
 * - typically the last cell will be smaller and that's expected.
 */
template <int dim>
void ValidateBlockMap(const BlockMap<dim> &map, const TensorShape<dim> &shape) {
  ASSERT_FALSE(map.inner.empty());
  int64_t i = 0;
  for (auto &p : map.inner) {
    ASSERT_EQ(p.first, i) << "Blocks don't cover the image";
    ASSERT_GT(p.second.end, i) << "Block end coordinate must be greater than start";
    i = p.second.end;
  }
  EXPECT_EQ(i, shape[0])
    << (i < shape[0] ? "Block does not cover whole image" : "Block exceeds image size");

  const BlockMap<dim-1> *first_slice = 0;
  for (auto &p : map.inner) {
    if (first_slice) {
      EXPECT_EQ(p.second.inner, first_slice->inner) << "Inner block layout must be uniform";
    } else {
      first_slice = &p.second;
      // Validate the first slice recursively - the remaining slices should be equal and
      // therefore don't require validation.
      ValidateBlockMap(p.second, shape.template last<dim-1>());
    }
  }
}

}  // namespace

TEST(BlockSetup, SetupBlocks_Variable_1D) {
  TensorListShape<1> TLS({
    { 1_i64 << 32 },
    { (1_i64 << 32) + 1023 },
    { 1024 },
    { 512 },
    { 733 },
  });

  BlockSetup<1, -1> setup(1 << 16);
  setup.SetDefaultBlockSize({1024});
  static_assert(setup.tensor_ndim == 1, "Incorrectly inferred tensor_ndim");
  setup.SetupBlocks(TLS);
  ASSERT_FALSE(setup.IsUniformSize());
  int prev = -1;
  BlockMap<1> map;
  for (auto &blk : setup.Blocks()) {
    if (blk.sample_idx != prev) {
      if (prev != -1) {
        ValidateBlockMap(map, TLS[prev]);
      }
      prev = blk.sample_idx;
      map = {};
    }
    map.inner[blk.start.x].end = blk.end.x;
  }
  if (prev != -1)
    ValidateBlockMap(map, TLS[prev]);

  EXPECT_EQ(setup.GridDimVec(), ivec3(setup.Blocks().size(), 1, 1));
}

TEST(BlockSetup, SetupBlocks_Variable_1D_Ch) {
  TensorListShape<2> TLS({
    { 1_i64 << 32, 3 },
    { (1_i64 << 32) + 1023, 5 },
    { 1024, 3 },
    { 512, 3 },
    { 733, 3 },
  });

  BlockSetup<1, 1> setup(1 << 16);
  setup.SetDefaultBlockSize({1024});
  static_assert(setup.tensor_ndim == 2, "Incorrectly inferred tensor_ndim");
  setup.SetupBlocks(TLS);
  ASSERT_FALSE(setup.IsUniformSize());
  int prev = -1;
  BlockMap<1> map;
  for (auto &blk : setup.Blocks()) {
    if (blk.sample_idx != prev) {
      if (prev != -1) {
        ValidateBlockMap(map, TLS[prev].first<1>());
      }
      prev = blk.sample_idx;
      map = {};
    }
    map.inner[blk.start.x].end = blk.end.x;
  }
  if (prev != -1)
    ValidateBlockMap(map, TLS[prev].first<1>());

  EXPECT_EQ(setup.GridDimVec(), ivec3(setup.Blocks().size(), 1, 1));
}



TEST(BlockSetup, SetupBlocks_Variable) {
  TensorListShape<3> TLS({
    { 480, 640, 3 },
    { 768, 1024, 3 },
    { 600, 800, 3 },
    { 720, 1280, 3 },
    { 480, 864, 3 },
    { 576, 720, 3 }
  });

  BlockSetup<2, 2> setup;
  static_assert(setup.tensor_ndim == 3, "Incorrectly inferred tensor_ndim");
  setup.SetupBlocks(TLS);
  ASSERT_FALSE(setup.IsUniformSize());
  int prev = -1;
  BlockMap<2> map;
  for (auto &blk : setup.Blocks()) {
    if (blk.sample_idx != prev) {
      if (prev != -1) {
        ValidateBlockMap(map, TLS[prev].first<2>());
      }
      prev = blk.sample_idx;
      map = {};
    }
    auto &b = map.inner[blk.start.y];
    b.end = blk.end.y;
    b.inner[blk.start.x].end = blk.end.x;
  }
  if (prev != -1)
    ValidateBlockMap(map, TLS[prev].first<2>());

  EXPECT_EQ(setup.GridDimVec(), ivec3(setup.Blocks().size(), 1, 1));
}


TEST(BlockSetup, SetupBlocks_Variable_DHWC) {
  TensorListShape<4> TLS({
    { 64, 480, 640, 3 },
    { 32, 768, 1024, 1 },
    { 25, 600, 800, 4 },
    { 1, 720, 1280, 3 },
    { 20, 480, 864, 2 },
    { 10, 576, 720, 1 }
  });

  BlockSetup<3, 3> setup;
  static_assert(setup.tensor_ndim == 4, "Incorrectly inferred tensor_ndim");
  setup.SetupBlocks(TLS);
  ASSERT_FALSE(setup.IsUniformSize());
  int prev = -1;
  BlockMap<3> map;
  for (auto &blk : setup.Blocks()) {
    if (blk.sample_idx != prev) {
      if (prev != -1) {
        ValidateBlockMap(map, TLS[prev].first<3>());
      }
      prev = blk.sample_idx;
      map = {};
    }
    auto &bz = map.inner[blk.start.z];
    bz.end = blk.end.z;
    bz.inner[blk.start.y].end = blk.end.y;
    auto &by = bz.inner[blk.start.y];
    by.end = blk.end.y;
    by.inner[blk.start.x].end = blk.end.x;
  }
  if (prev != -1)
    ValidateBlockMap(map, TLS[prev].first<3>());

  EXPECT_EQ(setup.GridDimVec(), ivec3(setup.Blocks().size(), 1, 1));
}


TEST(BlockSetup, SetupBlocks_Uniform_HWC) {
  const int W = 1920;
  const int H = 1080;
  TensorListShape<3> TLS({
    { H, W, 3 },
    { H, W, 3 },
    { H, W, 3 },
    { H, W, 3 },
    { H, W, 3 },
    { H, W, 3 },
    { H, W, 3 },
    { H, W, 3 },
    { H, W, 3 },
    { H, W, 3 }
  });

  BlockSetup<2, 2> setup;
  ivec2 def_block_size = { 256, 256 };
  setup.SetDefaultBlockSize(def_block_size);
  static_assert(setup.tensor_ndim == 3, "Incorrectly inferred tensor_ndim");
  setup.SetupBlocks(TLS);
  ASSERT_TRUE(setup.IsUniformSize());
  ivec3 expected_grid = {
    div_ceil(W, def_block_size.x),
    div_ceil(H, def_block_size.y),
    TLS.num_samples()
  };
  ivec3 block_dim = setup.BlockDimVec();
  ivec2 expected_block = {
    div_ceil(div_ceil(W, expected_grid.x), block_dim.x) * block_dim.x,
    div_ceil(div_ceil(H, expected_grid.y), block_dim.y) * block_dim.y
  };
  EXPECT_EQ(setup.UniformBlockSize(), expected_block);
  ivec3 grid = setup.GridDimVec();
  EXPECT_EQ(grid, expected_grid);
}

TEST(BlockSetup, SetupBlocks_Uniform_CDHW) {
  const int D = 64;
  const int W = 320;
  const int H = 240;
  TensorListShape<4> TLS({
    { 3, D, H, W },
    { 3, D, H, W },
    { 3, D, H, W },
    { 3, D, H, W },
    { 3, D, H, W },
    { 3, D, H, W },
    { 3, D, H, W },
    { 3, D, H, W },
    { 3, D, H, W },
    { 3, D, H, W }
  });

  BlockSetup<3, 0> setup;
  ivec3 def_block_size = { 256, 256, D };
  setup.SetDefaultBlockSize(def_block_size);
  static_assert(setup.tensor_ndim == 4, "Incorrectly inferred tensor_ndim");
  setup.SetupBlocks(TLS);
  ASSERT_TRUE(setup.IsUniformSize());

  while (volume(def_block_size) > 0x40000) {
    if (def_block_size.z > 1)
      def_block_size.z >>= 1;
  }
  ivec3 expected_grid = {
    div_ceil(W, def_block_size.x),
    div_ceil(H, def_block_size.y),
    TLS.num_samples()
  };
  ivec3 block_dim = setup.BlockDimVec();
  ivec3 expected_block = {
    div_ceil(div_ceil(W, expected_grid.x), block_dim.x) * block_dim.x,
    div_ceil(div_ceil(H, expected_grid.y), block_dim.y) * block_dim.y,
    D
  };
  EXPECT_EQ(setup.UniformBlockSize(), expected_block);
  ivec3 grid = setup.GridDimVec();
  EXPECT_EQ(grid, expected_grid);
}

}  // namespace kernels
}  // namespace dali
