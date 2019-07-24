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
#include "dali/kernels/common/block_setup.h"

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

namespace {

template <int dim>
struct BlockMap {
  unsigned end;
  std::map<unsigned, BlockMap<dim-1>> inner;
};

template <>
struct BlockMap<0> {
  unsigned end;
  // dummy - to avoid specialization
  const bool inner = false;
};

template <int dim>
bool operator==(const BlockMap<dim> &a, const BlockMap<dim> &b) {
  return a.end == b.end && a.inner == b.inner;
}

inline void ValidateBlockMap(const BlockMap<0> &map, const TensorShape<0> &shape) {}

/// @brief Check that the output shape is covered with rectangular grid.
///
/// The grid cells must be aligned between rows/slices, but don't have to be uniform
/// - typically the last cell will be smaller and that's expected.
template <int dim>
void ValidateBlockMap(const BlockMap<dim> &map, const TensorShape<dim> &shape) {
  ASSERT_FALSE(map.inner.empty());
  unsigned i = 0;
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
      // Validate the first slice recurisvely - the remaining slices should be equal and
      // therefore don't require validation.
      ValidateBlockMap(p.second, shape.template last<dim-1>());
    }
  }
}

}  // namespace


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


TEST(BlockSetup, SetupBlocks_Uniform) {
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

}  // namespace kernels
}  // namespace dali
