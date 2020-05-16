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

#include <gtest/gtest.h>
#include "dali/kernels/reduce/reduce_drop_dims.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

TEST(DropDims, Simplify) {
  DropDims<3> dd;
  unsigned in_mask = 0b10011, out_mask = 0xffffffffu;
  int64_t in_shape[5] = { 2, 3, 4, 5, 6 };
  int64_t out_shape[5];
  int d = dd.simplify(out_shape, out_mask, in_shape, in_mask);
  EXPECT_EQ(d, 3);
  EXPECT_EQ(out_mask, 0b101);
  EXPECT_EQ(out_shape[0], 2*3);
  EXPECT_EQ(out_shape[1], 4*5);
  EXPECT_EQ(out_shape[2], 6);

  in_mask = 0b00110;
  out_mask = 0xffffffffu;
  d = dd.simplify(out_shape, out_mask, in_shape, in_mask);
  EXPECT_EQ(d, 3);
  EXPECT_EQ(out_mask, 0b010);
  EXPECT_EQ(out_shape[0], 2);
  EXPECT_EQ(out_shape[1], 3*4);
  EXPECT_EQ(out_shape[2], 5*6);

  in_mask = 0;
  out_mask = 0xffffffffu;
  d = dd.simplify(out_shape, out_mask, in_shape, in_mask);
  EXPECT_EQ(d, 1);
  EXPECT_EQ(out_mask, 0);
  EXPECT_EQ(out_shape[0], 2*3*4*5*6);
}

TEST(DropDims, NoOp) {
  DropDims<3> dd;
  for (int i = 0; i < 10; i++)
    EXPECT_EQ(dd.reindex(i), i);
}

TEST(DropDims, Outer) {
  int h = 32;
  int w = 40;
  int shape[] = { h, w };
  DropDims<3> dd(shape, 0b01);
  for (int y = 0, idx = 0; y < h; y++) {
    for (int x = 0; x < w; x++, idx++) {
      EXPECT_EQ(dd.reindex(idx), x);
    }
  }
}

TEST(DropDims, Inner) {
  int h = 32;
  int w = 40;
  int shape[] = { h, w };
  DropDims<3> dd(shape, 0b10);
  for (int y = 0, idx = 0; y < h; y++) {
    for (int x = 0; x < w; x++, idx++) {
      EXPECT_EQ(dd.reindex(idx), y);
    }
  }
}

TEST(DropDims, Middle) {
  int d = 3;
  int h = 4;
  int w = 5;
  int shape[] = { d, h, w };
  DropDims<3> dd(shape, 0b010);
  for (int z = 0, idx = 0; z < d; z++) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++, idx++) {
        EXPECT_EQ(dd.reindex(idx), z * w + x);
      }
    }
  }
}

TEST(DropDims, TwoOuter) {
  int d = 3;
  int h = 4;
  int w = 5;
  int shape[] = { d, h, w };
  DropDims<3> dd(shape, 0b101);
  for (int z = 0, idx = 0; z < d; z++) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++, idx++) {
        EXPECT_EQ(dd.reindex(idx), y);
      }
    }
  }
}

TEST(DropDims, Multi4Odd) {
  int shape[] = { 3, 4, 5, 6 };
  DropDims<3> dd(shape, 0b1010);
  for (int i = 0, idx = 0; i < shape[0]; i++)
    for (int j = 0; j < shape[1]; j++)
      for (int k = 0; k < shape[2]; k++)
        for (int l = 0; l < shape[3]; l++, idx++) {
            int ridx = i * shape[2] + k;
            EXPECT_EQ(dd.reindex(idx), ridx);
          }
}

TEST(DropDims, Multi4Even) {
  int shape[] = { 3, 4, 5, 6 };
  DropDims<3> dd(shape, 0b0101);
  for (int i = 0, idx = 0; i < shape[0]; i++)
    for (int j = 0; j < shape[1]; j++)
      for (int k = 0; k < shape[2]; k++)
        for (int l = 0; l < shape[3]; l++, idx++) {
            int ridx = j * shape[3] + l;
            EXPECT_EQ(dd.reindex(idx), ridx);
          }
}

TEST(DropDims, TrailingOne) {
  int shape[] = { 3, 4, 5, 6, 1 };
  DropDims<3> dd(shape, 0b01010);
  for (int i = 0, idx = 0; i < shape[0]; i++)
    for (int j = 0; j < shape[1]; j++)
      for (int k = 0; k < shape[2]; k++)
        for (int l = 0; l < shape[3]; l++, idx++) {
            int ridx = (i * shape[2] + k) * shape[4];
            EXPECT_EQ(dd.reindex(idx), ridx);
        }
}

TEST(DropDims, Multi5All) {
  int base_shape[] = { 3, 4, 5, 6, 7 };
  for (unsigned degeneracy = 0; degeneracy <= 0b11111u; degeneracy++) {
    int shape[5];
    for (int d = 0; d < 5; d++) {
      shape[d] = degeneracy & (1u << d) ? 1 : base_shape[d];
    }
    int rshape[5];
    for (unsigned mask = 0; mask <= 0b11111u; mask++) {
      for (int d = 0; d < 5; d++) {
        rshape[d] = mask & (1u << d) ? 1 : shape[d];
      }
      DropDims<3> dd(shape, mask);
      for (int i = 0, idx = 0; i < shape[0]; i++) {
        int ri = mask & 0b00001 ? 0 : i;
        for (int j = 0; j < shape[1]; j++) {
          int rj = mask & 0b00010 ? 0 : j;
          for (int k = 0; k < shape[2]; k++) {
            int rk = mask & 0b00100 ? 0 : k;
            for (int l = 0; l < shape[3]; l++) {
              int rl = mask & 0b01000 ? 0 : l;
              for (int m = 0; m < shape[4]; m++, idx++) {
                int rm = mask & 0b10000 ? 0 : m;
                int ridx =
                  ((((ri * rshape[1] + rj) * rshape[2]) + rk) * rshape[3] + rl) * rshape[4] + rm;
                ASSERT_EQ(dd.reindex(idx), ridx);
              }
            }
          }
        }
      }
    }
  }
}

TEST(DropDims, CollapseUnitDims) {
  int shape[] = { 3, 4, 5, 1, 1, 6, 7 };
  // reduce:      ^     ^^^^        ^
  //
  // collapse unit dims, with different reduce/non-reduce flag

  DropDims<3> dd(shape, 0b1001101);
  for (int i = 0, idx = 0; i < shape[0]; i++)
    for (int j = 0; j < shape[1]; j++)
      for (int k = 0; k < shape[2]; k++)
        for (int l = 0; l < shape[5]; l++)
          for (int m = 0; m < shape[6]; m++, idx++) {
            int ridx = j * shape[5] + l;
            EXPECT_EQ(dd.reindex(idx), ridx);
          }
}

}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali
