// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <random>
#include <vector>
#include "dali/kernels/imgproc/structure/label_bbox.h"

namespace dali {
namespace kernels {

TEST(LabelBBoxes, 1D) {
  const int N = 10;
  const int labels[N] = {
    1, -1, 0, 1, 0, 0, 1, 4, 1, 3
  };
  std::vector<Box<1, int>> boxes(5);  // make room for box at -1 and 4

  label_bbox::GetLabelBoundingBoxes(
      make_span(boxes.data()+1, 3), make_tensor_cpu<1>(labels, { N }), 0);

  // remapped labels:
  // 0, -1, X, 0, X, X, 0, 3, 0, 2
  // X - background
  // -1 and 3 are out of bounds and should not produce a bounding box
  ASSERT_EQ(boxes[1-1], (Box<1, int>{})) << "Corruption: found a box for out-of-bounds label.";
  ASSERT_EQ(boxes[1+3], (Box<1, int>{})) << "Corruption: found a box for out-of-bounds label.";
  EXPECT_EQ(boxes[1+0], (Box<1, int>(0, 9)));
  EXPECT_EQ(boxes[1+1], (Box<1, int>(0, 0)));  // label 2 was absent, so ther's no box 1
  EXPECT_EQ(boxes[1+2], (Box<1, int>(9, 10)));

  // re-run with more room for boxes - should accommodate label 4
  label_bbox::GetLabelBoundingBoxes(
      make_span(boxes.data()+1, 4), make_tensor_cpu<1>(labels, { N }), 0);
  ASSERT_EQ(boxes[1-1], (Box<1, int>{})) << "Corruption: found a box for out-of-bounds label.";
  EXPECT_EQ(boxes[1+0], (Box<1, int>(0, 9)));
  EXPECT_EQ(boxes[1+1], (Box<1, int>(0, 0)));  // label 2 was absent, so ther's no box 1
  EXPECT_EQ(boxes[1+2], (Box<1, int>(9, 10)));
  EXPECT_EQ(boxes[1+3], (Box<1, int>(7, 8)));
}



TEST(LabelBBoxes, NegativeBackground) {
  const int N = 10;
  const int labels[N] = {
    1, -1, 0, 1, 0, 0, 1, 2, 1, 3
  };
  std::vector<Box<1, int>> boxes(4);
  label_bbox::GetLabelBoundingBoxes(
      make_span(boxes.data(), 4), make_tensor_cpu<1>(labels, { N }), -1);

  // no label remapping should occur
  EXPECT_EQ(boxes[0], (Box<1, int>(2, 6)));
  EXPECT_EQ(boxes[1], (Box<1, int>(0, 9)));
  EXPECT_EQ(boxes[2], (Box<1, int>(7, 8)));
  EXPECT_EQ(boxes[3], (Box<1, int>(9, 10)));
}


TEST(LabelBBoxes, RemapDim) {
  const int N = 10;
  const int labels[N] = {
    1, 1, 0, 1, 0, 0, 1, 4, 1, 3
  };
  std::vector<Box<3, int>> boxes(4);

  label_bbox::GetLabelBoundingBoxes(
      make_span(boxes.data(), 4), make_tensor_cpu<3>(labels, { 1, N, 1 }), 0);

  // remapped labels:
  // 0, -1, X, 0, X, X, 0, 3, 0, 2
  // X - background
  // -1 and 3 are out of bounds and should not produce a bounding box
  EXPECT_EQ(boxes[0], (Box<3, int>(0, {1, 9, 1})));
  EXPECT_EQ(boxes[1], (Box<3, int>(0, 0)));  // label 2 was absent, so ther's no box 1
  EXPECT_EQ(boxes[2], (Box<3, int>({0, 9, 0}, {1, 10, 1})));
  EXPECT_EQ(boxes[3], (Box<3, int>({0, 7, 0}, {1, 8, 1})));
}

TEST(LabelBBoxes, Degenerate) {
  int only_value = 1;
  auto input = make_tensor_cpu<3>(&only_value, {1, 1, 1});
  std::vector<Box<3, int>> boxes(3);
  label_bbox::GetLabelBoundingBoxes(make_span(boxes), input, 10);
  EXPECT_EQ(boxes[0], (Box<3, int>{}));
  EXPECT_EQ(boxes[1], (Box<3, int>({0, 0, 0}, {1, 1, 1})));
  EXPECT_EQ(boxes[2], (Box<3, int>{}));
}

TEST(LabelBBoxes, Random5D) {
  TensorShape<5> shape = { 5, 6, 7, 8, 9 };
  std::mt19937_64 rng;
  std::uniform_real_distribution<float> uni(0, 1);
  std::normal_distribution<float> norm(0, 1);
  const int64_t N = volume(shape);
  vector<int> data(N);
  auto in = make_tensor_cpu(data.data(), shape);

  const int nlbl = 32;
  std::vector<Box<5, int>> ref, out;
  ref.resize(nlbl);
  out.resize(nlbl);

  for (int lbl = 0; lbl < nlbl; lbl++) {
    vec<5> mean, sigma;
    for (int d = 0; d < 5; d++) {
      mean[d] = std::floor(uni(rng) * shape[d]);
      sigma[d] = uni(rng) * shape[d] * 0.5;
    }
    ivec<5> p;
    for (int i = 0; i < 20; i++) {
      do {
        for (int d = 0; d < 5; d++) {
            do {
            p[d] = static_cast<int>(std::floor(mean[d] + sigma[d] * norm(rng)));
            } while (p[d] < 0 || p[d] >= shape[d]);
        }
      } while (*in(p));  // don't overwrite previous points - this could invalidate boxes

      if (i == 0) {
        ref[lbl].lo = p;
        ref[lbl].hi = p + 1;
      } else {
        ref[lbl].lo = min(ref[lbl].lo, p);
        ref[lbl].hi = max(ref[lbl].hi, p + 1);
      }
      *in(p) = lbl + 1;
    }
  }

  label_bbox::GetLabelBoundingBoxes(make_span(out), in, 0);
  for (int lbl = 0; lbl < nlbl; lbl++) {
    EXPECT_EQ(out[lbl], ref[lbl]) << " @label " << lbl;
  }
}

}  // namespace kernels
}  // namespace dali
