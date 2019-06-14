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
#include <opencv2/imgcodecs.hpp>
#include <map>
#include "dali/kernels/imgproc/warp_gpu.h"
#include "dali/kernels/imgproc/warp/affine.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/test/mat2tensor.h"
#include "dali/test/dali_test_config.h"
#include "dali/kernels/test/test_tensors.h"

namespace dali {
namespace kernels {

void IsWarpKernelValid() {
  check_kernel<WarpGPU<AffineMapping2D, 2, float, uint8_t, float, DALI_INTERP_LINEAR>>();
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

TEST(WarpSetup, Setup_Blocks) {
  TensorListShape<3> TLS({
    { 480, 640, 3 },
    { 768, 1024, 3 },
    { 600, 800, 3 },
    { 720, 1280, 3 },
    { 480, 864, 3 },
    { 576, 720, 3 }
  });

  warp::WarpSetup<2> setup;
  setup.Setup(TLS);
  int prev = -1;
  BlockMap<2> map;
  for (auto &blk : setup.blocks_) {
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
}

TEST(WarpGPU, Affine) {
  AffineMapping<2> mapping = mat2x3{{
    { 0, 1, 0 },
    { 1, 0, 0 }
  }};

  cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/alley.png");
  auto gpu_img = copy_as_tensor<AllocType::GPU>(cv_img);
  auto img_tensor = gpu_img.first;

  TensorListView<StorageGPU, uint8_t, 3> in_list;
  in_list.resize(1, 3);
  in_list.shape.make_uniform(1, img_tensor.shape);
  in_list.data[0] = img_tensor.data;

  WarpGPU<decltype(mapping), 2, uint8_t, uint8_t, BorderClamp, DALI_INTERP_NN> warp;

  TensorShape<2> out_shape = { img_tensor.shape[1], img_tensor.shape[0] };
  KernelContext ctx = {};
  KernelRequirements req = warp.Setup(ctx, in_list, { &out_shape, 1 }, { &mapping, 1 });
  TestTensorList<uint8_t> out;
  out.reshape(req.output_shapes[0]);
  warp.Run(ctx, out.gpu(0), in_list, { &out_shape, 1 }, { &mapping, 1 });
}

}  // namespace kernels
}  // namespace dali
