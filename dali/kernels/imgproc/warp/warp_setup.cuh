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

#ifndef DALI_KERNELS_IMGPROC_WARP_WARP_SETUP_CUH_
#define DALI_KERNELS_IMGPROC_WARP_WARP_SETUP_CUH_

#include <vector>
#include <utility>
#include "dali/kernels/kernel.h"
#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {
namespace warp {

template <int ndim>
struct SampleDesc {
  const void *__restrict__ input;
  void *__restrict__ output;
  uvec<ndim> out_size, out_strides, in_size, in_strides;
  unsigned channels;
};

template <int ndim>
struct BlockDesc {
  int sample_idx;
  uvec<ndim> start, end;
};

template <int ndim>
struct WarpSetup {
  static_assert(ndim == 2 || ndim == 3,
    "Warping is defined only for 2D and 3D data with interleaved channels");

  static constexpr int tensor_dim = ndim+1;

  std::vector<SampleDesc<ndim>> samples_;
  std::vector<BlockDesc<ndim>> blocks_;
  uvec3 block_dim_{32, 32, 1};
  uvec<ndim> uniform_block_size_;

  bool is_uniform_ = false;

  static inline uvec<ndim> size2vec(const TensorShape<tensor_dim> &size) {
    uvec<ndim> v;
    for (int i = 0; i < ndim; i++)
      v[i] = size[ndim-1-i];
    return v;
  }


  unsigned BlockDim(int d) const {
    return d < 3 ? block_dim_[d] : 1;
  }

  template <int d>
  void MakeBlocks(BlockDesc<ndim> blk, uvec<ndim> size, uvec<ndim> block_size,
                  std::integral_constant<int, d>) {
    for (unsigned i = 0; i < size[d]; i += block_size[d]) {
      blk.start[d] = i;
      blk.end[d] = std::min(i + block_size[d], size[d]);
      if (d > 0) {
        constexpr int next_d = d > 0 ? d - 1: d;  // prevent infinite template expansion
        MakeBlocks(blk, size, block_size, std::integral_constant<int, next_d>());
      } else {
        blocks_.push_back(blk);
      }
    }
  }

  void MakeBlocks(int sample_idx, uvec<ndim> size, uvec<ndim> block_size) {
    BlockDesc<ndim> blk;
    blk.sample_idx = sample_idx;
    MakeBlocks<ndim-1>(blk, size, block_size, {});
  }

  static uvec2 BlockSize(const TensorShape<3> &shape) {
    (void)shape;
    return { 256, 256 };
  }

  static uvec3 BlockSize(const TensorShape<4> &shape) {
    (void)shape;
    unsigned z = std::max<unsigned>(1, volume(shape.last<3>()) / 65536);
    return { 256, 256, z };
  }

  void VariableSizeSetup(KernelRequirements &req,
                         const TensorListShape<tensor_dim> &output_shape) {
    for (int i = 0; i < output_shape.num_samples(); i++) {
      uvec<ndim> block_size = BlockSize(output_shape[i]);
      MakeBlocks(i, size2vec(output_shape[i]), block_size);
    }
    ScratchpadEstimator se;
    se.add<SampleDesc<ndim>>(AllocType::GPU, output_shape.num_samples());
    se.add<BlockDesc<ndim>>(AllocType::GPU, blocks_.size());
    req.scratch_sizes = se.sizes;
  }

  void UniformSizeSetup(KernelRequirements &req,
                        const TensorListShape<tensor_dim> &output_shape) {
    if (output_shape.empty())
      return;
    uniform_block_size_ = BlockSize(output_shape[0]);
  }

  KernelRequirements Setup(const TensorListShape<tensor_dim> &output_shape) {
    KernelRequirements req;
    is_uniform_ = is_uniform(output_shape);
    if (is_uniform_)
      UniformSizeSetup(req, output_shape);
    else
      VariableSizeSetup(req, output_shape);

    req.output_shapes = { std::move(output_shape) };
    return req;
  }

  TensorListShape<tensor_dim> GetOutputShape(
      const TensorListShape<tensor_dim> &in_shape,
      span<const TensorShape<ndim>> output_sizes) {
    TensorListShape<tensor_dim> shape;
    shape.resize(in_shape.num_samples(), tensor_dim);
    for (int i = 0; i < in_shape.num_samples(); i++) {
      int channels = in_shape.tensor_shape_span(i)[tensor_dim-1];
      shape.set_tensor_shape(i, shape_cat(output_sizes[i], channels));
    }
    return shape;
  }


  template <int shape_ndim>
  void ValidateOutputShape(
      const TensorListShape<shape_ndim> &out_shape,
      const TensorListShape<tensor_dim> &in_shape,
      span<const TensorShape<ndim>> output_sizes) {
    TensorListShape<tensor_dim> shape;
    shape.resize(in_shape.num_samples(), tensor_dim);
    for (int i = 0; i < in_shape.num_samples(); i++) {
      int channels = in_shape.tensor_shape_span(i)[tensor_dim-1];
      auto out_tensor_shape = out_shape.template tensor_shape<tensor_dim>(i);
      auto required_tensor_shape = shape_cat(output_sizes[i], channels);
      DALI_ENFORCE(out_tensor_shape == required_tensor_shape,
        "Invalid output tensor shape for sample: " + std::to_string(i));
    }
    return shape;
  }

  KernelRequirements Setup(
      const TensorListShape<tensor_dim> &in_shape,
      span<const TensorShape<ndim>> output_sizes) {
    auto out_shape = GetOutputShape(in_shape, output_sizes);
    return Setup(std::move(out_shape));
  }
};

}  // namespace warp
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_WARP_SETUP_CUH_
