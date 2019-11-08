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

#ifndef DALI_KERNELS_COMMON_BLOCK_SETUP_H_
#define DALI_KERNELS_COMMON_BLOCK_SETUP_H_

#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <utility>
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/roi.h"

namespace dali {
namespace kernels {

template <int ndim>
struct BlockDesc {
  int sample_idx;
  ivec<ndim> start, end;
};

/**
 * @brief A utility for calculating block layout for GPU kernels
 * @tparam _ndim         number dimensions to take into account while calculating the layout
 * @tparam _channel_dim  dimension in which channels are stored; channel dimension does not
 *                       participate in layout calculation \n
 *                       In cases where channel dimension can or should participate in layout
 *                       calaculation, do not specify channel dimenion and treat it as an
 *                       additional spatial dimension (e.g. for linear operations in CHW layout)\n
 *                       -1 indicates there are only spatial dimensions, all of which
 *                       participate in layout calculation.
 */
template <int _ndim, int _channel_dim>
class BlockSetup {
 public:
  static constexpr int ndim = _ndim;
  static constexpr int channel_dim = _channel_dim;
  static constexpr int tensor_ndim = (channel_dim < 0 ? ndim : ndim + 1);
  static_assert(channel_dim >= -1 && channel_dim <= ndim,
    "Channel dimension must be in range [0..ndim] or -1 (no channel dim)");
  using BlockDesc = kernels::BlockDesc<ndim>;

  BlockSetup() {
    ivec<ndim> block_size(1);
    for (int i = 0; i < ndim && i < 2; i++)
      block_size[i] = 256;
    SetDefaultBlockSize(block_size);
  }

  void SetupBlocks(const TensorListShape<tensor_ndim> &output_shape,
                   bool force_variable_size = false) {
    blocks_.clear();
    is_uniform_ = !force_variable_size && is_uniform(output_shape);

    if (is_uniform_)
      UniformSizeSetup(output_shape);
    else
      VariableSizeSetup(output_shape);
  }

  TensorListShape<tensor_ndim> GetOutputShape(
      const TensorListShape<tensor_ndim> &in_shape,
      span<const TensorShape<ndim>> output_sizes) {
    assert(in_shape.num_samples() == static_cast<int>(output_sizes.size()));
    TensorListShape<tensor_ndim> shape;
    shape.resize(in_shape.num_samples(), tensor_ndim);
    for (int i = 0; i < in_shape.num_samples(); i++) {
      auto out_tshape = shape.tensor_shape_span(i);
      int in_d = 0;
      for (int j = 0; j < tensor_ndim; j++) {
        out_tshape[j] = (j == channel_dim)
         ? in_shape.tensor_shape_span(i)[channel_dim]
         : output_sizes[i][in_d++];
      }
    }
    return shape;
  }


  void ValidateOutputShape(
      const TensorListShape<tensor_ndim> &out_shape,
      const TensorListShape<tensor_ndim> &in_shape,
      span<const TensorShape<ndim>> output_sizes) {
    TensorListShape<tensor_ndim> shape;
    shape.resize(in_shape.num_samples(), tensor_ndim);
    for (int i = 0; i < in_shape.num_samples(); i++) {
      auto out_tshape = out_shape[i];
      TensorShape<tensor_ndim> expected_shape;

      int in_d = 0;
      for (int j = 0; j < tensor_ndim; j++) {
        expected_shape[j] = (j == channel_dim)
         ? in_shape.tensor_shape_span(i)[channel_dim]
         : output_sizes[i][in_d++];
      }

      DALI_ENFORCE(out_tshape == expected_shape,
        "Invalid output tensor shape for sample: " + std::to_string(i));
    }
  }

  dim3 BlockDim() const {
    return dim3(block_dim_.x, block_dim_.y, block_dim_.z);
  }

  ivec3 BlockDimVec() const {
    return block_dim_;
  }

  void SetBlockDim(ivec3 block_dim) {
    block_dim_ = block_dim;
  }

  void SetBlockDim(dim3 block_dim) {
    block_dim_ = { block_dim.x, block_dim.y, block_dim.z };
  }

  dim3 GridDim() const {
    return dim3(grid_dim_.x, grid_dim_.y, grid_dim_.z);
  }

  ivec3 GridDimVec() const {
    return grid_dim_;
  }

  void SetDefaultBlockSize(ivec<ndim> block_size) {
    default_block_size_ = block_size;
    max_block_elements_ = 4*volume(block_size);
  }

  span<const BlockDesc> Blocks() const { return make_span(blocks_); }

  ivec<ndim> UniformOutputSize() const {
    assert(is_uniform_);
    return uniform_output_size_;
  }

  ivec<ndim> UniformBlockSize() const {
    assert(is_uniform_);
    return uniform_block_size_;
  }

  int UniformZBlocksPerSample() const {
    assert((z_blocks_per_sample_ & (z_blocks_per_sample_-1)) == 0 &&
           "z_block_per_sample_ must be a power of 2");
    return z_blocks_per_sample_;
  }

  bool IsUniformSize() const { return is_uniform_; }

  static inline ivec<ndim> shape2size(const TensorShape<tensor_ndim> &shape) {
    return shape2vec(skip_dim<channel_dim>(shape));
  }

 private:
  std::vector<BlockDesc> blocks_;
  ivec3 block_dim_{32, 32, 1};
  ivec3 grid_dim_{1, 1, 1};
  ivec<ndim> uniform_block_size_, uniform_output_size_;
  ivec<ndim> default_block_size_;
  int z_blocks_per_sample_ = 1;
  int max_block_elements_;
  bool is_uniform_ = false;

  template <int d>
  void MakeBlocks(BlockDesc blk, ivec<ndim> size, ivec<ndim> block_size,
                  std::integral_constant<int, d>) {
    for (int i = 0; i < size[d]; i += block_size[d]) {
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

  void MakeBlocks(int sample_idx, ivec<ndim> size, ivec<ndim> block_size) {
    BlockDesc blk;
    blk.sample_idx = sample_idx;
    MakeBlocks<ndim-1>(blk, size, block_size, {});
    grid_dim_ = ivec3(blocks_.size(), 1, 1);
  }

  ivec<ndim> VariableBlockSize(const ivec<ndim> &shape) const {
    ivec<ndim> ret;
    for (int i = 0; i < ndim; i++) {
      switch (i) {
      case 0:
      case 1:
        ret[i] = std::min<int>(shape[i], default_block_size_[i]);
        break;
      case 2:
        ret[i] = std::max<int>(1, ret[0]*ret[1] / max_block_elements_);
        break;
      default:
        ret[i] = 1;
      }
    }
    return min(shape, ret);
  }

  template <int n>
  std::enable_if_t<(n > 2), ivec<n>> SetZBlocksPerSample(ivec<n> block) {
    z_blocks_per_sample_ = 1;
    int depth = block[2];
    while (volume(block) > max_block_elements_ && block[2] > 0) {
      z_blocks_per_sample_ <<= 1;
      block[2] = div_ceil(depth, z_blocks_per_sample_);
    }
    return block;
  }

  template <int n>
  std::enable_if_t<(n <= 2), ivec<n>> SetZBlocksPerSample(ivec<n> block) {
    z_blocks_per_sample_ = 1;
    return block;  // no Z coordinate to adjust
  }

  ivec<ndim> SetUniformBlockSize(const ivec<ndim> &shape) {
    ivec<ndim> ret;
    for (int i = 0; i < ndim; i++) {
      switch (i) {
      case 0:
      case 1:
        ret[i] = default_block_size_[i];
        break;
      default:  // iterate dims other than XY
        ret[i] = shape[i];
        break;
      }
    }

    ret = SetZBlocksPerSample(ret);
    uniform_block_size_ = min(shape, ret);
    return uniform_block_size_;
  }

  void VariableSizeSetup(const TensorListShape<tensor_ndim> &output_shape) {
    for (int i = 0; i < output_shape.num_samples(); i++) {
      ivec<ndim> size = shape2size(output_shape[i]);
      ivec<ndim> block_size = VariableBlockSize(size);
      MakeBlocks(i, size, block_size);
    }
  }

  void UniformSizeSetup(const TensorListShape<tensor_ndim> &output_shape) {
    if (output_shape.empty())
      return;
    uniform_output_size_ = shape2size(output_shape[0]);
    // Get the rough estimate of block size
    uniform_block_size_ = SetUniformBlockSize(uniform_output_size_);

    // Make the blocks as evenly distributed as possible over the target area,
    // but maintain alignment to CUDA block dim.
    for (int i = 0; i < 2; i++) {  // only XY dimensions
      int blocks_in_axis = div_ceil(uniform_output_size_[i], uniform_block_size_[i]);
      int even_block_size = div_ceil(uniform_output_size_[i], blocks_in_axis);

      // a note on div_ceil + mul combo:
      // can't use align_up, because block_dim_ does not need to be a power of 2
      uniform_block_size_[i] = div_ceil(even_block_size, block_dim_[i]) * block_dim_[i];
    }

    grid_dim_ = {
      div_ceil(uniform_output_size_.x, uniform_block_size_.x),
      div_ceil(uniform_output_size_.y, uniform_block_size_.y),
      output_shape.num_samples() * z_blocks_per_sample_
    };
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_BLOCK_SETUP_H_
