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

/**
 * @brief Block descriptor specifying multidimensional range for given sample.
 */
template <int ndim>
struct BlockDesc {
  int sample_idx;
  ivec<ndim> start, end;
};

/**
 * @brief Block descriptor specifying range in given sample.
 *
 * Specialization for 1 dim to support 64bit addressing range.
 */
template <>
struct BlockDesc<1> {
  int sample_idx;
  i64vec<1> start, end;
};

/**
 * @brief A utility for calculating block layout for GPU kernels
 * @tparam _ndim         number dimensions to take into account while calculating the layout
 * @tparam _channel_dim  dimension in which channels are stored; channel dimension does not
 *                       participate in layout calculation \n
 *                       In cases where channel dimension can or should participate in layout
 *                       calculation, do not specify channel dimension and treat it as an
 *                       additional spatial dimension (e.g. for linear operations in CHW layout)\n
 *                       -1 indicates there are only spatial dimensions, all of which
 *                       participate in layout calculation.
 *
 * Typical usage:
 * * Generate blocks based on the shape by calling SetupBlocks.
 * * Copy the calculated BlockDesc to GPU - can be accessed by Blocks()
 * * Run the kernel with GridDim() and BlockDim() and pass the BlockDesc to it.
 *
 * Each kernel block should process the given multidimensional data range [start, end), for
 * given sample. Typically additional array of sample descriptors with input/output pointers
 * and per-sample parameters is passed.
 *
 * __global__ void
 * ProcessingKernel(const SampleDescriptor<2> *samples, const BlockDesc<2> *blocks) {
 *   const auto &block = blocks[blockIdx.x];
 *   const auto &sample = samples[block.sample_idx];
 *
 *   const auto *in = sample.in;
 *   auto *out = sample.out;
 *
 *   for (int y = threadIdx.y + block.start.y; y < block.end.y; y += blockDim.y) {
 *     for (int x = threadIdx.x + block.start.x; x < block.end.x; x += blockDim.x) {
 *       out[y * sample.out_pitch.x + x] = Foo(in[y * sample.in_pitch.x + x], sample.param);
 *     }
 *   }
 * }
 *
 * Note that by default for non-uniform blocks, the BlockDim.z == 1 - all dimensions apart from
 * the outermost two should be iterated over programatically and not by blocks of threads.
 *
 * @remark Depending on whether the uniform block coverage (see SetupBlocks) is used or not,
 * the calculated grid dimension allow to iterate over blocks with blockIdx.z or blockIdx.x
 * respectively.
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

  using coord_vec = std::conditional_t<ndim == 1, i64vec<1>, ivec<ndim>>;

  using coord_t = std::conditional_t<ndim == 1, int64_t, int32_t>;

  /**
   * @brief Configure how much data the block size will have to cover.
   *
   * The block will cover block_volume_scale * volume(BlockSize())
   */
  explicit BlockSetup(int block_volume_scale) : block_volume_scale_(block_volume_scale) {
    coord_vec block_size(1);
    for (int i = 0; i < ndim && i < 2; i++)
      block_size[i] = 256;
    SetDefaultBlockSize(block_size);
    if (ndim == 1) {
      block_dim_ = {256, 1, 1};
    }
  }

  BlockSetup() : BlockSetup(4) {}

  /**
   * @brief Generate block descriptors for given shape.
   *
   * @param output_shape - shape to cover with blocks
   * @param force_variable_size - true to always treat the shape as non-uniform
   *
   * @remark If the input is detected to be uniform and the `force_variable_size` is not used,
   * the blocks should be indexed with blockIdx.z instead of blockIdx.x
   */
  void SetupBlocks(const TensorListShape<tensor_ndim> &output_shape,
                   bool force_variable_size = false) {
    blocks_.clear();
    is_uniform_ = !force_variable_size && is_uniform(output_shape);

    if (is_uniform_)
      UniformSizeSetup(output_shape);
    else
      VariableSizeSetup(output_shape);
  }

  /**
   * @brief Prepare TensorListShape based on `in_shape` number of channels and spatial output
   * dimensions provided in `output_sizes`.
   */
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

  /**
   * @brief Check if `out_shape` matches the number of channels of `in_shape` and spatial dimensions
   * of `output_sizes` as if it was generated with GetOutputShape(in_shape, output_sizes).
   */
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

  void SetDefaultBlockSize(coord_vec block_size) {
    default_block_size_ = block_size;
    max_block_elements_ = block_volume_scale_ * volume(block_size);
  }

  span<const BlockDesc> Blocks() const { return make_span(blocks_); }

  coord_vec UniformOutputSize() const {
    assert(is_uniform_);
    return uniform_output_size_;
  }

  coord_vec UniformBlockSize() const {
    assert(is_uniform_);
    return uniform_block_size_;
  }

  coord_t UniformZBlocksPerSample() const {
    assert((z_blocks_per_sample_ & (z_blocks_per_sample_-1)) == 0 &&
           "z_block_per_sample_ must be a power of 2");
    return z_blocks_per_sample_;
  }

  bool IsUniformSize() const { return is_uniform_; }

  static inline coord_vec shape2size(const TensorShape<tensor_ndim> &shape) {
    return shape2vec<ndim, coord_t>(skip_dim<channel_dim>(shape));
  }

 private:
  std::vector<BlockDesc> blocks_;
  ivec3 block_dim_{32, 32, 1};
  ivec3 grid_dim_{1, 1, 1};
  coord_vec uniform_block_size_, uniform_output_size_;
  coord_vec default_block_size_;
  int z_blocks_per_sample_ = 1;
  coord_t max_block_elements_;
  bool is_uniform_ = false;
  int block_volume_scale_ = 4;

  template <int d>
  void MakeBlocks(BlockDesc blk, coord_vec size, coord_vec block_size,
                  std::integral_constant<int, d>) {
    for (coord_t i = 0; i < size[d]; i += block_size[d]) {
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

  void MakeBlocks(int sample_idx, coord_vec size, coord_vec block_size) {
    BlockDesc blk;
    blk.sample_idx = sample_idx;
    MakeBlocks<ndim-1>(blk, size, block_size, {});
    grid_dim_ = ivec3(blocks_.size(), 1, 1);
  }

  template <int n>
  std::enable_if_t<(n >= 2), coord_vec> VariableBlockSize(const coord_vec &shape) const {
    coord_vec ret;
    for (int i = 0; i < ndim; i++) {
      switch (i) {
      case 0:
      case 1:
        ret[i] = std::min<coord_t>(shape[i], default_block_size_[i]);
        break;
      case 2:
        ret[i] = std::max<coord_t>(1, ret[0]*ret[1] / max_block_elements_);
        break;
      default:
        ret[i] = 1;
      }
    }
    return min(shape, ret);
  }

  template <int n>
  std::enable_if_t<(n == 1), coord_vec> VariableBlockSize(const coord_vec &shape) const {
    return min(shape, coord_vec{max_block_elements_});
  }

  template <int n>
  std::enable_if_t<(n > 2), coord_vec> SetZBlocksPerSample(coord_vec block) {
    z_blocks_per_sample_ = 1;
    coord_t depth = block[2];
    while (volume(block) > max_block_elements_ && block[2] > 0) {
      z_blocks_per_sample_ <<= 1;
      block[2] = div_ceil(depth, z_blocks_per_sample_);
    }
    return block;
  }

  template <int n>
  std::enable_if_t<(n <= 2), coord_vec> SetZBlocksPerSample(coord_vec block) {
    z_blocks_per_sample_ = 1;
    return block;  // no Z coordinate to adjust
  }

  coord_vec SetUniformBlockSize(const coord_vec &shape) {
    coord_vec ret;
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

    ret = SetZBlocksPerSample<ndim>(ret);
    uniform_block_size_ = min(shape, ret);
    return uniform_block_size_;
  }

  void VariableSizeSetup(const TensorListShape<tensor_ndim> &output_shape) {
    for (int i = 0; i < output_shape.num_samples(); i++) {
      coord_vec size = shape2size(output_shape[i]);
      coord_vec block_size = VariableBlockSize<ndim>(size);
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

    constexpr int xy_dim = ndim > 1 ? 2 : 1;
    for (int i = 0; i < xy_dim; i++) {  // only XY dimensions
      coord_t blocks_in_axis = div_ceil(uniform_output_size_[i], uniform_block_size_[i]);
      coord_t even_block_size = div_ceil(uniform_output_size_[i], blocks_in_axis);

      // a note on div_ceil + mul combo:
      // can't use align_up, because block_dim_ does not need to be a power of 2
      uniform_block_size_[i] = div_ceil(even_block_size, block_dim_[i]) * block_dim_[i];
    }

    grid_dim_ = {
      div_ceil(uniform_output_size_[0], uniform_block_size_[0]),
      ndim > 1 ? div_ceil(uniform_output_size_[1], uniform_block_size_[1]) : 1,
      output_shape.num_samples() * z_blocks_per_sample_
    };
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_BLOCK_SETUP_H_
