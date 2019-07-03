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
  ivec<ndim> out_size, out_strides, in_size, in_strides;
  int channels;
};

template <int ndim>
struct BlockDesc {
  int sample_idx;
  ivec<ndim> start, end;
};

template <int ndim>
class WarpSetup {
  static_assert(ndim == 2 || ndim == 3,
    "Warping is defined only for 2D and 3D data with interleaved channels");

 public:
  static constexpr int tensor_dim = ndim+1;
  using SampleDesc = warp::SampleDesc<ndim>;
  using BlockDesc = warp::BlockDesc<ndim>;

  KernelRequirements Setup(const TensorListShape<tensor_dim> &output_shape,
                           bool force_variable_size = false) {
    ScratchpadEstimator se;
    is_uniform_ = !force_variable_size && is_uniform(output_shape);
    if (is_uniform_)
      UniformSizeSetup(se, output_shape);
    else
      VariableSizeSetup(se, output_shape);

    KernelRequirements req = {};
    req.output_shapes = { std::move(output_shape) };
    req.scratch_sizes = se.sizes;
    return req;
  }

  TensorListShape<tensor_dim> GetOutputShape(
      const TensorListShape<tensor_dim> &in_shape,
      span<const TensorShape<ndim>> output_sizes) {
    TensorListShape<tensor_dim> shape;
    shape.resize(in_shape.num_samples(), tensor_dim);
    for (int i = 0; i < in_shape.num_samples(); i++) {
      int channels = in_shape.tensor_shape_span(i).back();
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
      int channels = in_shape.tensor_shape_span(i).back();
      auto out_tensor_shape = out_shape.template tensor_shape<tensor_dim>(i);
      auto required_tensor_shape = shape_cat(output_sizes[i], channels);
      DALI_ENFORCE(out_tensor_shape == required_tensor_shape,
        "Invalid output tensor shape for sample: " + std::to_string(i));
    }
  }

  template <typename Backend, typename OutputType, typename InputType>
  void PrepareSamples(const OutList<Backend, OutputType, tensor_dim> &out,
                      const InList<Backend, InputType, tensor_dim> &in) {
    assert(out.num_samples() == in.num_samples());
    samples_.resize(in.num_samples());
    for (int i = 0; i < in.num_samples(); i++) {
      SampleDesc &sample = samples_[i];
      sample.input = in.tensor_data(i);
      sample.output = out.tensor_data(i);
      auto out_shape = out.tensor_shape(i);
      auto in_shape = in.tensor_shape(i);
      int channels = out_shape[ndim];
      sample.channels = channels;
      sample.out_size = size2vec(out_shape);
      sample.in_size = size2vec(in_shape);

      sample.out_strides.x = channels;
      sample.out_strides.y = sample.out_size.x * sample.out_strides.x;

      sample.in_strides.x = channels;
      sample.in_strides.y = sample.in_size.x * sample.in_strides.x;
    }
  }

  dim3 BlockDim() const {
    return dim3(block_dim_.x, block_dim_.y, block_dim_.z);
  }

  dim3 GridDim() const {
    return dim3(grid_dim_.x, grid_dim_.y, grid_dim_.z);
  }

  span<const SampleDesc> Samples() const { return make_span(samples_); }

  span<const BlockDesc> Blocks() const { return make_span(blocks_); }


  ivec<ndim> UniformOutputSize() const {
    assert(is_uniform_);
    return uniform_output_size_;
  }

  ivec<ndim> UniformBlockSize() const {
    assert(is_uniform_);
    return uniform_block_size_;
  }

  bool IsUniformSize() const { return is_uniform_; }

 private:
  std::vector<SampleDesc> samples_;
  std::vector<BlockDesc> blocks_;
  ivec3 block_dim_{32, 32, 1};
  ivec3 grid_dim_{1, 1, 1};
  ivec<ndim> uniform_block_size_, uniform_output_size_;
  bool is_uniform_ = false;

  static inline ivec<ndim> size2vec(const TensorShape<tensor_dim> &size) {
    ivec<ndim> v;
    for (int i = 0; i < ndim; i++)
      v[i] = size[ndim-1-i];
    return v;
  }

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

  static ivec2 BlockSize(const TensorShape<3> &shape) {
    (void)shape;
    return { 256, 256 };
  }

  static ivec3 BlockSize(const TensorShape<4> &shape) {
    (void)shape;
    int z = std::max<int>(1, volume(shape.last<3>()) / 65536);
    return { 256, 256, z };
  }

  void VariableSizeSetup(ScratchpadEstimator &se,
                         const TensorListShape<tensor_dim> &output_shape) {
    for (int i = 0; i < output_shape.num_samples(); i++) {
      ivec<ndim> block_size = BlockSize(output_shape[i]);
      MakeBlocks(i, size2vec(output_shape[i]), block_size);
    }
    se.add<SampleDesc>(AllocType::GPU, output_shape.num_samples());
    se.add<BlockDesc>(AllocType::GPU, blocks_.size());
  }

  void UniformSizeSetup(ScratchpadEstimator &se,
                        const TensorListShape<tensor_dim> &output_shape) {
    if (output_shape.empty())
      return;
    uniform_output_size_ = size2vec(output_shape[0]);
    // Get the rough estimate of block size
    uniform_block_size_ = BlockSize(output_shape[0]);

    // Make the blocks as evenly distributed as possible over the target area,
    // but maintain alignment to CUDA block dim.
    for (int i = 0; i < 2; i++) {  // only XY dimensions
      int blocks_in_axis = div_ceil(uniform_output_size_[i], uniform_block_size_[i]);
      int even_block_size = div_ceil(uniform_output_size_[i], blocks_in_axis);
      uniform_block_size_[i] = align_up(even_block_size, block_dim_[i]);
    }

    grid_dim_ = {
      div_ceil(uniform_output_size_.x, uniform_block_size_.x),
      div_ceil(uniform_output_size_.y, uniform_block_size_.y),
      output_shape.num_samples()
    };

    se.add<SampleDesc>(AllocType::GPU, output_shape.num_samples());
  }
};

}  // namespace warp
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_WARP_SETUP_CUH_
