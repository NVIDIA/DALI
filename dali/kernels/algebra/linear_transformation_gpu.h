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

#ifndef DALI_KERNELS_ALGEBRA_LINEAR_TRANSFORMATION_H_
#define DALI_KERNELS_ALGEBRA_LINEAR_TRANSFORMATION_H_

#include <vector>
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/imgproc/roi.h"

namespace dali {
namespace kernels {
namespace linear_transformation {
namespace detail {

using TransformationMatrixElementType = float;

template <class OutputType, class InputType, size_t N, size_t M, size_t spatial_ndims>
struct SampleDescriptor {
  const InputType *in;
  OutputType *out;
  ivec<spatial_ndims> in_size, in_strides, out_size, out_strides;
  int in_channels, out_channels;
  mat<N, M, TransformationMatrixElementType> transformation_matrix;
};


template <class OutputType, class InputType, size_t N, size_t M, size_t spatial_ndims>
std::vector<SampleDescriptor<OutputType, InputType, N, M, spatial_ndims>>
CreateSampleDescriptors
        (const OutListGPU<OutputType, spatial_ndims + 1> &out,
         const InListGPU<InputType, spatial_ndims + 1> &in,
         const std::vector<mat<N, M, TransformationMatrixElementType>> &transformation_matrices) {
  std::vector<SampleDescriptor<OutputType, InputType, N, M, spatial_ndims>> ret(
          in.num_samples());

  for (int i = 0; i < in.num_samples(); i++) {
    auto &sample = ret[i];
    sample.in = in[i].data;
    sample.out = out[i].data;

    auto get_size = [](const TensorShape<spatial_ndims + 1> &ts) -> auto {
        ivec<spatial_ndims> ret;
        for (int i = ret.size() - 1, j = 0; i >= 0; i--, j++) {
          ret[j] = ts[i];
        }
        return ret;
    };

    auto get_channels = [](const TensorShape<spatial_ndims + 1> &ts) -> auto {
        return ts.shape.back();
    };

    sample.in_size = get_size(in.tensor_shape(i));
    sample.out_size = get_size(out.tensor_shape(i));
    sample.in_channels = get_channels(in.tensor_shape(i));
    sample.out_channels = get_channels(out.tensor_shape(i));
    sample.in_strides = {sample.in_channels, sample.in_size.x * sample.in_channels};
    sample.out_strides = {sample.out_channels, sample.out_size.x * sample.out_channels};
    sample.transformation_matrix = transformation_matrices[i];
  }

  return ret;
}


}  // namespace detail


template <class OutputType, class InputType, size_t N, size_t M, size_t spatial_ndims>
void __global__ LinearTransformationKernel(
        const detail::SampleDescriptor<OutputType, InputType, N, M, spatial_ndims> *samples,
        const BlockDesc<spatial_ndims> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  const Surface2D<const InputType> in = {
          sample.in, sample.in_size.x, sample.in_size.y, sample.in_channels,
          sample.in_strides.x, sample.in_strides.y, 1
  };

  const Surface2D<OutputType> out = {
          sample.out, sample.out_size.x, sample.out_size.y, sample.out_channels,
          sample.out_strides.x, sample.out_strides.y, 1
  };

  for (int y = threadIdx.y + block.start.y; y < block.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + block.start.x; x < block.end.x; x += blockDim.x) {
      // Iterate over channels: `o` over output, `i` over input
      // It's a plain matrix multiplication
      for (int o = 0; o < N; o++) {
        OutputType val = 0;
        for (int i = 0; i < M; i++) {
          val += sample.transformation_matrix.at(o, i) * in(x, y, i);
        }
        out(x, y, o) = val;
      }
    }
  }
}


template <typename OutputType, typename InputType, size_t N, size_t M, size_t spatial_ndims>
class LinearTransformationGpu {
 private:
  static constexpr auto ndims_ = spatial_ndims + 1;
  using mat = ::dali::mat<N, M, detail::TransformationMatrixElementType>;
  using BlockDesc = kernels::BlockDesc<spatial_ndims>;
  using SampleDescriptor = detail::SampleDescriptor<OutputType, InputType, N, M, spatial_ndims>;

  std::vector<SampleDescriptor> sample_descriptors_;

 public:
  BlockSetup<spatial_ndims,
          spatial_ndims /* Assumed, that the channel dim is the last dim */> block_setup_;


  KernelRequirements
  Setup(KernelContext &context, const InListGPU<InputType, ndims_> &in,
        const std::vector<mat> &transformation_matrices,
        const std::vector<Roi<spatial_ndims>> &rois = {}) {
    DALI_ENFORCE(rois.empty() || rois.size() == static_cast<size_t>(in.num_samples()),
                 "Provide ROIs either for all or none input tensors");
    DALI_ENFORCE([=]() -> bool {
        for (const auto &roi : rois) {
          if (!all_coords(roi.hi >= roi.lo))
            return false;
        }
        return true;
    }(), "One or more regions of interests are invalid");
    DALI_ENFORCE([=]() -> bool {
        auto ref_shape = in.shape[0][ndims_ - 1];
        for (int i = 0; i < in.num_samples(); i++) {
          if (in.shape[i][ndims_ - 1] != ref_shape) {
            return false;
          }
        }
        return true;
    }(), "Number of channels for all images in batch must be equal");

    auto adjusted_rois = AdjustRoi(rois, in.shape);
    auto nchannels_out = N;
    KernelRequirements req;
    ScratchpadEstimator se;
    TensorListShape<ndims_> output_shape(ShapeFromRoi(adjusted_rois, nchannels_out));
    block_setup_.SetupBlocks(output_shape, true);
    se.add<SampleDescriptor>(AllocType::GPU, in.num_samples());
    se.add<BlockDesc>(AllocType::GPU, block_setup_.Blocks().size());
    req.output_shapes = {output_shape};
    req.scratch_sizes = se.sizes;
    return req;
  }


  void Run(KernelContext &context, const OutListGPU<OutputType, spatial_ndims + 1> &out,
           const InListGPU<InputType, spatial_ndims + 1> &in,
           const std::vector<::dali::mat<N, M, detail::TransformationMatrixElementType>> &tmatrices,
           const std::vector<Roi<spatial_ndims>> &rois = {}) {
    auto sample_descs = detail::CreateSampleDescriptors<OutputType, InputType, N, M, spatial_ndims>
            (out, in, tmatrices);

    typename decltype(sample_descs)::value_type *samples_gpu;
    BlockDesc *blocks_gpu;

    std::tie(samples_gpu, blocks_gpu) = context.scratchpad->ToContiguousGPU(
            context.gpu.stream, sample_descs, block_setup_.Blocks());

    dim3 grid_dim = block_setup_.GridDim();
    dim3 block_dim = block_setup_.BlockDim();
    auto stream = context.gpu.stream;
    // @autoformat:off
    LinearTransformationKernel
            <OutputType, InputType, N, M, spatial_ndims>
            <<<grid_dim, block_dim, 0, stream>>>
            (samples_gpu, blocks_gpu);
    // @autoformat:on
  }
};

}  // namespace linear_transformation
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_ALGEBRA_LINEAR_TRANSFORMATION_H_
