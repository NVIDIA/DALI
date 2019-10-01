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

#ifndef DALI_KERNELS_IMGPROC_POINTWISE_LINEAR_TRANSFORMATION_GPU_H_
#define DALI_KERNELS_IMGPROC_POINTWISE_LINEAR_TRANSFORMATION_GPU_H_

#include <vector>
#include "dali/core/format.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/imgproc/roi.h"

namespace dali {
namespace kernels {
namespace lin_trans {

template <typename OutputType, typename InputType,
        int channels_out, int channels_in, int spatial_ndims>
struct SampleDescriptor {
  const InputType *in;
  OutputType *out;
  ivec<spatial_ndims> in_size, in_strides, out_size, out_strides;

  /// A*x + B
  mat<channels_out, channels_in, float> A;
  vec<channels_out, float> B;

  Roi<spatial_ndims> roi;
};


template <typename OutputType, typename InputType,
        int channels_out, int channels_in, int spatial_ndims>
void __global__ LinearTransformationKernel(
        const lin_trans::SampleDescriptor<OutputType, InputType,
                channels_out, channels_in, spatial_ndims> *samples,
        const BlockDesc<spatial_ndims> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  const Surface2D<const InputType> in = {
          sample.in, sample.in_size.x, sample.in_size.y, channels_in,
          sample.in_strides.x, sample.in_strides.y, 1
  };

  const Surface2D<OutputType> out = {
          sample.out, sample.out_size.x, sample.out_size.y, channels_out,
          sample.out_strides.x, sample.out_strides.y, 1
  };

  auto in_roi = crop(in, sample.roi);

  for (int y = threadIdx.y + block.start.y; y < block.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + block.start.x; x < block.end.x; x += blockDim.x) {
      vec<channels_in> v_in;
      for (int i = 0; i < channels_in; i++) {
        v_in[i] = in_roi(x, y, i);
      }
      vec<channels_out> v_out = sample.A * v_in + sample.B;
      for (int i = 0; i < channels_out; i++) {
        out(x, y, i) = ConvertSat<OutputType>(v_out[i]);
      }
    }
  }
}

}  // namespace lin_trans

template <typename OutputType, typename InputType,
        int channels_out, int channels_in, int spatial_ndims>
class LinearTransformationGpu {
 private:
  static constexpr auto ndims_ = spatial_ndims + 1;
  using Mat = ::dali::mat<channels_out, channels_in, float>;
  using Vec = ::dali::vec<channels_out, float>;
  using BlockDesc = kernels::BlockDesc<spatial_ndims>;
  using SampleDescriptor = lin_trans::SampleDescriptor<OutputType, InputType,
          channels_out, channels_in, spatial_ndims>;

  std::vector<SampleDescriptor> sample_descriptors_;

  // Containers, that keep default values alive, in case they are needed
  std::vector<Mat> default_mats_;
  std::vector<Vec> default_vecs_;

 public:
  BlockSetup<spatial_ndims, spatial_ndims /* Assumed, that the channel dim is the last dim */>
          block_setup_;


  KernelRequirements
  Setup(KernelContext &context, const InListGPU<InputType, ndims_> &in,
        span<const Mat> tmatrices = {}, span<const Vec> tvectors = {},
        span<const Roi<spatial_ndims>> rois = {}) {
    DALI_ENFORCE(rois.empty() || rois.size() == static_cast<size_t>(in.num_samples()),
                 "Provide ROIs either for all or none input tensors");
    for (int i = 0; i < in.size(); i++) {
      DALI_ENFORCE(in[i].shape.shape.back() == channels_in,
                   make_string("Unexpected number of channels at index", i,
                               "in InListGPU. Number of channels in every InListGPU has to match"
                               " the number of channels, that the kernel is instantiated with"));
    }
    for (int i = 0; i < rois.size(); i++) {
      DALI_ENFORCE(all_coords(rois[i].hi >= rois[i].lo),
                   make_string("Found invalid ROI at index", i,
                               "ROI doesn't follow {lo, hi} convention.", rois[i]));
    }

    auto adjusted_rois = AdjustRoi(rois, in.shape);
    KernelRequirements req;
    ScratchpadEstimator se;
    TensorListShape<ndims_> output_shape(ShapeFromRoi(make_cspan(adjusted_rois), channels_out));
    block_setup_.SetupBlocks(output_shape, true);
    se.add<SampleDescriptor>(AllocType::GPU, in.num_samples());
    se.add<BlockDesc>(AllocType::GPU, block_setup_.Blocks().size());
    req.output_shapes = {output_shape};
    req.scratch_sizes = se.sizes;
    return req;
  }


  void Run(KernelContext &context, const OutListGPU<OutputType, ndims_> &out,
           const InListGPU<InputType, ndims_> &in, span<const Mat> tmatrices = {},
           span<const Vec> tvectors = {}, span<const Roi<spatial_ndims>> rois = {}) {
    cout<<"ASDASDASDD 6\n";
    DALI_ENFORCE(tmatrices.size()==0||tvectors.size()==0||tmatrices.size() == tvectors.size());
    if (tmatrices.empty()) {
      gen_default_values(in.num_samples());
      tmatrices = make_cspan(default_mats_);
      tvectors = make_cspan(default_vecs_);
    }
    cout<<"ASDASDASDD 61\n";

    CreateSampleDescriptors(out, in, tmatrices, tvectors, rois);
    cout<<"ASDASDASDD 62\n";

    SampleDescriptor *samples_gpu;
    BlockDesc *blocks_gpu;
    cout<<"ASDASDASDD 63\n";

    std::tie(samples_gpu, blocks_gpu) = context.scratchpad->ToContiguousGPU(
            context.gpu.stream, sample_descriptors_, block_setup_.Blocks());
    cout<<"ASDASDASDD 64\n";

    dim3 grid_dim = block_setup_.GridDim();
    dim3 block_dim = block_setup_.BlockDim();
    auto stream = context.gpu.stream;
    // @autoformat:off
    cout<<"ASDASDASDD 4\n";
    lin_trans::LinearTransformationKernel
            <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, blocks_gpu);
    // @autoformat:on
    cout<<"ASDASDASDD 5\n";
  }


 private:
  void CreateSampleDescriptors(const OutListGPU<OutputType, ndims_> &out,
                               const InListGPU<InputType, ndims_> &in, span<const Mat> tmatrices,
                               span<const Vec> tvectors, span<const Roi<spatial_ndims>> rois) {
    cout<<"ASDASDASDD 71\n";
    assert(tmatrices.size() == tvectors.size());
    cout<<"ASDASDASDD 72\n";
    auto adjusted_rois = AdjustRoi(rois, in.shape);
    sample_descriptors_.resize(in.num_samples());
    cout<<"ASDASDASDD 73\n"<<in.num_samples()<<endl;
    for (int i = 0; i < in.num_samples(); i++) {
      cout<<"ASDASDASDD 731\n";
      cout<<tmatrices.size()<<endl<<tvectors.size()<<endl<<tmatrices[i]<<endl<<tvectors[i]<<endl;
      auto &sample = sample_descriptors_[i];
      sample.in = in[i].data;
      sample.out = out[i].data;

      auto get_size = [](const TensorShape<ndims_> &ts) {
          ivec<spatial_ndims> ret;
          for (int i = ret.size() - 1, j = 0; i >= 0; i--, j++) {
            ret[j] = ts[i];
          }
          return ret;
      };

      cout<<"ASDASDASDD 732\n";
      sample.in_size = get_size(in.tensor_shape(i));
      cout<<"ASDASDASDD 733\n";
      sample.out_size = get_size(out.tensor_shape(i));
      cout<<"ASDASDASDD 734\n";
      sample.in_strides = {channels_in, sample.in_size.x * channels_in};
      cout<<"ASDASDASDD 735\n";
      sample.out_strides = {channels_out, sample.out_size.x * channels_out};
      cout<<"ASDASDASDD 736\n";
      sample.A = tmatrices[i];
      cout<<"ASDASDASDD 737\n";
      sample.B = tvectors[i];
      cout<<"ASDASDASDD 738\n";
      sample.roi = adjusted_rois[i];
      cout<<"ASDASDASDD 739\n";
    }
    cout<<"ASDASDASDD 74\n";
  }

  void gen_default_values(int how_many) {
    default_mats_={how_many, eye<channels_out, channels_in>()};
    Vec zero_vec;
    for (int i = 0; i < channels_out; i++) {
      zero_vec[i] = 0;
    }
    default_vecs_={how_many, zero_vec};
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_POINTWISE_LINEAR_TRANSFORMATION_GPU_H_
