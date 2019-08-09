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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_GPU_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_GPU_H_

#include <cuda_runtime.h>
#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/data/types.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast.h"

namespace dali {
namespace kernels {
namespace brightness_contrast {


template <size_t ndims>
using Roi_ = Box<ndims, int>;
using Roi = Roi_<2>;

template <class InputType, class OutputType, int ndims>
struct SampleDescriptor {
  const InputType *in;
  OutputType *out;
  ivec<ndims> in_pitch, out_pitch;
  float brightness, contrast;
};


template <size_t ndims>
TensorListShape<ndims> calc_shape(const std::vector<Roi_<ndims>> &rois, int nchannels) {
  std::vector<TensorShape<ndims>> ret;

  for (auto roi : rois) {
    assert(all_coords(roi.hi >= roi.lo) && "Cannot create a tensor shape from an invalid Box");
    TensorShape<ndims> ts;
    auto e = roi.extent();
    auto ridx = ndims;
    ts[--ridx] = e[0] * nchannels;
    for (size_t idx = 1; idx < ndims; idx++) {
      ts[--ridx] = e[idx];
    }
    ret.emplace_back(ts);
  }

  return TensorListShape<ndims>{ret};
}


template <size_t ndims>
std::vector<Roi_<ndims>> AdjustRois(const std::vector<Roi_<ndims>> rois, const TensorListShape<ndims + 1> &shapes) {
  assert(rois.empty() || rois.size() == static_cast<size_t>(shapes.num_samples()));
  std::vector<Roi_<ndims>> ret(shapes.num_samples());

  auto whole_image = [](auto shape) -> Roi_<ndims> {
      constexpr int spatial_dims = ndims;
      ivec<spatial_dims> size;
      for (size_t i = 0; i < spatial_dims; i++)
        size[i] = shape[spatial_dims - 1 - i];
      return {0, size};
  };

  if (rois.empty()) {
    for (int i = 0; i < shapes.num_samples(); i++) {
      ret[i] = whole_image(shapes[i]);
    }
  } else {
    for (size_t i = 0; i < rois.size(); i++) {
      ret[i] = intersection(rois[i], whole_image(shapes[i]));
    }
  }

  return ret;
}


template <class OutputType, class InputType, int ndims>
std::vector<SampleDescriptor<InputType, OutputType, ndims - 1>>CreateSampleDescriptors(const InListGPU<InputType, ndims> &in,                        const OutListGPU<OutputType, ndims> &out,                        const std::vector<float> &brightness, const std::vector<float> &contrast) {
  std::vector<SampleDescriptor<InputType, OutputType, ndims - 1>> ret(in.num_samples());

  for (int i = 0; i < in.num_samples(); i++) {
    auto &sample = ret[i];
    sample.in = in[i].data;
    sample.out = out[i].data;
    sample.in_pitch = {};
    sample.out_pitch = {};

    auto fill_pitch_with_flattening = [](const auto &tv, auto &pitch) {
        auto tl_end = end(tv.shape);
        auto pitch_end = pitch.end();
        std::copy(begin(tv.shape), end(tv.shape), pitch.begin());
        *--pitch_end *= *--tl_end;
    };


    fill_pitch_with_flattening(in[i], sample.in_pitch);
    fill_pitch_with_flattening(out[i], sample.out_pitch);

    sample.brightness = brightness[i];
    sample.contrast = contrast[i];
  }

  return ret;
}



template <class InputType, class OutputType, int ndims>
__global__ void CudaKernel(const SampleDescriptor<InputType, OutputType, ndims> *samples,                           const BlockDesc<ndims> *blocks) {
  auto block = blocks[blockIdx.x];
  auto sample = samples[block.sample_idx];

  auto *__restrict__ in = sample.in;
  auto *__restrict__ out = sample.out;
  printf("%d %d %d %d \n", threadIdx.x + block.start.x, threadIdx.x + block.end.x,
         threadIdx.y + block.start.y, threadIdx.y + block.end.y);

  for (int y = threadIdx.y + block.start.y; y < threadIdx.y + block.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + block.start.x; x < threadIdx.x + block.end.x; x += blockDim.x) {
      out[y * sample.out_pitch.x + x] =
              in[y * sample.in_pitch.x + x] * sample.contrast + sample.brightness;
    }
  }
}


template <typename InputType, typename OutputType, int ndims>
class BrightnessContrastGpu {
 private:
  static constexpr size_t spatial_dims = ndims - 1;
  using BlockDesc = kernels::BlockDesc<spatial_dims>;

  std::vector<SampleDescriptor<InputType, OutputType, ndims>> sample_descriptors_;

 public:
  BlockSetup<spatial_dims, -1> block_setup_;


  KernelRequirements Setup(KernelContext &context, const InListGPU<InputType, ndims> &in,                           const std::vector<float> &brightness, const std::vector<float> &contrast,                           const std::vector<Roi> &rois = {}) {
    DALI_ENFORCE(rois.empty() || rois.size() == static_cast<size_t>(in.num_samples()),
                 "Provide ROIs either for all or none input tensors");
    DALI_ENFORCE([=]() -> bool {
        for (const auto &roi : rois) {
          if (!all_coords(roi.hi >= roi.lo))
            return false;
        }
        return true;
    }(), "One or more regions of interests are invalid");

    auto adjusted_rois = AdjustRois(rois, in.shape);
    KernelRequirements req;
    ScratchpadEstimator se;
    TensorListShape<spatial_dims> output_shape({calc_shape(adjusted_rois, 3)});
    block_setup_.SetupBlocks(output_shape, true);
    se.add<SampleDescriptor<InputType, OutputType, ndims>>(AllocType::GPU, in.num_samples());
    se.add<BlockDesc>(AllocType::GPU, block_setup_.Blocks().size());
    req.output_shapes = {output_shape};
    req.scratch_sizes = se.sizes;
    return req;
  }


  void Run(KernelContext &context, const OutListGPU<OutputType, ndims> &out,           const InListGPU<InputType, ndims> &in, const std::vector<float> &brightness,           const std::vector<float> &contrast, const std::vector<Roi> &rois = {}) {
    auto sample_descs = CreateSampleDescriptors(in, out, brightness, contrast);

    typename decltype(sample_descs)::value_type *samples_gpu;
    BlockDesc *blocks_gpu;

    std::tie(samples_gpu, blocks_gpu) = context.scratchpad->ToContiguousGPU(
            context.gpu.stream, sample_descs, block_setup_.Blocks());

    dim3 grid_dim = block_setup_.GridDim();
    dim3 block_dim = block_setup_.BlockDim();

    CudaKernel<<<grid_dim, block_dim, 0, context.gpu.stream>>>(samples_gpu, blocks_gpu);


  }


};

}  // namespace brightness_contrast
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_H_
