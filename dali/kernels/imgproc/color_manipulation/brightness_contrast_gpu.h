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

#include <vector>
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/kernels/common/block_setup.h"

namespace dali {
namespace kernels {
namespace brightness_contrast {


template <size_t ndims>
using Roi = Box<ndims, int>;


template <class OutputType, class InputType, int ndims>
struct SampleDescriptor {
  const InputType *in;
  OutputType *out;
  ivec<ndims> in_pitch, out_pitch;
  float brightness, contrast;
};


/**
 * Coverts Roi to flattened TensorListShape
 *
 * Flattened TensorListShape, is a TensorListShape, which doesn't have `channels`
 * as separate dimension. Instead, Width dimension is larger by `nchannels` number of times.
 * This is because, the analysis is channel-agnostic.
 *
 * Function assumes, that memory layout is *HWC, and the Roi
 * is represented as [[x_lo, y_lo], [x_hi, y_hi]].
 * Therefore, while copying, order of values needs to be reversed.
 */
template <size_t ndims>
TensorListShape<ndims> RoiToShape(const std::vector<Roi<ndims>> &rois, int nchannels) {
  TensorListShape<ndims> ret(rois.size());

  for (const auto &roi : rois) {
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


/**
 * 1. If `rois` is empty, that means whole image is analysed: Roi will the the size of an image
 * 2. If `rois` is not empty, it is assumed, that Roi is provided for every image in batch.
 *    In this case, final Roi is an intersection of provided Roi and the image.
 *    (This is a sanity-check for Rois, that can be larger than image)
 */
template <size_t spatial_dims>
std::vector<Roi<spatial_dims>> AdjustRois(const std::vector<Roi<spatial_dims>> rois,
                                          const TensorListShape<spatial_dims + 1> &shapes) {
  assert(rois.empty() || rois.size() == static_cast<size_t>(shapes.num_samples()));
  std::vector<Roi<spatial_dims>> ret(shapes.num_samples());

  auto whole_image = [](const auto &shape) -> Roi<spatial_dims> {
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


/**
 * Note: Since the brightness-contrast calculation is channel-agnostic (it is performed in the same
 * way for every channel), SampleDescriptor assumes, that sample is channel-agnostic. Therefore it
 * needs to be flattened
 * @see RoiToShape
 */
template <class OutputType, class InputType, int ndims>
std::vector<SampleDescriptor<OutputType, InputType, ndims - 1>>
CreateSampleDescriptors(const OutListGPU<OutputType, ndims> &out,
                        const InListGPU<InputType, ndims> &in,

                        const std::vector<float> &brightness, const std::vector<float> &contrast) {
  std::vector<SampleDescriptor<OutputType, InputType, ndims - 1>> ret(in.num_samples());

  for (int i = 0; i < in.num_samples(); i++) {
    auto &sample = ret[i];
    sample.in = in[i].data;
    sample.out = out[i].data;
    sample.in_pitch = {};
    sample.out_pitch = {};

    auto fill_pitch_with_flattening = [](const auto &tv, auto &pitch) {
        for (size_t i = 0; i < pitch.size(); i++) {
          pitch[i] = tv.shape[i];
        }
        pitch[pitch.size() - 1] *= tv.shape[pitch.size()];
    };

    fill_pitch_with_flattening(in[i], sample.in_pitch);
    fill_pitch_with_flattening(out[i], sample.out_pitch);

    sample.brightness = brightness[i];
    sample.contrast = contrast[i];
  }

  return ret;
}


template <class OutputType, class InputType, int ndims>
__global__ void
BrightnessContrastKernel(const SampleDescriptor<OutputType, InputType, ndims> *samples,
                         const BlockDesc<ndims> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  const auto *__restrict__ in = sample.in;
  auto *__restrict__ out = sample.out;

  for (int y = threadIdx.y + block.start.y; y < threadIdx.y + block.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + block.start.x; x < threadIdx.x + block.end.x; x += blockDim.x) {
      out[y * sample.out_pitch.x + x] = ConvertSat<OutputType>(
              in[y * sample.in_pitch.x + x] * sample.contrast + sample.brightness);
    }
  }
}


template <typename OutputType, typename InputType, int ndims>
class BrightnessContrastGpu {
 private:
  static constexpr size_t spatial_dims = ndims - 1;
  using BlockDesc = kernels::BlockDesc<spatial_dims>;

  std::vector<SampleDescriptor<OutputType, InputType, ndims>> sample_descriptors_;

 public:
  BlockSetup<spatial_dims, -1 /* No channel dimension, only spatial */> block_setup_;


  KernelRequirements Setup(KernelContext &context, const InListGPU<InputType, ndims> &in,
                           const std::vector<float> &brightness, const std::vector<float> &contrast,
                           const std::vector<Roi<spatial_dims>> &rois = {}) {
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
        auto ref_shape = in.shape[0][ndims - 1];
        for (int i = 0; i < in.num_samples(); i++) {
          if (in.shape[i][ndims - 1] != ref_shape) {
            return false;
          }
        }
        return true;
    }(), "Number of channels for every image in batch must be equal");

    auto adjusted_rois = AdjustRois(rois, in.shape);
    auto shape = in.shape[0][ndims - 1];
    KernelRequirements req;
    ScratchpadEstimator se;
    TensorListShape<spatial_dims> output_shape({RoiToShape(adjusted_rois, shape)});
    block_setup_.SetupBlocks(output_shape, true);
    se.add<SampleDescriptor<InputType, OutputType, ndims>>(AllocType::GPU, in.num_samples());
    se.add<BlockDesc>(AllocType::GPU, block_setup_.Blocks().size());
    req.output_shapes = {output_shape};
    req.scratch_sizes = se.sizes;
    return req;
  }


  void Run(KernelContext &context, const OutListGPU<OutputType, ndims> &out,
           const InListGPU<InputType, ndims> &in, const std::vector<float> &brightness,
           const std::vector<float> &contrast, const std::vector<Roi<spatial_dims>> &rois = {}) {
    auto sample_descs = CreateSampleDescriptors(out, in, brightness, contrast);

    typename decltype(sample_descs)::value_type *samples_gpu;
    BlockDesc *blocks_gpu;

    std::tie(samples_gpu, blocks_gpu) = context.scratchpad->ToContiguousGPU(
            context.gpu.stream, sample_descs, block_setup_.Blocks());

    dim3 grid_dim = block_setup_.GridDim();
    dim3 block_dim = block_setup_.BlockDim();
    auto stream = context.gpu.stream;

    BrightnessContrastKernel<<<grid_dim, block_dim, 0, stream>>>(samples_gpu, blocks_gpu);
  }
};

}  // namespace brightness_contrast
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_GPU_H_
