// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SLICE_SLICE_HWC2CHW_NORMALIZE_GPU_H_
#define DALI_KERNELS_SLICE_SLICE_HWC2CHW_NORMALIZE_GPU_H_

#include <sys/types.h>
#include <tuple>
#include "dali/core/backend_tags.h"
#include "dali/core/common.h"
#include "dali/core/geom/vec.h"
#include "dali/core/small_vector.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace slice_flip_normalize {

/**
 * @brief Specialized version of SliceFlipNormalize that reads a HWC u8 image (with 3 channels)
 * and outputs a HWC or CHW normalized float image, that can be cropped in Y, X coordinates,
 * mirrored in X coordinate, and the channels can be padded.
 *
 * Cropping the input in y, x (HW) coordinates, flipping in x (W) coordinate
 * and padding the channels (from 3 to 4 in HWC->HWC variant) are optional, optimized implementation
 * will be selected when those features are not used across the batch.
 *
 * Overview of the kernel:
 * The image is processed in flattened coordinates. The Y, X stays the same between the interleaved
 * input layout and planar output layout. Assuming 3-channel input, we can look at the input as
 * a sequential stream of values, where we distribute them (sequentially) into 3 output planes.
 * Use a thread block size, that is divisible both by channel number (for the output loop),
 * and 4 (for input loop).
 * The processing steps:
 * 1. [Input loop] Load the linear chunk of input into shared memory, utilizing 4-byte aligned loads
 *    and cast it to float.
 *   a. Unaligned prologue loop - reads the first chunk till we get to address that is aligned with
 *      32 * 4.
 *   b. Main loop - do as many aligned 4byte reads as possible
 *   c. Epilogue loop - read the remaining values that were not possible to read as one 4byte read.
 * 2. Synchronize
 * 3. [Output loop] Each thread corresponds to a (Y, X) sequential offset into a plane, computes
 *    the values for all the channels and writes them.
 *   a. Optionally, mirroring is performed by inverting the X-coordinate in the output offset.
 *   b. Padding the output channels is performed by filling additional planes with fill values.
 *
 *
 * @tparam Out output type - fp16 and fp32 allowed.
 *
 * @details see SliceFlipNormalizeGPU::Args
 */
template <typename Out>
class DLL_PUBLIC SliceHwc2HwcChwNormalizeGPU {
 public:
  static constexpr int spatial_dim = 2;
  static constexpr int channel_dim = 2;
  static constexpr int ndim = 3;
  using In = uint8_t;
  struct SampleArgs {
    Roi<spatial_dim> roi;               // input region-of-interest, only proper crop, no padding.
    SmallVector<float, 4> mean;         // mean (as many values as num. of channels)
    SmallVector<float, 4> inv_stddev;   // reciprocal of stddev (as many as num. of channels)
    SmallVector<float, 4> fill_values;  // per-channel fill value (if more values than number
                                        // channels in the input, the channel dimension is padded
                                        // on all pixels)
    bool flip_x;                        // wether to mirror in x-axis, EnableFlipX must be true.
  };

  SliceHwc2HwcChwNormalizeGPU() = default;

  ~SliceHwc2HwcChwNormalizeGPU() = default;


  DLL_PUBLIC KernelRequirements Setup(KernelContext &ctx, const TensorListShape<ndim> &input_shape,
                                      span<const SampleArgs> args, TensorLayout output_layout);

  void Run(KernelContext &ctx, const TensorListView<StorageGPU, Out, ndim> &out,
           const TensorListView<StorageGPU, const In, ndim> &in, span<const SampleArgs> args);

 private:
  /**
   * @brief Extract the mean, inv_stddev and fill values from kernel's SampleArgs and transfer
   *        them to GPU.
   * @param ctx
   * @param args The kernel arguments from Setup/Run
   * @return std::tuple<float *, float *, Out *> pointer to gpu-allocated:
   *         (mean, inv_stdev, fill_values), first two have nchannels_ values per sample,
   *         the fill_values has out_nchannels_ values per sample.
   */
  std::tuple<float *, float *, Out *> SetupParams(KernelContext &ctx, span<const SampleArgs> args);

  /**
   * @brief Return the new TensorView and Roi for the input sample, so that the cropped rows
   *        (in y axis) are completely skipped and the Roi in x-axis starts from 0.
   *
   * @param in_sample Original view to the input
   * @param roi Description of the output Roi (the crop that we need to extract)
   * @return Adjusted view and roi.
   */
  std::tuple<TensorView<StorageGPU, const In, ndim>, Roi<spatial_dim>> RealignSample(
      TensorView<StorageGPU, const In, ndim> in_sample, Roi<spatial_dim> roi);

  /**
   * @brief Detect the input and output number of channels and validate if it is uniform.
   */
  void SetupNumChannels(const TensorListShape<ndim> &input_shape, span<const SampleArgs> args);

  // This is a multiple of LCM(3, 4) = LCM(Number of Channels, 4) where 4 is from 4-byte reads
  static constexpr int kBlockSizeMul = 24;
  static constexpr int kBlockWidth = 128;
  static constexpr int kThreadBlockSize = 128;
  // TODO(klecki): Generalize for other static channel values
  static constexpr int kStaticChannels = 3;

  // The shape of the produced outputs in CHW (actual layout)
  TensorListShape<ndim> out_shape_;
  // This is a collapsed shape that includes the size of HW output plane (after crop)
  // and the input number of channels (nchannels_)
  TensorListShape<1> collapsed_tiling_shape_;
  // number of channels in the input image
  int nchannels_ = -1;
  // number of channels in the output image (in case of padding)
  int out_nchannels_ = -1;
  // HWC -> {CHW, HWC} permutation
  std::array<int, ndim> perm_;
  TensorLayout output_layout_;
};

}  // namespace slice_flip_normalize

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_SLICE_SLICE_HWC2CHW_NORMALIZE_GPU_H_
