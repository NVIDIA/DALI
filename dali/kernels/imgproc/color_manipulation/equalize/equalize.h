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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_EQUALIZE_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_EQUALIZE_H_

#include <vector>

#include "dali/core/common.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/hist.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/lookup.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/lut.h"
#include "dali/kernels/kernel.h"
#include "include/dali/core/backend_tags.h"
#include "include/dali/core/tensor_view.h"

namespace dali {
namespace kernels {
namespace equalize {

struct EqualizeKernelGpu {
  static constexpr int hist_range = 256;

  /**
   * @brief Performs per-channel equalization.
   *
   * The input and output samples are flattened: the first extent is all but the channel extent
   * together and the second extent represent channels.
   *
   * Equalization is done in a number of steps: firstly the per-channel histograms are computed.
   * Then, the cumulative (prefix sum) of the histogram are computed. The cumulative(s) are then
   * turned into lookup tables for input samples by proper scaling of the cumulative. Finally,
   * the output is produced by remapping the inputs according to the per-channel lookup tables.
   */
  void Run(KernelContext &ctx, const TensorListView<StorageGPU, uint8_t, 2> &out,
           const TensorListView<StorageGPU, const uint8_t, 2> &in) {
    int batch_size = out.num_samples();
    assert(in.num_samples() == batch_size);
    auto hist_shape = GetHistogramShape(in.shape);
    auto hist_num_elements = hist_shape.num_elements();
    uint64_t *hist_dev_raw = ctx.scratchpad->AllocateGPU<uint64_t>(hist_num_elements);
    uint8_t *lut_dev_raw = ctx.scratchpad->AllocateGPU<uint8_t>(hist_num_elements);
    auto hist_view = make_tensor_list_gpu(hist_dev_raw, hist_shape);
    auto lut_view = make_tensor_list_gpu(lut_dev_raw, hist_shape);
    hist_kernel_.Run(ctx, hist_view, in);
    lut_kernel_.Run(ctx, lut_view, hist_view);
    lookup_kernel_.Run(ctx, out, in, lut_view);
  }

 protected:
  TensorListShape<2> GetHistogramShape(TensorListShape<2> in_shape) {
    int batch_size = in_shape.num_samples();
    TensorListShape<2> tls(batch_size);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_channels = in_shape[sample_idx][1];
      tls.set_tensor_shape(sample_idx, TensorShape<2>{num_channels, hist_range});
    }
    return tls;
  }

  equalize::hist::HistogramKernelGpu hist_kernel_;
  equalize::lut::LutKernelGpu lut_kernel_;
  equalize::lookup::LookupKernelGpu lookup_kernel_;
};

}  // namespace equalize
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_EQUALIZE_H_
