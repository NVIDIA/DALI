// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_DEBAYER_DEBAYER_NPP_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_DEBAYER_DEBAYER_NPP_H_

#include <tuple>

#include "dali/core/span.h"
#include "dali/kernels/imgproc/color_manipulation/debayer/debayer.h"
#include "dali/kernels/imgproc/color_manipulation/debayer/npp_debayer_call.h"
#include "dali/npp/npp.h"
#include "include/dali/core/backend_tags.h"
#include "include/dali/core/tensor_view.h"

namespace dali {
namespace kernels {
namespace debayer {

/**
 * @brief Transforms DALI's OpenCv-style bayer pattern specification to NPP's enum.
 * All the supported bayer patterns are 2x2 tiles.
 * For example the BGGR corresponds to
 * [BG,
 *  GR]
 * tile. Imagine covering a single-channel image with a given tile. Now, the letter at
 * any given position specifies which color channel's intensity is described by
 * a corresponding value. The OpenCV convention used by the `DALIBayerPattern` names the
 * pattern by looking at the image's ((1, 1), (3, 3)) rectangle, while NPP looks at
 * positions ((0, 0), (2, 2)). Thus, the quite surprising mapping below, which
 * seemingly permutes the patterns.
 */
inline NppiBayerGridPosition to_npp(DALIBayerPattern bayer_pattern) {
  switch (bayer_pattern) {
    case DALIBayerPattern::DALI_BAYER_BG:  // bg(gr)
      return NPPI_BAYER_RGGB;
    case DALIBayerPattern::DALI_BAYER_GB:  // gb(rg)
      return NPPI_BAYER_GRBG;
    case DALIBayerPattern::DALI_BAYER_GR:  // gr(bg)
      return NPPI_BAYER_GBRG;
    case DALIBayerPattern::DALI_BAYER_RG:  // rg(gb)
      return NPPI_BAYER_BGGR;
    default:
      throw std::runtime_error(
          make_string("Unsupported bayer pattern: ", to_string(bayer_pattern), "."));
  }
}

template <typename InOutT>
struct NppDebayerKernel : public DebayerKernelGpu<InOutT> {
  using SupportedInputTypes = std::tuple<uint8_t, uint16_t>;
  static_assert(contains_v<InOutT, SupportedInputTypes>, "Unsupported input type.");
  using Base = DebayerKernelGpu<InOutT>;
  using Base::in_ndim;
  using Base::out_ndim;

  explicit NppDebayerKernel(int device_id) : npp_ctx_{CreateNppContext(device_id)} {}

  void Run(KernelContext &context, TensorListView<StorageGPU, InOutT, out_ndim> output,
           TensorListView<StorageGPU, const InOutT, in_ndim> input,
           span<const DALIBayerPattern> patterns) override {
    constexpr int num_out_chanels = 3;
    int batch_size = input.num_samples();
    assert(output.num_samples() == batch_size);
    assert(patterns.size() == batch_size);
    UpdateNppContextStream(npp_ctx_, context.gpu.stream);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      const auto &in_view = input[sample_idx];
      const auto &out_view = output[sample_idx];
      const auto &sample_shape = in_view.shape;
      int width = sample_shape[1];
      int height = sample_shape[0];
      CUDA_CALL(npp_debayer_call(in_view.data, width * sizeof(InOutT), {width, height},
                                 {0, 0, width, height}, out_view.data,
                                 width * num_out_chanels * sizeof(InOutT),
                                 to_npp(patterns[sample_idx]), npp_ctx_));
    }
  }


 protected:
  NppStreamContext npp_ctx_{cudaStream_t(-1), 0};
};

}  // namespace debayer
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_DEBAYER_DEBAYER_NPP_H_
