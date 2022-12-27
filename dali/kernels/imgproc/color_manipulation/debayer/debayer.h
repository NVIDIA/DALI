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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_DEBAYER_DEBAYER_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_DEBAYER_DEBAYER_H_

#include <string>

#include "dali/core/common.h"
#include "dali/core/span.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace debayer {

enum class DALIBayerPattern {
  DALI_BAYER_BG = 0,
  DALI_BAYER_GB = 1,
  DALI_BAYER_GR = 2,
  DALI_BAYER_RG = 3
};

enum class DALIDebayerAlgorithm {
  DALI_DEBAYER_BILINEAR_NPP = 0
};

inline std::string to_string(DALIBayerPattern bayer_pattern) {
  switch (bayer_pattern) {
    case DALIBayerPattern::DALI_BAYER_BG:
      return "BG(GR)";
    case DALIBayerPattern::DALI_BAYER_GB:
      return "GB(RG)";
    case DALIBayerPattern::DALI_BAYER_GR:
      return "GR(BG)";
    case DALIBayerPattern::DALI_BAYER_RG:
      return "RG(GB)";
    default:
      return "<unknown>";
  }
}

inline std::string to_string(DALIDebayerAlgorithm alg) {
  switch (alg) {
    case DALIDebayerAlgorithm::DALI_DEBAYER_BILINEAR_NPP:
      return "bilinear_npp";
    default:
      return "<unknown>";
  }
}

inline DALIDebayerAlgorithm parse_algorithm_name(std::string alg) {
  std::transform(alg.begin(), alg.end(), alg.begin(), [](auto c) { return std::tolower(c); });
  if (alg == "bilinear_npp") {
    return DALIDebayerAlgorithm::DALI_DEBAYER_BILINEAR_NPP;
  }
  throw std::runtime_error(
      make_string("Unsupported debayer algorithm was specified: `", alg, "`."));
}

}  // namespace debayer

template <typename InOutT>
struct DebayerKernelGpu {
  static constexpr int in_ndim = 2;
  static constexpr int out_ndim = 3;
  virtual void Run(KernelContext &context, TensorListView<StorageGPU, InOutT, out_ndim> output,
                   TensorListView<StorageGPU, const InOutT, in_ndim> input,
                   span<const debayer::DALIBayerPattern> patterns) = 0;

  virtual ~DebayerKernelGpu() = default;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_DEBAYER_DEBAYER_H_
