// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_DCT_DCT_ARGS_H_
#define DALI_KERNELS_SIGNAL_DCT_DCT_ARGS_H_

#include <tuple>

namespace dali {
namespace kernels {
namespace signal {
namespace dct {

/**
 * @brief DCT kernel arguments
 */
struct DctArgs {
  /// @brief DCT type. Supported types are 1, 2, 3, 4
  /// @remarks DCT type I requires the input data length to be > 1.
  int dct_type = 2;

  /// @brief If true, the output DCT matrix will be normalized to be orthogonal
  /// @remarks Normalization is not supported for DCT type I
  bool normalize = false;

  /// @brief Number of coefficients we are interested in calculating.
  ///        By default, ndct = in_shape[axis]
  int ndct = -1;

  inline bool operator==(const DctArgs& oth) const {
    return dct_type == oth.dct_type &&
           ndct == oth.ndct &&
           normalize == oth.normalize;
  }

  inline bool operator!=(const DctArgs& oth) const {
    return !operator==(oth);
  }

  // needed to use the struct as a key in std::map
  inline bool operator<(const DctArgs& rhs) const {
    return std::make_tuple(dct_type, normalize, ndct)
           < std::make_tuple(rhs.dct_type, rhs.normalize, rhs.ndct);
  }
};

}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DCT_DCT_ARGS_H_
