// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_MOVING_MEAN_SQUARE_ARGS_H_
#define DALI_KERNELS_SIGNAL_MOVING_MEAN_SQUARE_ARGS_H_

namespace dali {
namespace kernels {
namespace signal {

struct MovingMeanSquareArgs {
  /**
   * Size of the sliding window
   */
  int window_size = 2048;

  /**
   * The number of samples after which the running
   * sum of squares is recalculated.
   * If `-1`, no recalculation will happen
   */
  int reset_interval = -1;
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_MOVING_MEAN_SQUARE_ARGS_H_
