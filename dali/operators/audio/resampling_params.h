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

#ifndef DALI_OPERATORS_AUDIO_RESAMPLING_PARAMS_H_
#define DALI_OPERATORS_AUDIO_RESAMPLING_PARAMS_H_

#include <cmath>

namespace dali {
namespace audio {

struct ResamplingParams {
  int lobes = 16;
  int lookup_size = 2048;

  static ResamplingParams FromQuality(double q) {
    int lobes = std::round(0.007 * q * q - 0.09 * q + 3);
    return { lobes, lobes * 64 + 1 };
  }
};

}  // namespace audio
}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_RESAMPLING_PARAMS_H_
