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


#ifndef DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_TEST_H_
#define DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_TEST_H_

#include <vector>
#include <cassert>
#include "dali/kernels/audio/mel_scale/mel_scale.h"
#include "dali/core/common.h"
#include "dali/kernels/kernel_params.h"

namespace dali {
namespace kernels {
namespace audio {
namespace test {

std::vector<std::vector<float>> ReferenceFilterBanks(int nfilter, int nfft, float sample_rate,
                                                     float low_freq, float high_freq);

}  // namespace test
}  // namespace audio
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_TEST_H_
