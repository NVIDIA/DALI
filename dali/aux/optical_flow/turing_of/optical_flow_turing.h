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

#ifndef DALI_AUX_OPTICAL_FLOW_TURING_OF_OPTICAL_FLOW_TURING_H_
#define DALI_AUX_OPTICAL_FLOW_TURING_OF_OPTICAL_FLOW_TURING_H_

#include <host_defines.h>
#include "dali/common.h"

namespace dali {
namespace optical_flow {
namespace kernel {

DLL_PUBLIC void DecodeFlowComponents(const int16_t *input, float *output, size_t num_values);

/**
 * Decode 16-bit float in S10.5 format
 */
DLL_PUBLIC __host__ __device__ float decode_flow_component(int16_t value);

}  // namespace kernel


}  // namespace optical_flow
}  // namespace dali

#endif  // DALI_AUX_OPTICAL_FLOW_TURING_OF_OPTICAL_FLOW_TURING_H_

