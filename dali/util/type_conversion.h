// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_TYPE_CONVERSION_H_
#define DALI_UTIL_TYPE_CONVERSION_H_

#include "dali/core/common.h"

namespace dali {

// Type conversions for data on GPU. All conversions
// run in the default stream
template <typename IN, typename OUT>
DLL_PUBLIC void Convert(const IN *data, int n, OUT *out);

}  // namespace dali

#endif  // DALI_UTIL_TYPE_CONVERSION_H_
