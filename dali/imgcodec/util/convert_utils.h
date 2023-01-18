// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_UTIL_CONVERT_UTILS_H_
#define DALI_IMGCODEC_UTIL_CONVERT_UTILS_H_

#include "dali/pipeline/data/types.h"

namespace dali {
namespace imgcodec {

inline int PositiveBits(DALIDataType dtype) {
  assert(IsIntegral(dtype));
  int positive_bits = 8 * TypeTable::GetTypeInfo(dtype).size() - IsSigned(dtype);
  return positive_bits;
}

/**
 * @brief Expected maximum value for a given type
 */
inline double MaxValue(DALIDataType dtype) {
  if (!IsIntegral(dtype)) return 1.0;
  return (1_u64 << PositiveBits(dtype)) - 1;
}

/**
 * @brief Whether given precision needs scaling to use the full width of the type
 */
inline bool NeedDynamicRangeScaling(int precision, DALIDataType dtype) {
  return PositiveBits(dtype) != precision;
}

/**
 * @brief Dynamic range multiplier to apply when precision is lower than the
 *        width of the data type
 */
inline double DynamicRangeMultiplier(int precision, DALIDataType dtype) {
  double input_max_value = (1_u64 << precision) - 1;
  return MaxValue(dtype) / input_max_value;
}


}  // namespace imgcodec
}  // namespace dali


#endif  // DALI_IMGCODEC_UTIL_CONVERT_UTILS_H_
