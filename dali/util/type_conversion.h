// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_UTIL_TYPE_CONVERSION_H_
#define DALI_UTIL_TYPE_CONVERSION_H_

#include "dali/common.h"

namespace dali {

// Type conversions for data on GPU. All conversions
// run in the default stream
template <typename IN, typename OUT>
void Convert(const IN *data, int n, OUT *out);

}  // namespace dali

#endif  // DALI_UTIL_TYPE_CONVERSION_H_
