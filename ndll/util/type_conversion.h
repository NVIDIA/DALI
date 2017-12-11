// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_UTIL_TYPE_CONVERSION_H_
#define NDLL_UTIL_TYPE_CONVERSION_H_

#include "ndll/common.h"

namespace ndll {

// Type conversions for data on GPU. All conversions
// run in the default stream
template <typename IN, typename OUT>
void Convert(const IN *data, int n, OUT *out);

}  // namespace ndll

#endif  // NDLL_UTIL_TYPE_CONVERSION_H_
