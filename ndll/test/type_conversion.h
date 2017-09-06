#ifndef NDLL_TEST_TYPE_CONVERSIONS_H_
#define NDLL_TEST_TYPE_CONVERSIONS_H_

#include "ndll/common.h"

namespace ndll {

// Type conversions for data on GPU. All conversions
// run in the default stream
template <typename IN, typename OUT>
void Convert(const IN *data, int n, OUT *out);

} // namespace ndll

#endif // NDLL_TEST_TYPE_CONVERSIONS_H_
