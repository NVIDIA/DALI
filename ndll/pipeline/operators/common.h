// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_COMMON_H_
#define NDLL_PIPELINE_OPERATORS_COMMON_H_

#include <vector>

#include "ndll/pipeline/operators/op_spec.h"

namespace ndll {
template <typename T>
inline void GetSingleOrDoubleArg(const OpSpec &spec, vector<T> *arg, const char *argName,
                          bool doubleArg = true) {
  try {
      *arg = spec.GetRepeatedArgument<T>(argName);
  } catch (std::runtime_error e) {
      try {
        *arg = {spec.GetArgument<T>(argName)};
      } catch (std::runtime_error e) {
          NDLL_FAIL("Invalid type of argument \"" + argName + "\"");
      }
  }

  if (doubleArg && arg->size() == 1)
    arg->push_back(arg->back());
}

}  // namespace ndll
#endif  // NDLL_PIPELINE_OPERATORS_COMMON_H_
