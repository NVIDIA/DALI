// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_OPERATORS_COMMON_H_
#define DALI_PIPELINE_OPERATORS_COMMON_H_

#include <vector>

#include "dali/pipeline/operators/op_spec.h"

namespace dali {
template <typename T>
inline void GetSingleOrDoubleArg(const OpSpec &spec, vector<T> *arg, const char *argName,
                          bool doubleArg = true) {
  try {
      *arg = spec.GetRepeatedArgument<T>(argName);
  } catch (std::runtime_error e) {
      try {
        *arg = {spec.GetArgument<T>(argName)};
      } catch (std::runtime_error e) {
          DALI_FAIL("Invalid type of argument \"" + argName + "\"");
      }
  }

  if (doubleArg && arg->size() == 1)
    arg->push_back(arg->back());
}

}  // namespace dali
#endif  // DALI_PIPELINE_OPERATORS_COMMON_H_
