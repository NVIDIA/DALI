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

#ifndef DALI_OPERATORS_GENERIC_REDUCE_SUM_H__
#define DALI_OPERATORS_GENERIC_REDUCE_SUM_H__

#include "dali/core/static_map.h"
#include "dali/operators/generic/reduce/reduce.h"

/// Definition of mapping between input types and possible output types
/// for Sum operator. Is uesed as parameter to TYPE_MAP macro.
#define SUM_TYPES_MAP ( \
    ((uint8_t), (uint8_t, uint64_t, float)), \
    ((int8_t), (int8_t, int64_t, float)), \
    ((uint16_t), (uint16_t, uint64_t, float)), \
    ((int16_t), (int16_t, int64_t, float)), \
    ((uint32_t), (uint32_t, uint64_t, float)), \
    ((int32_t), (int32_t, int64_t, float)), \
    ((uint64_t), (uint64_t)), \
    ((int64_t), (int64_t)), \
    ((float), (float)))

namespace dali {

template <template <typename T, typename R> class ReductionType, typename Backend>
class SumOp : public Reduce<ReductionType, Backend, SumOp> {
 public:
  explicit inline SumOp(const OpSpec &spec) :
    Reduce<ReductionType, Backend, SumOp>(spec) {}

  DALIDataType OutputTypeImpl(DALIDataType input_type) const {
    if (this->output_type_ != DALI_NO_TYPE) {
      return this->output_type_;
    }

    switch (input_type) {
      case DALI_UINT8:
      case DALI_UINT16:
      case DALI_UINT32:
        return DALI_UINT64;
        break;

      case DALI_INT8:
      case DALI_INT16:
      case DALI_INT32:
        return DALI_INT64;
        break;

      default:
        return input_type;
    }
  }

  void RunImplImpl(workspace_t<Backend> &ws) {
    auto& in = ws.template InputRef<Backend>(0);
    DALIDataType input_type = in.type().id();
    DALIDataType output_type = this->OutputType(input_type);

    TYPE_MAP(
      input_type,
      output_type,
      type2id,
      InputType,
      OutputType,
      SUM_TYPES_MAP,
      (this->template RunTyped<OutputType, InputType>(ws);),
      (DALI_FAIL(make_string("Unsupported input type: ", input_type));),
      (DALI_FAIL(make_string("Unsupported types: ", input_type, ", ", output_type));))
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_REDUCE_SUM_H_
