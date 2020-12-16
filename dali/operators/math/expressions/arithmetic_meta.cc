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

#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include "dali/operators/math/expressions/arithmetic_meta.h"

namespace dali {

DALIDataType BinaryTypePromotion(DALIDataType left, DALIDataType right) {
  DALIDataType result = DALIDataType::DALI_NO_TYPE;
  TYPE_SWITCH(left, type2id, Left_t, ARITHMETIC_ALLOWED_TYPES, (
    TYPE_SWITCH(right, type2id, Right_t, ARITHMETIC_ALLOWED_TYPES, (
        using Result_t = binary_result_t<Left_t, Right_t>;
        result = TypeInfo::Create<Result_t>().id();
    ), (DALI_FAIL(make_string("Right operand data type not supported, DALIDataType: ", right));))  // NOLINT
  ), (DALI_FAIL(make_string("Left operand data type not supported, DALIDataType: ", left));));  // NOLINT
  return result;
}

DALIDataType TypePromotion(ArithmeticOp op, span<DALIDataType> types) {
  if (IsIntToFloatResult(op)) {
    bool all_integral = true;
    for (auto t : types)
      all_integral = all_integral && IsIntegral(t);
    if (all_integral)
      return DALIDataType::DALI_FLOAT;
  }
  if (types.size() == 1) {
    return types[0];
  }
  if (IsComparison(op)) {
    return DALIDataType::DALI_BOOL;
  }
  DALIDataType result = BinaryTypePromotion(types[0], types[1]);
  for (int i = 2; i < types.size(); i++) {
    result = BinaryTypePromotion(result, types[i]);
  }
  return result;
}

}  // namespace dali
