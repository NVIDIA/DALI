// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_CAST_H_
#define DALI_OPERATORS_GENERIC_CAST_H_

#include <vector>

#include "dali/core/convert.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"

namespace dali {

#define CAST_ALLOWED_TYPES                                                                         \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16, float, \
  double)

template <typename Backend>
class Cast : public StatelessOperator<Backend> {
 public:
  explicit inline Cast(const OpSpec &spec)
      : StatelessOperator<Backend>(spec) {
    if (spec.name() == "Cast") {
      dtype_arg_ = spec.GetArgument<DALIDataType>("dtype");
      if (dtype_arg_ == DALI_NO_TYPE) {
        DALI_FAIL(make_string("Unexpected data type argument", dtype_arg_));
      }
    } else {
      assert(spec.name() == "CastLike");
      is_cast_like_ = true;
    }
  }
  inline ~Cast() override = default;

  DISABLE_COPY_MOVE_ASSIGN(Cast);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<Backend>(0);
    DALIDataType out_type = is_cast_like_ ?  ws.GetInputDataType(1) : dtype_arg_;
    output_desc.resize(1);
    output_desc[0].shape = input.shape();
    output_desc[0].type = out_type;
    return true;
  }

 private:
  bool is_cast_like_ = false;
  DALIDataType dtype_arg_ = DALI_NO_TYPE;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_CAST_H_
