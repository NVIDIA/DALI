// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_AUDIO_PREEMPHASIS_FILTER_OP_H_
#define DALI_OPERATORS_AUDIO_PREEMPHASIS_FILTER_OP_H_

#include <string>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/operator.h"

#define PREEMPH_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)

namespace dali {
namespace detail {

const std::string kCoeff = "preemph_coeff";  // NOLINT
const std::string kBorder = "border";  // NOLINT
const int kNumOutputs = 1;

}  // namespace detail

template<typename Backend>
class PreemphasisFilter : public Operator<Backend> {
 public:
  enum class BorderType : uint8_t {
    Zero = 0,
    Clamp,
    Reflect,
  };

  explicit PreemphasisFilter(const OpSpec &spec)
      : Operator<Backend>(spec),
        output_type_(spec.GetArgument<DALIDataType>(arg_names::kDtype)) {
    auto border_str = spec.GetArgument<std::string>(detail::kBorder);
    if (border_str == "zero") {
      border_type_ = BorderType::Zero;
    } else if (border_str == "reflect") {
      border_type_ = BorderType::Reflect;
    } else if (border_str == "clamp") {
      border_type_ = BorderType::Clamp;
    } else {
      DALI_FAIL(make_string("``border`` mode \"", border_str, "\" is not supported."));
    }
  }

  ~PreemphasisFilter() override = default;
  DISABLE_COPY_MOVE_ASSIGN(PreemphasisFilter);

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    AcquireArguments(ws);
    output_desc.resize(detail::kNumOutputs);
    auto shape = input.shape();
    output_desc[0].shape = shape;
    output_desc[0].type = TypeTable::GetTypeInfo(output_type_);
    return true;
  }

 protected:
  void AcquireArguments(const workspace_t<Backend> &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    this->GetPerSampleArgument(preemph_coeff_, detail::kCoeff, ws, curr_batch_size);
  }

  USE_OPERATOR_MEMBERS();
  std::vector<float> preemph_coeff_;
  const DALIDataType output_type_;
  BorderType border_type_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_PREEMPHASIS_FILTER_OP_H_
