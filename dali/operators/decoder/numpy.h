// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_NUMPY_H_
#define DALI_OPERATORS_DECODER_NUMPY_H_

#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/util/numpy.h"

namespace dali {

class NumpyDecoder : public StatelessOperator<CPUBackend> {
 public:
  explicit inline NumpyDecoder(const OpSpec &spec)
      : StatelessOperator<CPUBackend>(spec), dtype_{std::nullopt}, ndim_{std::nullopt} {
    if (spec.HasArgument("dtype")) {
      dtype_override_ = spec.GetArgument<DALIDataType>("dtype");
    } else {
      // If dtype is not specified, it will be inferred from the first sample
      dtype_override_ = std::nullopt;
    }
  }

  inline ~NumpyDecoder() override = default;
  DISABLE_COPY_MOVE_ASSIGN(NumpyDecoder);

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;

  std::optional<DALIDataType> dtype_override_;
  std::optional<DALIDataType> dtype_;
  std::optional<std::size_t> ndim_;
  std::vector<numpy::HeaderData> headers_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NUMPY_H_
