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

#ifndef DALI_PIPELINE_OPERATORS_TRANSPOSE_TRANSPOSE_H_
#define DALI_PIPELINE_OPERATORS_TRANSPOSE_TRANSPOSE_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/transpose/cutt/cutt.h"

namespace dali {

template <typename Backend>
class Transpose : public Operator<Backend> {
 public:
  explicit inline Transpose(const OpSpec &spec)
      : Operator<Backend>(spec)
      , perm_(spec.GetRepeatedArgument<int>("perm"))
      , transpose_layout_(spec.GetArgument<bool>("transpose_layout"))
      , output_layout_arg_(spec.GetArgument<TensorLayout>("output_layout")) {
    DALI_ENFORCE(
      [](std::vector<int> perm) {
        std::sort(perm.begin(), perm.end());
        for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
          if (perm[i] != i) {
            return false;
          }
        }
        return true;
      }(perm_), "Invalid permutation: sorted `perm` is not equal to [0, ..., n-1].");
    }

  ~Transpose() override;

  DISABLE_COPY_MOVE_ASSIGN(Transpose);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    const auto &input = ws.template Input<Backend>(0);
    auto in_layout = input.GetLayout();
    output_layout_ = in_layout;
    if (!output_layout_arg_.empty()) {
      output_layout_ = output_layout_arg_;
    } else if (transpose_layout_) {
      for (int d = 0; d < in_layout.ndim(); d++) {
        output_layout_[d] = in_layout[perm_[d]];
      }
    }
    return false;
  }

  void RunImpl(Workspace<Backend> &ws) override;

 private:
  std::vector<int> perm_;
  bool transpose_layout_;
  TensorLayout output_layout_arg_;
  TensorLayout output_layout_;

  cuttHandle cutt_handle_ = 0;
  // used by dense TL cuttHandle
  kernels::TensorShape<> previous_iter_shape_;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_TRANSPOSE_TRANSPOSE_H_
