// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_RANDOM_PASTE_RANDOM_PASTE_H_
#define DALI_PIPELINE_OPERATORS_RANDOM_PASTE_RANDOM_PASTE_H_

#include <cstring>
#include <utility>
#include <vector>
#include <random>

#include "dali/common.h"
#include "dali/pipeline/operators/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class RandomPaste : public Operator<Backend> {
 public:
  explicit inline RandomPaste(const OpSpec &spec) :
    Operator<Backend>(spec),
    max_ratio_(spec.GetArgument<float>("max_ratio")),
    rgb_(spec.GetRepeatedArgument<int>("fill_color")) {
      DALI_ENFORCE(rgb_.size() == 3, "Argument `fill_color` expects a list of 3 elements, "
          + to_string(rgb_.size()) + " given.");

      input_ptrs_.Resize({batch_size_});
      output_ptrs_.Resize({batch_size_});
      // 6 values: in_H, in_W, out_H, out_W, paste_y, paste_x
      in_out_dims_paste_yx_.Resize({batch_size_ * 6});
  }

  virtual inline ~RandomPaste() = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  void SetupSampleParams(Workspace<Backend> *ws, const int idx);

  void RunHelper(Workspace<Backend> *ws);

  // Op parameters
  float max_ratio_;
  std::vector<int> rgb_;
  int C_;

  Tensor<CPUBackend> input_ptrs_, output_ptrs_, in_out_dims_paste_yx_;
  Tensor<GPUBackend> input_ptrs_gpu_, output_ptrs_gpu_, in_out_dims_paste_yx_gpu_;
  std::mt19937 rand_gen_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_RANDOM_PASTE_RANDOM_PASTE_H_
