// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_
#define DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/core/format.h"
#include "dali/util/crop_window.h"
#include "dali/pipeline/data/types.h"


namespace dali {

template <typename Backend>
class MultiPasteOp : public Operator<Backend> {
 public:
  ~MultiPasteOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(MultiPasteOp);

 protected:
  explicit MultiPasteOp(const OpSpec &spec)
      : Operator<Backend>(spec)
      , output_type_arg_(spec.GetArgument<DALIDataType>("dtype"))
      , output_type_(DALI_NO_TYPE)
      , input_type_(DALI_NO_TYPE)
      , output_size_("output_size", spec)
      , in_idx_("in_ids", spec)
      , in_anchors_("in_anchors", spec)
      , in_shapes_("shapes", spec)
      , out_anchors_("out_anchors", spec)
      , no_intersections_(spec.GetArgument<bool>("no_intersections")) {
    if (std::is_same<Backend, GPUBackend>::value) {
      kernel_manager_.Resize(1, 1);
    } else {
      kernel_manager_.Resize(num_threads_, max_batch_size_);
    }
  }

  bool CanInferOutputs() const override {
    return true;
  }

  void AcquireArguments(const OpSpec &spec, const workspace_t<Backend> &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    output_size_.Acquire(spec, ws, curr_batch_size, true);
    in_idx_.Acquire(spec, ws, curr_batch_size, false);
    out_anchors_.Acquire(spec, ws, curr_batch_size, false);
    in_anchors_.Acquire(spec, ws, curr_batch_size, false);
    in_shapes_.Acquire(spec, ws, curr_batch_size, false);
    input_type_ = ws.template InputRef<Backend>(0).type().id();
    output_type_ =
        output_type_arg_ != DALI_NO_TYPE
        ? output_type_arg_
        : input_type_;
  }

  USE_OPERATOR_MEMBERS();
  DALIDataType output_type_arg_, output_type_, input_type_;

  ArgValue<int, 1> output_size_;

  ArgValue<int, 1> in_idx_;
  ArgValue<int, 2> in_anchors_;
  ArgValue<int, 2> in_shapes_;
  ArgValue<int, 2> out_anchors_;

  kernels::KernelManager kernel_manager_;

  bool no_intersections_;
};


class MultiPasteCpu : public MultiPasteOp<CPUBackend> {
 public:
  explicit MultiPasteCpu(const OpSpec &spec) : MultiPasteOp(spec) {}

  using Operator<CPUBackend>::RunImpl;

  ~MultiPasteCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(MultiPasteCpu);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;

  void RunImpl(workspace_t<CPUBackend> &ws) override;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_
