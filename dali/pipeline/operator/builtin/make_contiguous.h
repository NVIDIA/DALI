// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_

#include <algorithm>
#include <vector>
#include <utility>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/core/common.h"

// Found by benchmarking coalesced vs non coalesced on diff size images
#define COALESCE_THRESHOLD 8192

namespace dali {

template<typename Backend>
class MakeContiguousBase : public Operator<Backend> {
 public:
  inline explicit MakeContiguousBase(const OpSpec &spec) :
      Operator<Backend>(spec) {}

  virtual inline ~MakeContiguousBase() = default;

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousBase);

 protected:
  USE_OPERATOR_MEMBERS();
  TensorList<CPUBackend> cpu_output_buff;
  bool coalesced = true;
};

class MakeContiguousMixed : public MakeContiguousBase<MixedBackend> {
 public:
  inline explicit MakeContiguousMixed(const OpSpec &spec) :
      MakeContiguousBase<MixedBackend>(spec) {}
  using Operator<MixedBackend>::Run;

  void Run(MixedWorkspace &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<MixedBackend> &ws) override {
    output_desc.resize(1);
    auto &input = ws.template InputRef<CPUBackend>(0);
    output_desc[0].shape = input.shape();
    output_desc[0].type = input.type();
    return true;
  }

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousMixed);
};

class MakeContiguousCPU : public MakeContiguousBase<CPUBackend> {
 public:
  inline explicit MakeContiguousCPU(const OpSpec &spec) :
      MakeContiguousBase<CPUBackend>(spec) {}

  using Operator<CPUBackend>::RunImpl;
  void RunImpl(HostWorkspace &ws) override;


  bool CanInferOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override {
    output_desc.resize(1);
    auto &input = ws.template InputRef<CPUBackend>(0);
    output_desc[0].shape = input.shape();
    output_desc[0].type = input.type();
    /*
     * if input is not continuos and we need to copy allocate whole output in one go
     * otherwise we will share the input and no allocation is needed
     * it is runtime decision and we cannot relay on the executor as CanInferOutputs result must
     * match while it cannot be inferred without the access to the input
     */
    if (!input.IsContiguous()) {
      ws.template OutputRef<CPUBackend>(0).Resize(output_desc[0].shape);
      ws.template OutputRef<CPUBackend>(0).set_type(output_desc[0].type);
    }
    return false;
  }

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousCPU);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_
