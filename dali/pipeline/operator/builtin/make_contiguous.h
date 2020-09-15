// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
      Operator<Backend>(spec) {
    std::vector<int> hints;
    GetSingleOrRepeatedArg(spec, hints, "bytes_per_sample_hint", spec.NumOutput());
    if (!hints.empty())
      bytes_per_sample_hint = hints[0];
  }

  virtual inline ~MakeContiguousBase() = default;

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    output_desc.resize(1);
    auto &input = ws.template InputRef<CPUBackend>(0);
    output_desc[0].shape = input.shape();
    output_desc[0].type = input.type();
    return true;
  }

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousBase);

 protected:
  USE_OPERATOR_MEMBERS();
  TensorList<CPUBackend> cpu_output_buff;
  bool coalesced = true;
  int bytes_per_sample_hint = 0;
};

class MakeContiguousMixed : public MakeContiguousBase<MixedBackend> {
 public:
  inline explicit MakeContiguousMixed(const OpSpec &spec) :
      MakeContiguousBase<MixedBackend>(spec) {}
  using Operator<MixedBackend>::Run;

  void Run(MixedWorkspace &ws) override;

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousMixed);
};

class MakeContiguousCPU : public MakeContiguousBase<CPUBackend> {
 public:
  inline explicit MakeContiguousCPU(const OpSpec &spec) :
      MakeContiguousBase<CPUBackend>(spec) {}

  using Operator<CPUBackend>::RunImpl;
  void RunImpl(HostWorkspace &ws) override;
  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousCPU);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_
