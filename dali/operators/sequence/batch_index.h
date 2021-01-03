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

#ifndef DALI_OPERATORS_SEQUENCE_BATCH_INDEX_H_
#define DALI_OPERATORS_SEQUENCE_BATCH_INDEX_H_

#include <random>
#include <vector>
#include "dali/pipeline/operator/operator.h"

namespace dali {

class BatchIndex : public Operator<CPUBackend> {
 public:
  explicit BatchIndex(const OpSpec &spec)
  : Operator<CPUBackend>(spec) {}

  int GetBatchSize(const HostWorkspace &) {
    return spec_.GetArgument<int>("batch_size");
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    output_desc.resize(1);
    output_desc[0].shape = TensorListShape<0>(GetBatchSize(ws));
    output_desc[0].type = TypeTable::GetTypeInfo(DALI_INT32);
    return true;
  }
  void RunImpl(HostWorkspace &ws) override;
  bool CanInferOutputs() const override { return true; }
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEQUENCE_BATCH_INDEX_H_
