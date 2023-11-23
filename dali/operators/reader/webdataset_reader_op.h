// Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_WEBDATASET_READER_OP_H_
#define DALI_OPERATORS_READER_WEBDATASET_READER_OP_H_

#include <vector>
#include "dali/operators/reader/loader/webdataset_loader.h"
#include "dali/operators/reader/reader_op.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

class DLL_PUBLIC WebdatasetReader
    : public DataReader<CPUBackend, vector<Tensor<CPUBackend>>, vector<Tensor<CPUBackend>>, true> {
 public:
  explicit WebdatasetReader(const OpSpec& spec) : DataReader(spec) {
    loader_ = InitLoader<WebdatasetLoader>(spec);
    this->SetInitialSnapshot();
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace&) override;
  void RunImpl(Workspace &ws) override;
  bool CanInferOutputs() const override {
    return true;
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, vector<Tensor<CPUBackend>>,
                              vector<Tensor<CPUBackend>>, true);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_WEBDATASET_READER_OP_H_
