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

#ifndef DALI_OPERATORS_READER_MXNET_READER_OP_H_
#define DALI_OPERATORS_READER_MXNET_READER_OP_H_

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/recordio_loader.h"
#include "dali/operators/reader/parser/recordio_parser.h"

namespace dali {
class MXNetReader : public DataReader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit MXNetReader(const OpSpec& spec)
  : DataReader<CPUBackend, Tensor<CPUBackend>>(spec) {
    loader_ = InitLoader<RecordIOLoader>(spec);
    parser_.reset(new RecordIOParser(spec));
  }

  void RunImpl(SampleWorkspace &ws) override {
    const auto& tensor = GetSample(ws.data_idx());
    ParseIfNeeded(tensor, &ws);
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, Tensor<CPUBackend>);
};
}  // namespace dali

#endif  // DALI_OPERATORS_READER_MXNET_READER_OP_H_
