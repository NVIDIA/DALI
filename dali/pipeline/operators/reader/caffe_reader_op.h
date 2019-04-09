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

#ifndef DALI_PIPELINE_OPERATORS_READER_CAFFE_READER_OP_H_
#define DALI_PIPELINE_OPERATORS_READER_CAFFE_READER_OP_H_

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/lmdb.h"
#include "dali/pipeline/operators/reader/parser/caffe_parser.h"

namespace dali {

class CaffeReader : public DataReader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit CaffeReader(const OpSpec& spec)
  : DataReader<CPUBackend, Tensor<CPUBackend>>(spec) {
    loader_ = InitLoader<LMDBReader>(spec);
    parser_.reset(new CaffeParser(spec));
  }

  void RunImpl(SampleWorkspace* ws, const int i) override {
    const auto& tensor = GetSample(ws->data_idx());
    if (tensor.ShouldSkipSample())
      return;
    parser_->Parse(tensor, ws);
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, Tensor<CPUBackend>);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_CAFFE_READER_OP_H_
