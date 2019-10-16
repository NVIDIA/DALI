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

#ifndef DALI_OPERATORS_READER_TFRECORD_READER_OP_H_
#define DALI_OPERATORS_READER_TFRECORD_READER_OP_H_

#ifdef DALI_BUILD_PROTO3

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/indexed_file_loader.h"
#include "dali/operators/reader/parser/tfrecord_parser.h"

namespace dali {

class TFRecordReader : public DataReader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit TFRecordReader(const OpSpec& spec)
  : DataReader<CPUBackend, Tensor<CPUBackend>>(spec) {
    loader_ = InitLoader<IndexedFileLoader>(spec);
    parser_.reset(new TFRecordParser(spec));
    DALI_ENFORCE(!skip_cached_images_,
      "TFRecordReader doesn't support `skip_cached_images` option");
  }

  void RunImpl(SampleWorkspace &ws) override {
    const auto& tensor = GetSample(ws.data_idx());
    parser_->Parse(tensor, &ws);
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, Tensor<CPUBackend>);
};

}  // namespace dali

#endif  // DALI_BUILD_PROTO3
#endif  // DALI_OPERATORS_READER_TFRECORD_READER_OP_H_
