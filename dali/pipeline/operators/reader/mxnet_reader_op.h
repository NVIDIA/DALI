// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_OPERATORS_READER_MXNET_READER_OP_H_
#define DALI_PIPELINE_OPERATORS_READER_MXNET_READER_OP_H_

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/recordio_loader.h"
#include "dali/pipeline/operators/reader/parser/recordio_parser.h"

namespace dali {
class MXNetReader : public DataReader<CPUBackend> {
 public:
  explicit MXNetReader(const OpSpec& spec)
  : DataReader<CPUBackend>(spec) {
    loader_.reset(new RecordIOLoader(spec));
    parser_.reset(new RecordIOParser(spec));
  }

  DEFAULT_READER_DESTRUCTOR(MXNetReader, CPUBackend);

  void RunImpl(SampleWorkspace* ws, const int i) override {
    const int idx = ws->data_idx();

    auto* raw_data = prefetched_batch_[idx];

    parser_->Parse(raw_data->data<uint8_t>(), raw_data->size(), ws);

    return;
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend);
};
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_MXNET_READER_OP_H_
