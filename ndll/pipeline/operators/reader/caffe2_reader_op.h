// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_CAFFE2_READER_OP_H_
#define NDLL_PIPELINE_OPERATORS_READER_CAFFE2_READER_OP_H_

#include "ndll/pipeline/operators/reader/reader_op.h"
#include "ndll/pipeline/operators/reader/loader/lmdb.h"
#include "ndll/pipeline/operators/reader/parser/caffe2_parser.h"

namespace ndll {

class Caffe2Reader : public DataReader<CPUBackend> {
 public:
  explicit Caffe2Reader(const OpSpec& spec)
  : DataReader<CPUBackend>(spec) {
    loader_.reset(new LMDBReader(spec));
    parser_.reset(new Caffe2Parser(spec));
  }

  DEFAULT_READER_DESTRUCTOR(Caffe2Reader, CPUBackend);

  void RunPerSampleCPU(SampleWorkspace* ws, const int i) override {
    const int idx = ws->data_idx();

    auto* raw_data = prefetched_batch_[idx];

    parser_->Parse(raw_data->data<uint8_t>(), raw_data->size(), ws);

    return;
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend);
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_READER_CAFFE2_READER_OP_H_

