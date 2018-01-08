// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_CAFFE_READER_OP_H_
#define NDLL_PIPELINE_CAFFE_READER_OP_H_

#include "ndll/pipeline/reader_op.h"
#include "ndll/pipeline/loader/lmdb.h"
#include "ndll/pipeline/parser/caffe_parser.h"

namespace ndll {

class CaffeReader : public DataReader<CPUBackend> {
 public:
  explicit CaffeReader(const OpSpec& spec)
  : DataReader<CPUBackend>(spec) {
    loader_.reset(new LMDBReader(spec));
    parser_.reset(new CaffeParser(spec));
  }

  DEFAULT_READER_DESTRUCTOR(CaffeReader, CPUBackend);

  void RunPerSampleCPU(SampleWorkspace* ws) override {
    const int idx = ws->data_idx();

    auto* raw_data = prefetched_batch_[idx];

    parser_->Parse(raw_data->data<uint8_t>(), raw_data->size(), ws);

    return;
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend);
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_CAFFE_READER_OP_H_
