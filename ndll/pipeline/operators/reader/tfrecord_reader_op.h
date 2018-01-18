// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_TFRECORD_READER_OP_H_
#define NDLL_PIPELINE_TFRECORD_READER_OP_H_

#if NDLL_USE_PROTOBUF

#include "ndll/pipeline/operators/reader/reader_op.h"
#include "ndll/pipeline/operators/reader/loader/tfrecord.h"
#include "ndll/pipeline/operators/reader/parser/tfrecord_parser.h"

namespace ndll {

class TFRecordReader : public DataReader<CPUBackend> {
 public:
  explicit TFRecordReader(const OpSpec& spec)
  : DataReader<CPUBackend>(spec) {
    loader_.reset(new TFRecordLoader(spec));
    parser_.reset(new TFRecordParser(spec));
  }

  DEFAULT_READER_DESTRUCTOR(TFRecordReader, CPUBackend);

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

#endif  // NDLL_USE_PROTOBUF
#endif  // NDLL_PIPELINE_TFRECORD_READER_OP_H_
