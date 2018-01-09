// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_PARSER_CAFFE_PARSER_H_
#define NDLL_PIPELINE_PARSER_CAFFE_PARSER_H_

#include "ndll/pipeline/operators/reader/parser/parser.h"
#include "ndll/pipeline/operators/reader/parser/caffe.pb.h"

namespace ndll {

class CaffeParser : public Parser {
 public:
  explicit CaffeParser(const OpSpec& spec) :
    Parser(spec) {}

  void Parse(const uint8_t* data, const size_t size, SampleWorkspace* ws) override {
    caffe::Datum datum;
    // NDLL_ENFORCE(datum.ParseFromString(string(reinterpret_cast<const char*>(data), size)));
    NDLL_ENFORCE(datum.ParseFromArray(data, size));

    auto* image = ws->Output<CPUBackend>(0);
    auto* label = ws->Output<CPUBackend>(1);

    // copy label
    label->Resize({1});
    label->mutable_data<int>()[0] = datum.label();

    // copy image
    image->Resize({datum.data().size()});
    std::memcpy(image->mutable_data<uint8_t>(), datum.data().data(),
                datum.data().size()*sizeof(uint8_t));
  }
};

};  // namespace ndll

#endif  // NDLL_PIPELINE_PARSER_CAFFE_PARSER_H_
