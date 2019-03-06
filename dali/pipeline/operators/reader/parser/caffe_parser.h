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

#ifndef DALI_PIPELINE_OPERATORS_READER_PARSER_CAFFE_PARSER_H_
#define DALI_PIPELINE_OPERATORS_READER_PARSER_CAFFE_PARSER_H_

#include "dali/pipeline/operators/reader/parser/parser.h"
#include "dali/pipeline/operators/reader/parser/caffe.pb.h"

namespace dali {

class CaffeParser : public Parser<Tensor<CPUBackend>> {
 public:
  explicit CaffeParser(const OpSpec& spec) :
    Parser(spec) {}

  void Parse(const Tensor<CPUBackend>& data, SampleWorkspace* ws) override {
    caffe::Datum datum;
    // DALI_ENFORCE(datum.ParseFromString(string(reinterpret_cast<const char*>(data), size)));
    DALI_ENFORCE(datum.ParseFromArray(data.raw_data(), data.size()));

    auto& image = ws->Output<CPUBackend>(0);
    auto& label = ws->Output<CPUBackend>(1);

    // copy label
    label.Resize({1});
    label.mutable_data<int>()[0] = datum.label();

    // copy image
    image.Resize({static_cast<Index>(datum.data().size())});
    std::memcpy(image.mutable_data<uint8_t>(), datum.data().data(),
                datum.data().size()*sizeof(uint8_t));
    image.SetSourceInfo(data.GetSourceInfo());
  }
};

};  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_PARSER_CAFFE_PARSER_H_
