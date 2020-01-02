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

#ifndef DALI_OPERATORS_READER_PARSER_CAFFE_PARSER_H_
#define DALI_OPERATORS_READER_PARSER_CAFFE_PARSER_H_

#include "dali/operators/reader/parser/parser.h"
#include "dali/operators/reader/parser/caffe.pb.h"

namespace dali {

class CaffeParser : public Parser<Tensor<CPUBackend>> {
 public:
  explicit CaffeParser(const OpSpec& spec) :
    Parser(spec),
    image_available_(spec.GetArgument<bool>("image_available")),
    label_available_(spec.GetArgument<bool>("label_available")) {}

  void Parse(const Tensor<CPUBackend>& data, SampleWorkspace* ws) override {
    caffe::Datum datum;
    int out_tensors = 0;
    DALI_ENFORCE(datum.ParseFromArray(data.raw_data(), data.size()),
      make_string("Error while parsing Caffe file: ", data.GetSourceInfo(),
                  " (raw data length: ", data.size(), " bytes)."));

    if (image_available_ && datum.has_data()) {
      bool encoded_data = true;
      auto& image = ws->Output<CPUBackend>(out_tensors);
      if (datum.has_encoded() && !datum.encoded()) {
        encoded_data = false;
      }
      // copy image
      if (encoded_data) {
        image.Resize({static_cast<Index>(datum.data().size())});
      } else {
        image.Resize({datum.height(), datum.width(), datum.channels()});
      }
      std::memcpy(image.mutable_data<uint8_t>(), datum.data().data(),
                  datum.data().size()*sizeof(uint8_t));
      image.SetSourceInfo(data.GetSourceInfo());
      out_tensors++;
    }

    if (label_available_ && datum.has_label()) {
      auto& label = ws->Output<CPUBackend>(out_tensors);

      // copy label
      label.Resize({1});
      label.mutable_data<int>()[0] = datum.label();
      out_tensors++;
    }
  }

 private:
  bool image_available_;
  bool label_available_;
};

};  // namespace dali

#endif  // DALI_OPERATORS_READER_PARSER_CAFFE_PARSER_H_
