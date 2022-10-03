// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    if (image_available_) {
      bool encoded_data = true;
      int data_size = 0;
      auto& image = ws->Output<CPUBackend>(out_tensors);
      if (datum.has_encoded() && !datum.encoded()) {
        encoded_data = false;
      }
      if (datum.has_data()) {
        data_size = datum.data().size();
      }
      // copy image
      if (encoded_data) {
        image.Resize({static_cast<Index>(data_size)}, DALI_UINT8);
      } else {
        DALI_ENFORCE(datum.height() * datum.width() * datum.channels() == data_size,
                    "The content size of the raw image in LMDB caffe entry doesn't"
                    " match its dimensions");
        image.Resize({datum.height(), datum.width(), datum.channels()}, DALI_UINT8);
      }
      std::memcpy(image.mutable_data<uint8_t>(), datum.data().data(), data_size * sizeof(uint8_t));
      image.SetSourceInfo(data.GetSourceInfo());
      out_tensors++;
    }

    if (label_available_) {
      auto& label = ws->Output<CPUBackend>(out_tensors);
      if (datum.has_label()) {
        // copy label
        label.Resize({1}, DALI_INT32);
        label.mutable_data<int>()[0] = datum.label();
      } else {
        label.Resize({0}, DALI_INT32);
      }
      out_tensors++;
    }
  }

 private:
  bool image_available_;
  bool label_available_;
};

};  // namespace dali

#endif  // DALI_OPERATORS_READER_PARSER_CAFFE_PARSER_H_
